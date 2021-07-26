import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim

from model_label import WarnConv_digits_Label, WarnMLP_Label
from solver import Convex, BBSL, NLLSL
from load_data import load_numpy_data, data_loader, multi_data_loader, shift_trainset
import utils

parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    help="Name of the dataset: [amazon|digits].",
    type=str,
    choices=["amazon", "digits"],
    default="amazon",
)
parser.add_argument("--result_path", help="Where to save results.", type=str, default="./results")
parser.add_argument("--data_path", help="Where to find the data.", type=str, default="./datasets")
parser.add_argument("--lr", help="Learning rate.", type=float, default=0.5)
parser.add_argument(
    "--mu",
    help="Hyperparameter of the coefficient for the domain adversarial loss.",
    type=float,
    default=1e-3,
)
parser.add_argument(
    "--gp_coef", help="Coefficent of gradient penality loss(mu * gp_coef).", type=float, default=1.0
)
parser.add_argument(
    "--sem_coef", help="Coefficent of semantic loss(mu * sem_coef).", type=float, default=1.0
)
parser.add_argument("--gamma", help="Inverse temperature hyperparameter.", type=float, default=1.0)
parser.add_argument("--epoch", help="Number of training epochs.", type=int, default=50)
parser.add_argument("--batch_size", help="Batch size during training.", type=int, default=20)
parser.add_argument("--cuda", help="Which cuda device to use.", type=int, default=0)
parser.add_argument("--seed", help="Random seed.", type=int, default=0)
parser.add_argument(
    "--alpha_solver",
    help="solver type used to resolve alpha.",
    choices=["bbsl", "nllsl"],
    default="nllsl",
)
args = parser.parse_args()

device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size

exp_flags = "lr_{:g}_mu_{:g}_gp_{:g}_sem_{:g}_seed_{:d}_{}_shift_labelled".format(
    args.lr, args.mu, args.gp_coef, args.sem_coef, args.seed, args.alpha_solver
)
result_path = os.path.join(args.result_path, args.name, exp_flags)
if not os.path.exists(result_path):
    os.makedirs(result_path)


logger = utils.get_logger(os.path.join(result_path, "log_{}.log".format(exp_flags)))
logger.info("Hyperparameter setting = %s" % args)

# Set random number seed.
np.random.seed(args.seed)
torch.manual_seed(args.seed)

#################### Loading the datasets ####################
print(torch.__version__)
time_start = time.time()

data_names, train_insts, train_labels, test_insts, test_labels, configs = load_numpy_data(
    args.name, args.data_path, logger
)
# number of srouce classes,
num_classes_dict = {"digits": 10, "office_home": 65, "amazon": 2}
num_src_classes = num_classes_dict[args.name]
# configs["mode"] = args.mode

### this is the required feature space dimension (digits 2304, office 100, amazon review: 100)
feature_dim_dict = {
    #'digits': 1024,
    "digits": 100,
    "office_home": 100,
    "amazon": 100,
}
configs["feauture_dim"] = feature_dim_dict[args.name]
configs["mu"] = args.mu
configs["gp_coef"] = args.gp_coef
configs["sem_coef"] = args.sem_coef
configs["gamma"] = args.gamma
configs["num_src_domains"] = len(data_names) - 1
configs["num_src_classes"] = num_src_classes
num_datasets = len(data_names)

logger.info("Time used to process the %s = %g seconds." % (args.name, time.time() - time_start))
logger.info("-" * 100)

test_results = {}
np_test_results = np.zeros(num_datasets)

if args.name == "amazon":
    # for Amazon
    src_shift_labels = [0]
    src_drop_ratios = [0.5]
elif args.name == "digits":
    # for digits
    src_shift_labels = [5, 6, 7, 8, 9]
    src_drop_ratios = [0.5, 0.5, 0.5, 0.5, 0.5]


#################### Model ####################
num_src_domains = configs["num_src_domains"]

logger.info("Model setting = %s." % configs)

#################### Train ####################
lambda_list = np.zeros([num_datasets, num_src_domains, args.epoch])

for tar_dom_idx, tar_dom_name in enumerate(data_names):
    # collect source data names from full data names except for target data name
    src_data_names = [name for name in data_names if name != tar_dom_name]
    # display sources v.s. target
    logger.info("*" * 100)
    logger.info(
        "*  Source domains: [{}], target domain: [{}] ".format(
            "/".join(src_data_names), tar_dom_name
        )
    )
    logger.info("*" * 100)

    # Build source instances
    source_insts = []
    source_labels = []
    for j in range(num_datasets):
        if j != tar_dom_idx:
            train_x_temps, train_y_temps = shift_trainset(
                train_insts[j].astype(np.float32),
                train_labels[j].astype(np.int64),
                src_shift_labels,
                src_drop_ratios,
            )
            source_insts.append(train_x_temps)
            source_labels.append(train_y_temps)

    # Build target instances
    # construct a random drop 90% of the data (save a limited target lablled data)
    target_x_temps = train_insts[tar_dom_idx].astype(np.float32)
    target_y_temps = train_labels[tar_dom_idx].astype(np.int64)
    label_idx = np.arange(len(target_y_temps))
    np.random.shuffle(label_idx)
    num_drop = int(np.ceil(len(target_y_temps) * 0.9))
    dropped_idx = label_idx[:num_drop]
    # tar_test_idx = label_idx[:num_drop]
    # the dropped 90% data are treated as test set (since they are unseen)
    tar_test_insts = np.take(target_x_temps, dropped_idx, axis=0)
    tar_test_labels = np.take(target_y_temps, dropped_idx, axis=0)
    target_insts = np.delete(target_x_temps, dropped_idx, axis=0)
    target_labels = np.delete(target_y_temps, dropped_idx, axis=0)
    logger.info("#samples in target domain for train = {}".format(target_labels.shape[0]))
    logger.info("#samples in target domain for test = {}".format(len(tar_test_labels)))

    # Compute ground truth source/ target label distribution (normalized)
    src_true = np.zeros([num_src_domains, num_src_classes])
    tar_true = np.zeros([num_src_classes])
    for tsk in range(num_src_domains):
        for j in range(num_src_classes):
            src_true[tsk, j] = np.count_nonzero(source_labels[tsk] == j)
        src_true[tsk, :] = src_true[tsk, :] / len(source_labels[tsk])

    for j in range(num_src_classes):
        tar_true[j] = np.count_nonzero(target_labels == j)
    tar_true = tar_true / len(target_labels)

    # Model
    if args.name in ["amazon", "office_home"]:  # MLP
        # model = DarnMLP(configs).to(device)
        # model = WarnMLP(configs).to(device)
        model = WarnMLP_Label(configs).to(device)
    elif args.name == "digits":  # ConvNet
        # model = DarnConv(configs).to(device)
        # model = WarnConv(configs).to(device)
        model = WarnConv_digits_Label(configs).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    # Training phase
    model.train()
    time_start = time.time()

    # defining lambda and alpha (global)
    task_lambda = np.ones([num_src_domains]) / num_src_domains

    # alpha can be directly estimated
    task_alpha = np.ones([num_src_domains, num_src_classes], dtype=np.float32)

    for tsk in range(num_src_domains):
        task_alpha[tsk, :] = tar_true * 1.0 / src_true[tsk, :]

    L2_regularization = 1

    global_step = 0
    for epoch_idx in range(args.epoch):
        model.train()

        running_loss = 0.0
        loss_acc = np.zeros(num_src_domains)
        train_loader = multi_data_loader(source_insts, source_labels, batch_size)
        lam_cuda, alpha_cuda = (
            torch.FloatTensor(task_lambda).to(device),
            torch.FloatTensor(task_alpha).to(device),
        )
        src_true_cuda = torch.FloatTensor(src_true).to(device)

        for batch_idx, (xs, ys) in enumerate(
            tqdm(train_loader, desc="Epoch {}...".format(epoch_idx + 1))
        ):
            global_step += 1

            for j in range(num_src_domains):
                xs[j] = torch.tensor(xs[j], requires_grad=False).to(device)
                ys[j] = torch.tensor(ys[j], requires_grad=False).to(device)

            ridx = np.random.choice(target_insts.shape[0], batch_size)
            tinputs = target_insts[ridx, :]
            tinputs = torch.tensor(tinputs, requires_grad=False).to(device)
            toutpouts = target_labels[ridx]
            toutpouts = torch.tensor(toutpouts, requires_grad=False).to(device)

            optimizer.zero_grad()

            train_loss, convex_loss, losses_tuple = model(
                xs, ys, tinputs, toutpouts, alpha_cuda, tar_true
            )
            cls_losses, domain_losses, domain_gradient_losses, src_semantic_losses = losses_tuple

            # lambda alpha based loss
            lambda_loss = torch.sum(train_loss * lam_cuda)

            with torch.no_grad():
                # convert to L2 optimization mode
                loss_np = convex_loss.cpu().numpy()
                loss_acc += loss_np

            lambda_loss.backward()
            optimizer.step()
            running_loss += lambda_loss.item()

        # updating lambda after each epoch
        loss_acc /= batch_idx + 1

        if epoch_idx > 1 and epoch_idx % 3 == 0:
            L2_regularization = np.max(loss_acc)
            task_lambda_temp = Convex(loss_acc, L2_regularization)
            task_lambda = 0.8 * task_lambda + 0.2 * task_lambda_temp
        else:
            logger.info("Epoch[{}/{}], no updates for lambda!".format(epoch_idx + 1, args.epoch))

        for src_dom_idx, src_dom in enumerate(src_data_names):
            logger.info("alpha[{}] = {}".format(src_dom, task_alpha[src_dom_idx, :]))

        # display
        lambdas_in_str = [
            " {}:{:.6f} ".format(dom_name, task_lambda[idx])
            for idx, dom_name in enumerate(src_data_names)
        ]
        logger.info(
            "Epoch[{}/{}], Lambda=[{}]".format(epoch_idx + 1, args.epoch, ",".join(lambdas_in_str))
        )
        lambda_list[tar_dom_idx, :, epoch_idx] = task_lambda
        logger.info(
            "Epoch[{}/{}], running_loss = {:.4f}".format(epoch_idx + 1, args.epoch, running_loss)
        )
        logger.info("Finish training in {:.6g} seconds".format(time.time() - time_start))

        model.eval()

        # Test (use another hold-out target)
        test_loader = data_loader(tar_test_insts, tar_test_labels, batch_size=1000, shuffle=False)
        test_acc = 0.0
        for xt, yt in test_loader:
            xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(device)
            yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(device)
            preds_labels = torch.argmax(model.inference(xt), 1)
            test_acc += torch.sum(preds_labels == yt).item()
        test_acc /= tar_test_labels.shape[0]
        logger.info(
            "Epoch[{}/{}], test accuracy on [{}] = {:.6g}".format(
                epoch_idx + 1, args.epoch, tar_dom_name, test_acc
            )
        )
        test_results[tar_dom_name] = test_acc
        np_test_results[tar_dom_idx] = test_acc
    logger.info("All test accuracies: ")
    logger.info(test_results)

    # Save results to files
    with open(os.path.join(result_path, "test_{}.txt".format(exp_flags)), "w") as test_file:
        for tar_dom_name, test_acc in test_results.items():
            test_file.write("{} = {:.6g}\n".format(tar_dom_name, test_acc))

    logger.info("Finish {}_{}".format(exp_flags, tar_dom_name))
    logger.info("*" * 100)

logger.info("All finished!")
