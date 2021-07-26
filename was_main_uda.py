import os
import time
import argparse
from tqdm import tqdm

import numpy as np
import torch
import torch.optim as optim

from model import WarnConv, WarnMLP, WarnConv_digits
from solver import Convex, BBSL, NLLSL
from load_data import (
    load_numpy_data,
    load_shifted_data,
    data_loader,
    multi_data_loader,
    shift_trainset,
)
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
    default=1e-2,
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
parser.add_argument("--data_shift", help="use shifted data or not", action="store_true")
args = parser.parse_args()

device = torch.device("cuda:%d" % args.cuda if torch.cuda.is_available() else "cpu")
batch_size = args.batch_size

exp_flags = "lr_{:g}_mu_{:g}_gp_{:g}_sem_{:g}_seed_{:d}_{}_{}".format(
    args.lr,
    args.mu,
    args.gp_coef,
    args.sem_coef,
    args.seed,
    args.alpha_solver,
    "shift" if args.data_shift else "noshift",
)
result_path = os.path.join(
    args.result_path,
    args.name,
    exp_flags
    #    args.method,
    #    args.mode
)
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
    # 'digits':2304,
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

#################### Model ####################
num_src_domains = configs["num_src_domains"]

logger.info("Model setting = %s." % configs)

if args.name == "amazon":
    # for amazon
    src_shift_labels = [0]
    src_drop_ratios = [0.5]
elif args.name == "digits":
    src_shift_labels = [5, 6, 7, 8, 9]
    # src_drop_ratios = [0.4, 0.4, 0.4, 0.4, 0.4]
    src_drop_ratios = [0.5, 0.5, 0.5, 0.5, 0.5]
else:
    logger.info("{} isn't supportted in this setting!".format(args.name))

#################### Train ####################
# we have two parameter lambda and alpha

lambda_list = np.zeros([num_datasets, num_src_domains, args.epoch])
alpha_list = np.zeros([num_datasets, num_src_domains, num_src_classes, args.epoch])

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
            if not args.data_shift:
                source_insts.append(train_insts[j].astype(np.float32))
                source_labels.append(train_labels[j].astype(np.int64))
            else:
                ## apply same label shift on all source domains
                train_x_temps, train_y_temps = shift_trainset(
                    train_insts[j].astype(np.float32),
                    train_labels[j].astype(np.int64),
                    src_shift_labels,
                    src_drop_ratios,
                )
                source_insts.append(train_x_temps)
                source_labels.append(train_y_temps)

    # Build target instances
    target_insts = train_insts[tar_dom_idx].astype(np.float32)
    target_labels = train_labels[tar_dom_idx].astype(np.int64)

    # Compute ground truth source label distribution (normalized)
    src_true = np.zeros([num_src_domains, num_src_classes])
    for tsk in range(num_src_domains):
        for j in range(num_src_classes):
            src_true[tsk, j] = np.count_nonzero(source_labels[tsk] == j)
        src_true[tsk, :] = src_true[tsk, :] / len(source_labels[tsk])

    # Model
    if args.name in ["amazon"]:  # MLP
        model = WarnMLP(configs).to(device)
    elif args.name == "digits":  # ConvNet
        model = WarnConv_digits(configs).to(device)

    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    time_start = time.time()

    # defining lambda and alpha (global)
    task_lambda = np.ones([num_src_domains]) / num_src_domains
    task_alpha = np.ones([num_src_domains, num_src_classes], dtype=np.float32)

    L2_regularization = 1

    global_step = 0
    for epoch_idx in range(args.epoch):
        # train mode
        model.train()

        # estimated y label distribution
        tar_y_estimated = np.dot(task_lambda, np.multiply(task_alpha, src_true))

        # define the confusion matrix, source taget prediction output distribution
        C = np.zeros([num_src_classes, num_src_classes, num_src_domains])
        tar_pred = np.zeros([num_src_classes])

        running_loss = 0.0
        # loss_acc (local)

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

            optimizer.zero_grad()

            train_loss, C_batch, tar_pred_batch, convex_loss, losses_tuple = model(
                xs, ys, tinputs, alpha_cuda, src_true_cuda
            )
            cls_losses, domain_losses, domain_gradient_losses, src_semantic_losses = losses_tuple

            ## updating C, src_true, tar_pred
            C += C_batch
            tar_pred += tar_pred_batch

            # lambda alpha based loss
            lambda_loss = torch.sum(train_loss * lam_cuda)

            with torch.no_grad():
                # convert to L2 optimization mode
                loss_np = convex_loss.cpu().numpy()
                loss_acc += loss_np

            lambda_loss.backward()
            optimizer.step()

            running_loss += lambda_loss.item()

        # updating lamda after each eopch
        loss_acc /= batch_idx + 1
        if args.name == "digits":
            START_EPOCH = 5
        elif args.name == "amazon":
            START_EPOCH = 1
        # update lambda
        if args.name == "digits":
            L2_regularization = np.sum(loss_acc)
        if epoch_idx > START_EPOCH and epoch_idx % 1 == 0:
            task_lambda_temp = Convex(loss_acc, L2_regularization)
            task_lambda = 0.8 * task_lambda + 0.2 * task_lambda_temp
            logger.info(
                "Epoch[{}/{}], update lambda with moving average!".format(epoch_idx + 1, args.epoch)
            )
        else:
            logger.info("Epoch[{}/{}], no updates for lambda!".format(epoch_idx + 1, args.epoch))

        # updating alpha after each epoch
        # IMPORTANT! in the multi-source partial DA, (Cao, ECCV 2018)
        # alpha = tar_pred /np.max(tar_pred) (they did not define lambda)
        if epoch_idx > 1 and epoch_idx % 1 == 0:
            task_alpha_temp = np.ones([num_src_domains, num_src_classes], dtype=np.float32)
            tar_pred = tar_pred / np.sum(tar_pred)
            for src_dom_idx in range(num_src_domains):
                alpha_s = task_alpha[src_dom_idx, :]
                Con_s = C[:, :, src_dom_idx]
                Con_s = Con_s / np.sum(Con_s)
                src_true_s = src_true[src_dom_idx, :]
                if args.alpha_solver == "bbsl":
                    task_alpha_temp[src_dom_idx, :] = BBSL(Con_s, tar_pred, src_true_s)
                elif args.alpha_solver == "nllsl":
                    task_alpha_temp[src_dom_idx, :] = NLLSL(Con_s, tar_pred, src_true_s)
            task_alpha = 0.8 * task_alpha + 0.2 * task_alpha_temp
            logger.info(
                "Epoch[{}/{}], update alpha with moving average!".format(epoch_idx + 1, args.epoch)
            )
        else:
            logger.info("Epoch[{}/{}], no updates for alpha!".format(epoch_idx + 1, args.epoch))
        for src_dom_idx, src_dom in enumerate(src_data_names):
            logger.info(
                "Epoch[{}/{}], alpha[{}] = {}".format(
                    epoch_idx + 1, args.epoch, src_dom, task_alpha[src_dom_idx, :]
                )
            )

        # display
        lambdas_in_str = [
            " {}:{:.6f} ".format(dom_name, task_lambda[idx])
            for idx, dom_name in enumerate(src_data_names)
        ]
        logger.info(
            "Epoch[{}/{}], Lambda=[{}]".format(epoch_idx + 1, args.epoch, ",".join(lambdas_in_str))
        )
        lambda_list[tar_dom_idx, :, epoch_idx] = task_lambda
        alpha_list[tar_dom_idx, :, :, epoch_idx] = task_alpha
        logger.info(
            "Epoch[{}/{}], running_loss(sum in epoch) = {:.4f}".format(
                epoch_idx + 1, args.epoch, running_loss
            )
        )
        logger.info(
            "Epoch[{}/{}], convex_loss(avg on epochs) = {}".format(
                epoch_idx + 1, args.epoch, loss_acc
            )
        )
        logger.info("Finish training in {:.6g} seconds".format(time.time() - time_start))

        model.eval()

        # Test (use another hold-out target)
        test_loader = data_loader(
            test_insts[tar_dom_idx], test_labels[tar_dom_idx], batch_size=1000, shuffle=False
        )
        test_acc = 0.0
        for xt, yt in test_loader:
            xt = torch.tensor(xt, requires_grad=False, dtype=torch.float32).to(device)
            yt = torch.tensor(yt, requires_grad=False, dtype=torch.int64).to(device)
            preds_labels = torch.argmax(model.inference(xt), 1)
            test_acc += torch.sum(preds_labels == yt).item()
        test_acc /= test_insts[tar_dom_idx].shape[0]
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
    with open(
        os.path.join(result_path, "test_{}_{}.txt".format(exp_flags, tar_dom_name)), "w"
    ) as test_file:
        for tar_dom_name, test_acc in test_results.items():
            test_file.write("{} = {:.6g}\n".format(tar_dom_name, test_acc))

    logger.info("Finish {}_{}".format(exp_flags, tar_dom_name))
    logger.info("*" * 100)

logger.info("All finished!")
