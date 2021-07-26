import os

import numpy as np
from scipy.sparse import coo_matrix


def load_numpy_data(name, data_path, logger):

    if name == "amazon":

        data_names = ["books", "dvd", "electronics", "kitchen"]
        input_dim = 5000  # Number of features to be used in the experiment
        num_trains = 2000

        # Load & process the amazon dataset
        amazon = np.load(os.path.join(data_path, "%s.npz" % name))
        amazon_xx = coo_matrix(
            (amazon["xx_data"], (amazon["xx_col"], amazon["xx_row"])),
            shape=amazon["xx_shape"][::-1],
        ).tocsc()
        amazon_xx = amazon_xx[:, :input_dim]
        amazon_yy = amazon["yy"]
        amazon_yy = (amazon_yy + 1) / 2  # from {-1, 1} to {0, 1}
        amazon_offset = amazon["offset"].flatten()  # starting indices of the four domains

        # Partition the data into four domains and for each domain partition the data set into training and test set
        train_insts, train_labels, num_insts, test_insts, test_labels = [], [], [], [], []
        for i, dataset in enumerate(data_names):
            train_insts.append(amazon_xx[amazon_offset[i] : amazon_offset[i + 1]])
            train_labels.append(np.squeeze(amazon_yy[amazon_offset[i] : amazon_offset[i + 1]]))
            logger.info(
                "%s with %d instances." % (dataset, amazon_offset[i + 1] - amazon_offset[i])
            )
            num_insts.append(amazon_offset[i + 1] - amazon_offset[i])
            # Random shuffle
            ridx = np.arange(num_insts[i])
            np.random.shuffle(ridx)
            test_insts.append(train_insts[i][ridx[num_trains:]].todense())
            test_labels.append(train_labels[i][ridx[num_trains:]].ravel())
            train_insts[i] = train_insts[i][ridx[:num_trains]].todense()
            train_labels[i] = train_labels[i][ridx[:num_trains]].ravel()

        configs = {
            "input_dim": input_dim,
            "hidden_layers": [1000, 500, 100],
            "num_classes": 2,
            "drop_rate": 0.7,
        }

    elif name == "digits":

        # data_names = ["mnist", "mnist_m", "svhn", "synth"]
        # num_trains = 20000
        # num_trains = 10000
        # num_tests = 9000

        data_names = ["mnist", "svhn", "synth", "usps"]
        num_trains = 7000
        num_tests = 2000

        input_dim = 2304  # number of features after CovNet

        train_insts, train_labels, test_insts, test_labels = [], [], [], []
        for dataset in data_names:
            data = np.load(os.path.join(data_path, dataset, "%s.npz" % dataset))
            logger.info(
                "%s with %d training and %d test instances"
                % (dataset, data["train_x"].shape[0], data["test_x"].shape[0])
            )
            # Shuffle and get training and test data
            ridx = np.arange(data["train_x"].shape[0])
            np.random.shuffle(ridx)
            train_insts.append(data["train_x"][ridx[:num_trains]])
            train_labels.append(data["train_y"][ridx[:num_trains]])

            ridx = np.arange(data["test_x"].shape[0])
            np.random.shuffle(ridx)
            test_insts.append(data["test_x"][ridx[:num_tests]])
            test_labels.append(data["test_y"][ridx[:num_tests]])

        configs = {
            "input_dim": input_dim,
            "channels": 3,
            "conv_layers": [64, 128, 256],
            #    "cls_fc_layers": [2048, 1024],
            #    "dom_fc_layers": [2048, 2048],
            "cls_fc_layers": [512, 100],
            "dom_fc_layers": [256],
            "num_classes": 10,
            "drop_rate": 0.0,
        }

    elif name == "office_home":

        data_names = ["Art", "Clipart", "Product", "Real_World"]
        num_trains = 2000

        train_insts, train_labels, num_insts, test_insts, test_labels = [], [], [], [], []
        for i, dataset in enumerate(data_names):
            data_file = os.path.join(data_path, "office_home_%s.npz" % dataset.lower())
            data = np.load(data_file)

            train_insts.append(data["x"])
            train_labels.append(np.squeeze(data["y"]))
            num_insts.append(train_insts[i].shape[0])

            logger.info("%s with %d instances." % (dataset, num_insts[i]))
            # Random shuffle.
            ridx = np.arange(num_insts[i])
            np.random.shuffle(ridx)

            test_insts.append(train_insts[i][ridx[num_trains:]])
            test_labels.append(train_labels[i][ridx[num_trains:]])

            train_insts[i] = train_insts[i][ridx[:num_trains]]
            train_labels[i] = train_labels[i][ridx[:num_trains]]

        configs = {
            "input_dim": 2048,
            "hidden_layers": [1000, 500, 100],
            "num_classes": 65,
            "drop_rate": 0.7,
        }

    else:

        raise ValueError("Unknown dataset.")

    return data_names, train_insts, train_labels, test_insts, test_labels, configs


def data_loader(inputs, targets, batch_size, shuffle=True):
    assert inputs.shape[0] == targets.shape[0]
    inputs_size = inputs.shape[0]
    if shuffle:
        random_order = np.arange(inputs_size)
        np.random.shuffle(random_order)
        inputs, targets = inputs[random_order, :], targets[random_order]
    num_blocks = int(inputs_size / batch_size)
    for i in range(num_blocks):
        yield inputs[i * batch_size : (i + 1) * batch_size, :], targets[
            i * batch_size : (i + 1) * batch_size
        ]
    if num_blocks * batch_size != inputs_size:
        yield inputs[num_blocks * batch_size :, :], targets[num_blocks * batch_size :]


def multi_data_loader(inputs, targets, batch_size, shuffle=True):
    """
    Both inputs and targets are list of numpy arrays, containing instances and labels from multiple sources.
    """
    assert len(inputs) == len(targets)
    input_sizes = [data.shape[0] for data in inputs]
    max_input_size = max(input_sizes)
    num_domains = len(inputs)
    if shuffle:
        for i in range(num_domains):
            r_order = np.arange(input_sizes[i])
            np.random.shuffle(r_order)
            inputs[i], targets[i] = inputs[i][r_order], targets[i][r_order]
    num_blocks = int(max_input_size / batch_size)
    for j in range(num_blocks):
        xs, ys = [], []
        for i in range(num_domains):
            ridx = np.random.choice(input_sizes[i], batch_size)
            xs.append(inputs[i][ridx])
            ys.append(targets[i][ridx])
        yield xs, ys


def loader_gen(loader, mode="inf"):
    # https://github.com/pytorch/pytorch/issues/1917#issuecomment-479482530
    while True:
        for images, targets in loader:
            yield images, targets
        if mode != "inf":
            break


def shift_trainset(xx, yy, lables_list, drop_ratios):
    assert len(lables_list) == len(
        drop_ratios
    ), "'labels_list' and 'drop_rates' should have same length!"
    assert lables_list is not None
    assert (
        all(map(lambda r: r > 0 and r < 1, drop_ratios)) == True
    ), "please gurantee 0<drop_rate<1 !"

    all_dropped_idx = []
    for label, ratio in zip(lables_list, drop_ratios):
        label_idx = np.where(yy == label)[0]
        np.random.shuffle(label_idx)

        num_total = np.count_nonzero(yy == label)
        num_drop = int(np.ceil(num_total * ratio))

        dropped_idx = label_idx[:num_drop]
        all_dropped_idx.append(dropped_idx)
        # print("num total of class {} = {}, drop {:.0%}, drop {} smaples".format(label, num_total, ratio, num_drop))

    all_dropped_idx = np.concatenate(all_dropped_idx)

    return np.delete(xx, all_dropped_idx, axis=0), np.delete(yy, all_dropped_idx, axis=0)


def load_shifted_data(name, data_path, logger):

    if name == "amazon":

        data_names = ["books", "dvd", "electronics", "kitchen"]
        input_dim = 5000  # Number of features to be used in the experiment
        num_trains = 2000

        # Load & process the amazon dataset
        amazon = np.load(os.path.join(data_path, "%s.npz" % name))
        amazon_xx = coo_matrix(
            (amazon["xx_data"], (amazon["xx_col"], amazon["xx_row"])),
            shape=amazon["xx_shape"][::-1],
        ).tocsc()
        amazon_xx = amazon_xx[:, :input_dim]
        amazon_yy = amazon["yy"]
        amazon_yy = (amazon_yy + 1) / 2  # from {-1, 1} to {0, 1}
        amazon_offset = amazon["offset"].flatten()  # starting indices of the four domains

        # Partition the data into four domains and for each domain partition the data set into training and test set
        train_insts, train_labels, num_insts, test_insts, test_labels = [], [], [], [], []
        for i, dataset in enumerate(data_names):
            xx = amazon_xx[amazon_offset[i] : amazon_offset[i + 1]].todense()
            yy = np.squeeze(amazon_yy[amazon_offset[i] : amazon_offset[i + 1]].ravel())
            logger.info(
                "%s with %d instances (original)."
                % (dataset, amazon_offset[i + 1] - amazon_offset[i])
            )
            # drop samples in certain classes
            shift_labels, drop_ratios = None, None
            if dataset == "books":
                shift_labels = [1]
                drop_ratios = [0.2]
            elif dataset == "dvd":
                shift_labels = [0]
                drop_ratios = [0.2]
            elif dataset == "electronics":
                shift_labels = [1]
                drop_ratios = [0.4]
            elif dataset == "kitchen":
                shift_labels = [0]
                drop_ratios = [0.4]
            if shift_labels is not None:
                xx, yy = shift_trainset(xx, yy, shift_labels, drop_ratios)

            logger.info("%s with %d instances after label shift." % (dataset, xx.shape[0]))
            # Random shuffle
            ridx = np.arange(xx.shape[0])
            np.random.shuffle(ridx)
            train_insts.append(xx[:num_trains])
            train_labels.append(yy[:num_trains])
            test_insts.append(xx[num_trains:])
            test_labels.append(yy[num_trains:])

        configs = {
            "input_dim": input_dim,
            "hidden_layers": [1000, 500, 100],
            "num_classes": 2,
            "drop_rate": 0.7,
        }

    elif name == "digits":

        data_names = ["mnist", "mnist_m", "svhn", "synth"]
        # num_trains = 20000
        num_trains = 10000
        num_tests = 9000
        input_dim = 2304  # number of features after CovNet

        train_insts, train_labels, test_insts, test_labels = [], [], [], []
        for dataset in data_names:
            data = np.load(os.path.join(data_path, dataset, "%s.npz" % dataset))
            logger.info(
                "%s with %d training and %d test instances"
                % (dataset, data["train_x"].shape[0], data["test_x"].shape[0])
            )
            xx, yy = data["train_x"], data["train_y"]
            # drop samples in certain classes
            shift_labels, drop_ratios = None, None
            if dataset == "mnist":
                shift_labels = [4, 5, 6, 7, 8, 9]
                drop_ratios = [0.2, 0.2, 0.2, 0.4, 0.4, 0.4]
            elif dataset == "mnist_m":
                shift_labels = [0, 1, 2, 3, 7, 8, 9]
                drop_ratios = [0.4, 0.4, 0.4, 0.4, 0.2, 0.2, 0.2]
            elif dataset == "svhn":
                shift_labels = None
                drop_ratios = None
            elif dataset == "synth":
                shift_labels = [0, 1, 2, 3, 4, 5, 6]
                drop_ratios = [0.2, 0.2, 0.2, 0.2, 0.4, 0.4, 0.4]
            if shift_labels is not None:
                xx, yy = shift_trainset(xx, yy, shift_labels, drop_ratios)

            num_remain = xx.shape[0]
            logger.info(
                "%s with %d training and %d test instances after label shift"
                % (dataset, xx.shape[0], data["test_x"].shape[0])
            )
            # Shuffle and get training and test data
            ridx = np.arange(xx.shape[0])
            np.random.shuffle(ridx)
            train_insts.append(xx[ridx[:num_trains]])
            train_labels.append(yy[ridx[:num_trains]])

            ridx = np.arange(data["test_x"].shape[0])
            np.random.shuffle(ridx)
            test_insts.append(data["test_x"][ridx[:num_tests]])
            test_labels.append(data["test_y"][ridx[:num_tests]])

        configs = {
            "input_dim": input_dim,
            "channels": 3,
            "conv_layers": [64, 128, 256],
            "cls_fc_layers": [2048, 1024],
            "dom_fc_layers": [2048, 2048],
            "num_classes": 10,
            "drop_rate": 0.0,
        }

    elif name == "office_home":
        # TODO: shift for offce_home

        data_names = ["Art", "Clipart", "Product", "Real_World"]
        num_trains = 2000

        train_insts, train_labels, num_insts, test_insts, test_labels = [], [], [], [], []
        for i, dataset in enumerate(data_names):
            data_file = os.path.join(data_path, "office_home_%s.npz" % dataset.lower())
            data = np.load(data_file)

            train_insts.append(data["x"])
            train_labels.append(np.squeeze(data["y"]))
            num_insts.append(train_insts[i].shape[0])

            logger.info("%s with %d instances." % (dataset, num_insts[i]))
            # Random shuffle.
            ridx = np.arange(num_insts[i])
            np.random.shuffle(ridx)

            test_insts.append(train_insts[i][ridx[num_trains:]])
            test_labels.append(train_labels[i][ridx[num_trains:]])

            train_insts[i] = train_insts[i][ridx[:num_trains]]
            train_labels[i] = train_labels[i][ridx[:num_trains]]

        configs = {
            "input_dim": 2048,
            "hidden_layers": [1000, 500, 100],
            "num_classes": 65,
            "drop_rate": 0.7,
        }

    else:

        raise ValueError("Unknown dataset.")

    return data_names, train_insts, train_labels, test_insts, test_labels, configs
