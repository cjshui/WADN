import os
import argparse
import gzip
from tqdm import tqdm
import numpy as np
import scipy.io as sio
from skimage.transform import resize
import glob
import imageio


def mnist_to_np(data_path, train_test):
    if train_test == "train":
        flag = "train"
    elif train_test == "test":
        flag = "t10k"
    files = {
        "x": "{}-images-idx3-ubyte".format(flag),
        "y": "{}-labels-idx1-ubyte".format(flag),
    }
    with gzip.GzipFile(os.path.join(data_path, files["x"] + ".gz")) as gfile:
        x = np.frombuffer(gfile.read(), dtype=np.uint8, offset=16)
        if train_test == "train":
            x = x.reshape((60000, 28, 28, 1))
        elif train_test == "test":
            x = x.reshape((10000, 28, 28, 1))

    with gzip.GzipFile(os.path.join(data_path, files["y"] + ".gz")) as gfile:
        y = np.frombuffer(gfile.read(), dtype=np.uint8, offset=8)

    n_imgs = x.shape[0]
    new_x = np.zeros((n_imgs, 32, 32, 1), dtype=np.uint8)
    for i in tqdm(range(n_imgs)):
        new_x[i, :, :, :] = (resize(x[i, :, :], [32, 32, 1]) * 255).astype(np.uint8)
    x = np.concatenate((new_x,) * 3, axis=3)
    x = np.transpose(x, (0, 3, 1, 2))  # to N,C_in,H,W for PyTorch
    return x, y


def mnist_m_to_np(data_path, train_test):

    labels = {}
    label_file = os.path.join(data_path, "mnist_m_%s_labels.txt" % train_test)
    with open(label_file) as f:
        for line in f:
            key, val = line.split()
            labels[key] = int(val)

    y = []
    x = np.zeros([0, 32, 32, 3], dtype=np.uint8)
    img_files = os.path.join(data_path, "mnist_m_%s/*.png" % train_test)
    for im_path in tqdm(glob.glob(img_files)):
        img_file = im_path.split("/")[-1]
        y.append(labels[img_file])
        im = imageio.imread(im_path)
        im = np.expand_dims(im, axis=0)
        x = np.concatenate([x, im], axis=0)

    y = np.array(y, dtype=np.uint8)

    return np.transpose(x, (0, 3, 1, 2)), y


def svhn_to_np(data_path, train_test):

    data_file = os.path.join(data_path, "%s_32x32.mat" % train_test)
    loaded_mat = sio.loadmat(data_file)
    x = loaded_mat["X"]
    y = loaded_mat["y"].squeeze()
    np.place(y, y == 10, 0)
    x = np.transpose(x, (3, 2, 0, 1))  # to N,C_in,H,W for PyTorch
    return x, y


def synth_to_np(data_path, train_test):

    # loaded_mat = sio.loadmat(data_dir + 'synth_' + file_name + '_32x32_small.mat')  # small test
    data_file = os.path.join(data_path, "synth_%s_32x32.mat" % train_test)
    loaded_mat = sio.loadmat(data_file)
    x = loaded_mat["X"]
    y = loaded_mat["y"].squeeze()
    x = np.transpose(x, (3, 2, 0, 1))  # to N,C_in,H,W for PyTorch
    return x, y


def usps_to_np(data_path, train_test):
    import h5py

    with h5py.File(os.path.join(data_path, "usps.h5"), "r") as hf:
        x_y = hf.get(train_test)

        x = x_y.get("data")[:]
        y = x_y.get("target")[:]

        n_imgs = x.shape[0]
        new_x = np.zeros((n_imgs, 32, 32, 1), dtype=np.uint8)
        for i in tqdm(range(n_imgs), desc="preprocessing {}".format(train_test)):
            x_i = np.expand_dims(np.reshape(x[i, :], (16, 16)), axis=-1)
            new_x[i, :, :, :] = (resize(x_i, [32, 32, 1]) * 255).astype(np.uint8)

        new_x = np.concatenate((new_x,) * 3, axis=-1)
        new_x = np.transpose(new_x, (0, 3, 1, 2))
        return new_x, y


parser = argparse.ArgumentParser()
parser.add_argument(
    "--name",
    help="Name of the dataset: [mnist|mnist_m|svhn|synth|usps].",
    type=str,
    choices=["mnist", "mnist_m", "svhn", "synth", "usps"],
    default="mnist",
)
parser.add_argument("--data_path", help="Where to find the data.", type=str, default="./datasets")
args = parser.parse_args()

data_path = os.path.join(args.data_path, args.name)

if args.name == "mnist":
    test_x, test_y = mnist_to_np(data_path, "test")
    train_x, train_y = mnist_to_np(data_path, "train")
elif args.name == "mnist_m":
    test_x, test_y = mnist_m_to_np(data_path, "test")
    train_x, train_y = mnist_m_to_np(data_path, "train")
elif args.name == "svhn":
    test_x, test_y = svhn_to_np(data_path, "test")
    train_x, train_y = svhn_to_np(data_path, "train")
elif args.name == "synth":
    test_x, test_y = synth_to_np(data_path, "test")
    train_x, train_y = synth_to_np(data_path, "train")
elif args.name == "usps":
    test_x, test_y = usps_to_np(data_path, "test")
    train_x, train_y = usps_to_np(data_path, "train")
else:
    raise NotImplementedError("Unknown data.")

np.savez(
    "{}/{}.npz".format(data_path, args.name),
    train_x=train_x,
    train_y=train_y,
    test_x=test_x,
    test_y=test_y,
)
