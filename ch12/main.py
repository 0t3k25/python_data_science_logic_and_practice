from google.colab import drive

drive.mount("/content/drive")

# MNISTファイル読み込み
import os
import struct
import numpy as np


# 読み込み関数
def load_mnist(path, kind="train"):
    """Load MNIST data from `path`"""
    labels_path = os.path.join(path, "%s-labels-idx1-ubyte" % kind)
    images_path = os.path.join(path, "%s-images-idx3-ubyte" % kind)

    with open(labels_path, "rb") as lbpath:
        magic, n = struct.unpack(">II", lbpath.read(8))
        labels = np.fromfile(lbpath, dtype=np.uint8)

    with open(images_path, "rb") as imgpath:
        magic, num, rows, cols = struct.unpack(">IIII", imgpath.read(16))
        images = np.fromfile(imgpath, dtype=np.uint8).reshape(len(labels), 784)
        images = ((images / 255.0) - 0.5) * 2

    return images, labels


# データ数表示
X_train, y_train = load_mnist("/content/drive/My Drive", kind="train")
print("Rows: %d, columns: %d" % (X_train.shape[0], X_train.shape[1]))

X_test, y_test = load_mnist("/content/drive/My Drive", kind="t10k")
print("Rows: %d, columns: %d" % (X_test.shape[0], X_test.shape[1]))
