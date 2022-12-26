import os
import cv2
import gzip
import torch
import pickle
import numpy as np
from PIL import Image
from torch.utils import data
from torchvision import transforms

import utils


class MnistDataset(data.Dataset):
    # image_pkg_path: mnist.pkl.gz
    def __init__(self, image_pkg_path, transformer=None):
        self.transformer = transformer
        with gzip.open(image_pkg_path, "rb") as f:
            (x_train, y_train), (x_valid, y_valid), (x_test, y_test) = pickle.load(f, encoding="latin-1")

        # 单通道图像，大小28x28
        self.images = np.reshape(np.concatenate([x_train, x_valid, x_test]), newshape=[-1, 28, 28])
        self.labels = np.concatenate([y_train, y_valid, y_test])

    def __getitem__(self, item):
        image = Image.fromarray(self.images[item])
        label = self.labels[item]
        if self.transformer is not None:
            image = self.transformer(image)
        return image
        # return image, label

    def __len__(self):
        return len(self.images)


class FashionMnistDataset(data.Dataset):
    def __init__(self, data_dir, transformer=None):
        self.transformer = transformer
        self.images = []
        self.labels = []
        for data_name in ["training.pt", "test.pt"]:
            image, label = torch.load(os.path.join(data_dir, data_name))
            self.images.extend(image.cpu().numpy())
            self.labels.extend(label.cpu().numpy())
        self.class_mapping = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
                              'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

    def __getitem__(self, item):
        image = Image.fromarray(self.images[item])
        label = self.labels[item]
        if self.transformer is not None:
            image = self.transformer(image)
        return image
        # return image, label

    def __len__(self):
        return len(self.images)


class Cifar10_Dataset(data.Dataset):
    def __init__(self, image_floder, transformer=None):
        self.transformer = transformer

        # 可选:获取图像的类别映射
        with open((os.path.join(image_floder, "batches.meta")), 'rb') as infile:
            label_data = pickle.load(infile, encoding='latin1')
            classes = label_data["label_names"]
        self.class_mapping = {i: _class for i, _class in enumerate(classes)}

        images, self.labels = [], []
        data_names = ["data_batch_{}".format(i) for i in range(1, 6)] + ["test_batch"]
        for data_name in data_names:
            with open(os.path.join(image_floder, data_name), 'rb') as fr:
                entry = pickle.load(fr, encoding='latin1')
                images.append(entry['data'])
                self.labels.extend(entry['labels'] if 'labels' in entry else entry['fine_labels'])

        # 三通道彩图，32x32大小
        images = np.vstack(images).reshape(-1, 3, 32, 32)
        self.images = images.transpose((0, 2, 3, 1))

    def __getitem__(self, item):
        img_npy = cv2.cvtColor(self.images[item], cv2.COLOR_BGR2RGB)

        image = Image.fromarray(img_npy)
        label = self.labels[item]

        if self.transformer is not None:
            image = self.transformer(image)
        return image
        # return image, label

    def __len__(self):
        return len(self.images)


class CommonDataset(data.Dataset):
    # Flower102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
    def __init__(self, data_dir, transformer=None):
        self.transformer = transformer
        self.image_paths = [os.path.join(data_dir, img_name) for img_name in os.listdir(data_dir)]
        # class_mapping官网暂时下载不了

    def __getitem__(self, item):
        image = Image.fromarray(cv2.imread(self.image_paths[item]))
        if self.transformer is not None:
            image = self.transformer(image)
        return image

    def __len__(self):
        return len(self.image_paths)


if __name__ == '__main__':
    transforms = transforms.Compose(
                      [transforms.Resize((512, 512)),
                       transforms.ToTensor(),  # 多通道:HWC->CHW  单通道:HW-1HW
                       # transforms.Normalize((0.5), (0.5))
                       transforms.Normalize((0.406, 0.456, 0.485), (0.225, 0.224, 0.229))  # BGR模式下的imagenet归一化
                       ])

    # 测试Cifar_10
    flower102_data = Flower102("/Users/georgeqi/Code/Diffusion/dataset/Flower102", transformer=transforms)
    for ind, input_img in enumerate(flower102_data):
        print(input_img.shape, torch.min(input_img), torch.max(input_img))
        img = utils.unnormlize(input_img, imagenet_norm=True)
        img = np.clip(np.array(img*255, dtype=np.uint8), 0, 255)
        img = np.moveaxis(img, 0, -1)
        print(img.shape)
        cv2.imshow("img", img)
        cv2.waitKey()

    # 测试Cifar_10
    # cifar10_data = Cifar10_Dataset("/Users/georgeqi/Code/Diffusion/dataset/cifar-10-batches-py",
    #                                transformer=transforms)
    # for ind, input_img in enumerate(cifar10_data):
    #     print(input_img.shape, torch.min(input_img), torch.max(input_img))
    #     img = utils.unnormlize(input_img)
    #     img = np.clip(np.array(img*255, dtype=np.uint8), 0, 255)
    #     img = np.moveaxis(img, 0, -1)
    #     cv2.imshow("img", img)
    #     cv2.waitKey()

    # 测试fashion mnist
    # fashion_mnist_data = FashionMnistDataset("/Users/georgeqi/Code/Diffusion/dataset/FashionMNIST/processed",
    #                                          transformer=transforms)
    # for input_img, input_label in fashion_mnist_data:
    #
    #     img = utils.unnormlize(input_img)
    #     img = np.clip(np.array(img*255, dtype=np.uint8), 0, 255)
    #     img = np.moveaxis(img, 0, -1)
    #     print(input_img.shape, torch.min(input_img), torch.max(input_img),
    #           "label:{}".format(input_label), fashion_mnist_data.class_mapping[input_label])
    #     cv2.imshow("img", img)
    #     cv2.waitKey()

    # 测试Mnist
    # mnist_data = MnistDataset("/Users/georgeqi/Code/Diffusion/dataset/mnist.pkl.gz", transformer=transforms)
    # import torch
    # for ind, input_img in enumerate(mnist_data):
    #     print(input_img.shape, torch.min(input_img), torch.max(input_img))
    #     img = utils.unnormlize(input_img)
    #     img = np.clip(np.array(img*255, dtype=np.uint8), 0, 255)
    #     img = np.moveaxis(img, 0, -1)
    #     cv2.imshow("img", img)
    #     cv2.waitKey()


    # 测试make_grid函数
    # from torchvision.utils import make_grid
    # show_img = make_grid(tensor, nrow=10, normalize=True)
    # show_img = np.clip(np.array(show_img*255, dtype=np.uint8), 0, 255)
    # show_img = np.moveaxis(show_img, 0, -1)
    # print(show_img.shape)
    # cv2.imshow("img", show_img)
    # cv2.waitKey()
