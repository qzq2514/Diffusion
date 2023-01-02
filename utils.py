import os
import torch
import numpy as np
from copy import deepcopy
from torch.utils import data
from torchvision import transforms
from torchvision.utils import make_grid
from Dataset import MnistDataset, Cifar10_Dataset, FashionMnistDataset, CommonDataset


def normlize(tensor):
    # tensor:0~1之间的torch张量
    # 返回-1~1范围的张量
    return tensor * 2 - 1


def unnormlize(tensor):
    # tensor:-1~1之间的torch张量
    # 返回0~1范围的张量
    return (tensor + 1) * 0.5


# 将dataloader变成可迭代直接获取元素的generator
def dataloader_itera(dataloader):
    while True:
        for data in dataloader:
            yield data


# 元素从(-1,1)反归一化到(0,1)然后拼接成(c, h*b1, w*b2)再转成(H,W,C)形式
def make_grid_npy(tensors, nrow):
    tensors = make_grid(unnormlize(tensors), nrow=nrow)
    tensors = np.clip(np.array(tensors * 255, dtype=np.uint8), 0, 255)
    image_npy = np.moveaxis(tensors, 0, -1)  # [C, H, W] --> [H, W, C]
    return image_npy


def get_img_transformer(img_size=32):
    trans = transforms.Compose(
        [transforms.Resize((img_size, img_size)),
         transforms.RandomHorizontalFlip(),
         transforms.CenterCrop((img_size, img_size)),
         transforms.ToTensor(),  # 元素值归一化到(0,1), shape上多通道:HWC->CHW  单通道:HW-1HW
         ])
    return trans


def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    return dir_path


def load_dataset(data_name, config):
    data_path = config["Train_Data"][data_name]["data_path"]
    image_size = config["Train_Data"][data_name]["image_size"]
    batch_size = config["Train_Data"][data_name]["batch_size"]
    transform = get_img_transformer(image_size)

    if data_name == "Mnist":
        dataloader = data.DataLoader(MnistDataset(data_path, transformer=transform),
                                     batch_size=batch_size, shuffle=True, drop_last=True)
    elif data_name == "Fashion_Mnist":
        dataloader = data.DataLoader(FashionMnistDataset(data_path, transformer=transform),
                                     batch_size=batch_size, shuffle=True, drop_last=True)
    elif data_name == "Cifar10":
        dataloader = data.DataLoader(Cifar10_Dataset(data_path, transformer=transform),
                                     batch_size=batch_size, shuffle=True, drop_last=True)
    else:  # 其他都是默认从文件夹下读取所有图片
        dataloader = data.DataLoader(CommonDataset(data_path, transformer=transform),
                                     batch_size=batch_size, shuffle=True, drop_last=True)

    return dataloader


def load_model(network, pretrained_dict):

    model_dict_org = network.state_dict()
    pretrained_dict_org = deepcopy(pretrained_dict)
    try:
        # pretrain模型和network所有的参数完全吻合！
        network.load_state_dict(pretrained_dict)
        print("Loaded model Precisely!!")
    except:
        try:
            # pretrain模型包含network所有的参数并且还多一些额外的参数, 那么就直接从pretrain模型中模型中取出network所有的参数就行了
            pretrained_dict = {}
            excessive_dict = {}
            for key, value in pretrained_dict_org.items():
                if key in model_dict_org and model_dict_org[key].size() == value.size():
                    pretrained_dict[key] = value
                else:
                    excessive_dict[key] = value
            network.load_state_dict(pretrained_dict)
            print('Pretrained checkpoint has excessive layers; Only loading layers that are used in network')
            print("Following parameters in checkpoint will not be loaded into network~")
            for key, value in excessive_dict.items():
                print("{}: {}".format(key, value.shape))
        except:
            # pretrain模型缺少network中一些参数, 那么以network的state_dict为准, 取出pretrain模型中有的network中的参数
            model_dict = deepcopy(model_dict_org)
            noInitialized_dict = {}
            for key, value in model_dict.items():
                if key in pretrained_dict_org and pretrained_dict_org[key].size() == value.size():
                    model_dict[key] = value
                else:
                    noInitialized_dict[key] = value
            network.load_state_dict(model_dict)
            print('Pretrained checkpoint has fewer layers; Only loading parameters that appear in checkpoint')
            print("Following parameters in network will not be initialized~")
            for key, value in noInitialized_dict.items():
                print("{}: {}".format(key, value.shape))



