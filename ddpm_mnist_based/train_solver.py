# coding=UTF-8
# This Python file uses the following encoding: utf-8

import os
import cv2
import yaml
import torch
import argparse
import numpy as np
from tqdm import tqdm
from datetime import datetime

import utils
from Diffusion_model import Diffusion
from models.UNet_arch import UNetModel
from models.Unet_Luci_arch import Unet as Unet_Luci


parser = argparse.ArgumentParser()
parser.add_argument('--data_name', required=True, choices=["Mnist", "Fashion_Mnist", "Cifar10", "Flower102",
                                                           "StyleGAN_Face"], help='image data prepared to train')
opt = parser.parse_args()

# 训练数据相关配置
data_name = opt.data_name
config = yaml.load(open("./config.yaml", "r"), Loader=yaml.SafeLoader)
image_size = config["Train_Data"][data_name]["image_size"]
img_channel = config["Train_Data"][data_name]["image_channel"]
batch_size = config["Train_Data"][data_name]["batch_size"]
sample_num = config["Train_Data"][data_name]["sample_num"]
epoch_num = config["Train_Data"][data_name]["train_cpoch"]
pth_save_name = config["Train_Data"][data_name]["pth_save_name"]
train_result_save_dir = utils.check_dir(config["Train_Data"][data_name]["save_dir"])
data_path = config["Train_Data"][data_name]["data_path"]
model_save_path = os.path.join(train_result_save_dir, pth_save_name)


# 训练模型相关配置
learning_rate = config["Train_Setting"]["learning_rate"]
timestep_num = config["Train_Setting"]["timestep_num"]
adam_beta1, adam_beta2 = config["Train_Setting"]["Adam_beta1"], config["Train_Setting"]["Adam_beta2"]

os.system("cp {} {}".format("./config.yaml", train_result_save_dir))

print("*"*40)
print("Dataset_name: ", data_name)
print("Exp_save_dir: ", train_result_save_dir)
print("Image shape: {}x{}x{}".format(image_size, image_size, img_channel))
print("Batch_size: ", batch_size)
print("*"*40)


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
unet = UNetModel(in_channels=img_channel, out_channels=img_channel).to(device)
# unet = Unet_Luci(dim=64, dim_mults=(1, 2, 4, 8), channels=img_channel).to(device)
optimizer = torch.optim.Adam(params=unet.parameters(), lr=learning_rate, betas=(adam_beta1, adam_beta2))
diffusion_model = Diffusion(model=unet, loss_type=config["Train_Setting"]["diffusion_loss_type"],
                            beta_shedule=config["Train_Setting"]["diffusion_beta_shedule"], device=device)
dataloader = utils.load_dataset(data_name, config)


def train():
    resume_epoch = 0
    if os.path.exists(model_save_path):
        resume_info = torch.load(model_save_path)
        resume_epoch = resume_info["epoch"] + 1
        utils.load_model(unet, resume_info["unet_model"])
        print("Loading pretrain model from {}".format(model_save_path))
    else:
        print("Dont loading any pretrain model, train from scratch !!! ")

    for epoch in range(resume_epoch, epoch_num):
        running_loss = 0
        progress_bar = tqdm(dataloader, leave=False)
        for data_batch in progress_bar:  # 返回元素值在(0,1)之间的[N,C,H,W]的数据
            optimizer.zero_grad()
            data_batch = utils.normlize(data_batch).to(device)
            time_sequence = torch.randint(0, timestep_num, (batch_size,), device=device).long()
            loss = diffusion_model.train_loss(data_batch, time_sequence)
            progress_bar.set_description("Datatime:{}, Epoch[{}/{}]] loss:{:.4f}".
                                         format(datetime.strftime(datetime.now(), "%Y %m %d %H:%M:%S"),
                                                epoch, epoch_num, loss))
            running_loss += loss.cpu().item()
            loss.backward()
            optimizer.step()

        # 计算每个epoch的平均损失
        running_loss /= len(dataloader)
        tqdm.write("Datatime:{}, Mean loss for epoch {}: {}".
                   format(datetime.strftime(datetime.now(), "%Y %m %d %H:%M:%S"), epoch, running_loss))

        # 每训练一个epoch后测试生成结果
        # 返回每一轮去噪声后的生成结果，所以sample_imgs是一个长度为轮数的list,
        # 其内每个元素是每一轮去噪后的结果[b,c,h,w],元素值在(-1,1)范围或者使用ImageNet归一化的
        sample_tensors = diffusion_model.sample(image_size=image_size, batch_size=sample_num,
                                                channels=img_channel, clip_denoised=True)
        show_img = utils.make_grid_npy(sample_tensors[-1], nrow=int(np.sqrt(sample_num)))
        cv2.imwrite("{}/epoch_{}.jpg".format(train_result_save_dir, epoch), show_img)

        print("Latest model was saved to {}!".format(model_save_path))
        save_dict = {"epoch": epoch,
                     "unet_model": unet.state_dict()}
        torch.save(save_dict, model_save_path)


if __name__ == '__main__':
    train()

