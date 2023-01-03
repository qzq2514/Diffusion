import os
import cv2
import torch
import argparse
import numpy as np
from datetime import datetime

# coding=UTF-8
# This Python file uses the following encoding: utf-8

import utils
from Diffusion_model import Diffusion
from models.UNet_arch import UNetModel
from models.Unet_Luci_arch import Unet as Unet_Luci


parser = argparse.ArgumentParser()
parser.add_argument('--model_path', required=True, type=str, help='image data prepared to train')
parser.add_argument('--test_mode', choices=["random_gen", "interpolate"], help='image data prepared to train')
parser.add_argument('--sample_info', type=str, default="False", help='information for sampling')
parser.add_argument('--image_size', required=True, type=int, help='image data prepared to train')
parser.add_argument('--image_channel', required=True, type=int, help='image data prepared to train')
parser.add_argument('--sample_num', required=True, type=int, help='image data prepared to train')
parser.add_argument('--training_timestep_num', type=int, default=1000, help='image data prepared to train')
parser.add_argument('--data_name',  choices=["Mnist", "Fashion_Mnist", "Cifar10", "Flower102",
                                             "StyleGAN_Face"], help='image data prepared to train')
parser.add_argument('--data_path',  type=str, help='path to dataset')

opt = parser.parse_args()
model_path = opt.model_path
image_size = opt.image_size
image_channel = opt.image_channel
sample_num = opt.sample_num
training_timestep_num = opt.training_timestep_num
data_name = opt.data_name
data_path = opt.data_path
sample_info = opt.sample_info
sample_timestep_num = int(sample_info.split("_")[1])
print("*"*40)
print("Data Name:", data_name)
print("Training timestep_num:", training_timestep_num)
print("Sample size: {}x{}x{}x{}".format(sample_num, image_size, image_size, image_channel))
print("Test mode:", opt.test_mode)
print("Sample info:", sample_info)

print("*"*40)


device = torch.device("cuda") if torch.cuda.is_available() else "cpu"
unet = UNetModel(in_channels=image_channel, out_channels=image_channel).to(device)
# unet = Unet_Luci(dim=64, dim_mults=(1, 2, 4, 8), channels=image_channel).to(device)
unet.eval()

diffusion_model = Diffusion(model=unet, timestep_num=training_timestep_num, device=device)
if os.path.exists(model_path):
    resume_info = torch.load(model_path)
    utils.load_model(unet, resume_info["unet_model"])
    print("Loading pretrain model from {}".format(model_path))
else:
    raise RuntimeError("Model {} doesnt exist!")


def test_random_generate():
    image_save_path = "{}_{}".format(datetime.strftime(datetime.now(), "%Y%m%d%H%M%S"),
                                         os.path.splitext(os.path.basename(model_path))[0])

    # 获取去噪生成过程中每一步的结果,最终len(sample_tensors)=sample_timestep_num
    sample_tensors = diffusion_model.sample(image_size=image_size, batch_size=sample_num,
                                            channels=image_channel, clip_denoised=True, sample_info=sample_info)
    # 1.显示中间不断去噪的过程(上面sample函数返回的是从T-->0的逐渐去噪过程，并且一开始去噪的慢，后面去噪的快)
    # 因为去噪嫌慢后快，所以均匀采样可视化效果不好，指数型采样可视化效果更好
    # time_milestones = np.linspace(0, training_timestep_num-1, num=int(np.sqrt(sample_num)))
    time_milestones = np.power(np.linspace(0, 1, num=int(np.sqrt(sample_num))*2), 0.3)*sample_timestep_num
    time_milestones[-1] -= 1
    sample_tensors_progress = []
    for time_step in time_milestones:
        cur_tensor = sample_tensors[int(time_step)][:int(np.sqrt(sample_num))]
        sample_tensors_progress.append(cur_tensor)
    sample_tensors_progress = torch.transpose(torch.cat(sample_tensors_progress, dim=0), dim0=2, dim1=3)
    show_img_progress = utils.make_grid_npy(sample_tensors_progress, nrow=int(np.sqrt(sample_num)))
    cv2.imwrite("{}_{}_progress.jpg".format(image_save_path, sample_info),
                np.transpose(show_img_progress, axes=[1, 0, 2]))

    # 2.保存最终的结果
    final_tensors = torch.transpose(sample_tensors[-1], dim0=2, dim1=3)
    show_img_final = utils.make_grid_npy(final_tensors, nrow=int(np.sqrt(sample_num)))
    cv2.imwrite("{}_{}_final.jpg".format(image_save_path, sample_info),
                np.transpose(show_img_final, axes=[1, 0, 2]))

    print("Image generated by ddpm is save to {}*.jpg!".format(image_save_path))


@torch.no_grad()
# 对两个归一化后的输入图像x_start1, x_start2在t时间戳内的含噪隐变量进行插值，插值系数为lamdba
def interpolate(x_start1, x_start2, diffusion_step, lamdba=0.5):
    from tqdm import tqdm
    batch_size = x_start1.shape[0]
    assert diffusion_step < sample_timestep_num, "interpolate step must less than sample_timestep_num"

    t_batched = torch.stack([torch.tensor(diffusion_step, device=device, dtype=torch.long)] * batch_size)
    x_t1, x_t2 = diffusion_model.q_sample(x_start1, t_batched), diffusion_model.q_sample(x_start2, t_batched)

    inter_img = (1 - lamdba) * x_t1 + lamdba * x_t2
    times = torch.linspace(0, diffusion_step, steps=sample_timestep_num//2)
    for time in tqdm(list(reversed(times.int().tolist())), desc='Interpolation with lamdba [{}] and '
                                              'diffusion_step [{}].....'.format(lamdba, diffusion_step),
                  total=diffusion_step):
        time_sequence = torch.full((batch_size,), time, device=device, dtype=torch.long)
        inter_img = diffusion_model.p_sample(inter_img, time_sequence, clip_denoised=True)
    return inter_img


def test_interpolate():
    interpolate_ratios_num = 11  # 从0~1s均匀采样多少次-插值系数的种类(默认0,0.1,0.2,....0.9,1.0)
    diffusion_step = sample_timestep_num // 2  # 前向扩散的次数,默认使用全部采样步数的一半
    from torch.utils import data
    from Dataset import MnistDataset, Cifar10_Dataset, FashionMnistDataset, CommonDataset
    transform = utils.get_img_transformer(image_size)
    if data_name == "Mnist":
        dataloader = data.DataLoader(MnistDataset(data_path, transformer=transform),
                                  batch_size=sample_num, shuffle=True)
    elif data_name == "Fashion_Mnist":
        dataloader = data.DataLoader(FashionMnistDataset(data_path, transformer=transform),
                                  batch_size=sample_num, shuffle=True)
    elif data_name == "Cifar10":
        dataloader = data.DataLoader(Cifar10_Dataset(data_path, transformer=transform),
                                  batch_size=sample_num, shuffle=True)
    else:  # 其他都是默认从文件夹下读取所有图片
        dataloader = data.DataLoader(CommonDataset(data_path, transformer=transform),
                                  batch_size=sample_num, shuffle=True)
    dataloader = utils.dataloader_itera(dataloader)

    # 从数据集中随机获取两组样本准备进行插值
    x_start1, x_start2 = utils.normlize(next(dataloader)).to(device), utils.normlize(next(dataloader)).to(device)

    show_results = [x_start1.cpu()]
    for lamdba in np.linspace(0.0, 1.0, num=interpolate_ratios_num):
        inter_img = interpolate(x_start1, x_start2, diffusion_step, lamdba=lamdba)
        show_results.append(inter_img.cpu())
    show_results.append(x_start2.cpu())
    show_img = torch.cat(show_results, dim=0)
    inter_img = utils.make_grid_npy(show_img, nrow=sample_num)
    image_save_path = "{}_{}".format(datetime.strftime(datetime.now(), "%Y%m%d%H%M%S"),
                                     os.path.splitext(os.path.basename(model_path))[0])
    cv2.imwrite(image_save_path+"_interpolate.jpg", inter_img)


if __name__ == '__main__':
    if opt.test_mode == "random_gen":
        test_random_generate()
    elif opt.test_mode == "interpolate":
        test_interpolate()
