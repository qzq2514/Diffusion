

## 各数据集参数配置及效果

训练:

```
CUDA_VISIBLE_DEVICES=0 python train_solver.py --data_name "Flower102"
```

在config.yaml中各个数据集使用默认的Training Setting，每个数据集特有的配置见config.yaml下的Train_Data.

生成效果如下:

| 数据集                                                       | 去噪过程可视化                                               | 最终去噪效果                                                 |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Mnist                                                        | <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221225095139_mnist_latest_progress.jpg" alt="20221225095139_mnist_latest_progress" style="zoom:150%;" /> | <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221225095139_mnist_latest_final.jpg" alt="20221225095139_mnist_latest_final" style="zoom:150%;" /> |
| Fashion_Mnist                                                | <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221225200037_fashion_mnist_latest_progress.jpg" alt="20221225200037_fashion_mnist_latest_progress" style="zoom:150%;" /> | <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221225200037_fashion_mnist_latest_final.jpg" alt="20221225200037_fashion_mnist_latest_final" style="zoom:150%;" /> |
| Cifar10                                                      | <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221225200604_cifar10_latest_progress.jpg" alt="20221225200604_cifar10_latest_progress" style="zoom: 150%;" /> | <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221225200604_cifar10_latest_final.jpg" alt="20221225200604_cifar10_latest_final" style="zoom: 150%;" /> |
| [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | ![20221226111542_flower102_size64_progress](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226111542_flower102_size64_progress.jpg) | ![20221226111542_flower102_size64_final](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226111542_flower102_size64_final.jpg) |
| StyleGAN2人脸数据                                            | ![20221226130201_StyleGAN_face_size128_progress](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226130201_StyleGAN_face_size128_progress.jpg) | ![20221226130201_StyleGAN_face_size128_final](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226130201_StyleGAN_face_size128_final.jpg) |

上述训练数据集和已经训练好的模型放在[这里](https://drive.google.com/drive/folders/1yInbcK5pq9qMhkl9ES3QIZr69LXkeeQK).

## Interpolate

使用[原DDPM](https://arxiv.org/pdf/2006.11239.pdf) 论文中提出的插值的方法进行测试，默认配置使用 `扩散步数=500，插值系数从0~1均匀采样10次`

其中第一行和最后一行分别为两个原始的插值图片，中间11行为插值的结果，并且第二行和倒数第二行插值系数分别为0和1，可以看成是对原始两个插值图片的重构。

| Minst                                                        | Fashion_Mnist                                                |
| ------------------------------------------------------------ | ------------------------------------------------------------ |
| <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226174548_mnist_latest_interpolate.jpg" alt="20221226174548_mnist_latest_interpolate" style="zoom:150%;" /> | <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226175012_fashion_mnist_latest_interpolate.jpg" alt="20221226175012_fashion_mnist_latest_interpolate" style="zoom:155%;" /> |
| **Cifar10**                                                  | **Flower102**                                                |
| <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226174822_cifar10_latest_interpolate.jpg" alt="20221226174822_cifar10_latest_interpolate" style="zoom: 150%;" /> | <img src="/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226173958_flower102_size64_858_interpolate.jpg" alt="20221226173958_flower102_size64_858_interpolate" style="zoom:120%;" /> |
| **StyleGAN_Face#1**                                          | **StyleGAN_Face#2**                                          |
| ![20221226174754_StyleGAN_face_size128_225_interpolate](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226174754_StyleGAN_face_size128_225_interpolate.jpg) | ![20221226181429_StyleGAN_face_size128_225_interpolate](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/20221226181429_StyleGAN_face_size128_225_interpolate.jpg) |

感觉插值的结果并非是平缓的，而是中间会有一个比较“陡峭”的突变过程.....暂时没找到这种突变的解释~




## 注意事项与其他对比实验

- 单通道较简单的数据集(如Mnist, Fashion_Mnist等)可以直接使用Linear的Beta采样，与Cosine采样无大区别
- 3通道相对复杂的数据集(如Cifar10, Flower102, StyleGAN_Face等)的Beta采样最好使用[Improved DDPM](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)中提出的Cosine schedule,不然会导致最终生成的图片偏白。
- L1损失和L2损失对比，L1损失下生成效果会更加"尖锐"有时候会稍显乱+脏，而L2损失则显得更加平滑。

上述提到的L1/L2损失、Linear Schedule/Cosine Schedule的效果对比如下

| 配置      | L1 loss、Cosine Beta                                         | L2 loss、Cosine Beta                                         | L1 loss、Linear Beta                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Cifar10   | ![cifar10_epoch27](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/cifar10_epoch27.jpg) | ![cifar10_L2loss_epoch27](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/cifar10_L2loss_epoch27.jpg) | ![cifar10_linearBeta_epoch27_](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/cifar10_linearBeta_epoch27_.jpg) |
| Flower102 | ![Flower102_epoch_520](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/Flower102_epoch_520.jpg) | ![Flower102_L2loss_epoch_520](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/Flower102_L2loss_epoch_520.jpg) | ![FLower102_linearBeta_epoch_520](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/FLower102_linearBeta_epoch_520.jpg) |

  

## 其他Tricks

此外又使用[lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch) 的代码验证了一下其中比较重要配置(如EMA、P2 loss、Clip_denoised)的必要性，对比效果如下:

| 配置            | EMA✅                      P2 loss✅              Clip denoised✅ | EMA❌                       P2 loss✅             Clip denoised✅ |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 效果(26K steps) | ![StyleGANFace_128-sample-130](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/StyleGANFace_128-sample-130.png) | ![StyleGANFace_128_noEMA-sample-130](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/StyleGANFace_128_noEMA-sample-130.png) |
| **配置**        | **EMA✅                        P2 loss❌            Clip denoised✅** | **EMA✅                      P2 loss✅              Clip denoised❌** |
| 效果(26K steps) | ![StyleGANFace_128_noP2weight-sample-130](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/StyleGANFace_128_noP2weight-sample-130.png) | ![StyleGANFace_128_noClipDenoised-sample-130](/Users/georgeqi/Code/Diffusion/ddpm_mnist_based/images/StyleGANFace_128_noClipDenoised-sample-130.png) |

上面看来无明显的大差别，所以在本仓库中未使用EMA和P2 Loss，但是用了Clip denoised。

其他重要的配置则使用[lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch)原代码仓库中默认的:

```
loss_type='l1'     beta_schedule='cosine'     objective='pred_noise'   self_condition=False
```




## 参考文献/Code:

[DDPM](https://arxiv.org/pdf/2006.11239.pdf)  
[Improved DDPM](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)  
[What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)  
[lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)  
[zoubohao/DenoisingDiffusionProbabilityModel-ddpm](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-)  
[LianShuaiLong/CV_Applications](https://github.com/LianShuaiLong/CV_Applications)  
[yiyixuxu/denoising-diffusion-flax](https://github.com/yiyixuxu/denoising-diffusion-flax)   
