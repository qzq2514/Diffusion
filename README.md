# 理论推导
具体理论公式的推导见[Georgeqi'Blob](https://qzq2514.github.io/2022/12/20/Diffusion%E5%AD%A6%E4%B9%A02-%E7%90%86%E8%AE%BA%E6%8E%A8%E5%AF%BC/)


# Diffusion.pytorch

基于pytorch 实现的Diffusion模型  

# 训练和测试

**训练:**  
运行train_solver.py文件并且通过data_name指定数据集，具体可以见config.yaml文件。

```
CUDA_VISIBLE_DEVICES=0 python train_solver.py --data_name "Flower102"
```

在config.yaml中各个数据集使用默认的Training Setting，每个数据集特有的配置见config.yaml下的Train_Data.

**测试:**  
下载训练好的模型，在run_test.sh指定model_path、image_size/image_channel、sample_num等一些常规参数，此外有一些重要参数如下:  

- test_mode: 选择测试的模式，目前是支持random_gen(随机生成)和interpolate(插值效果)两种方式

- training_timestep_num: 加噪步数(一般是和训练时候采用的步数相同，默认1000)

- sample_info: 采样方式，默认是两种形式:

  - "DDPM_1000_0p0": 常规的DDPM采样，且去噪迭代次数为1000(也可以设定为小于1000的数，此时会默认采用间隔步数采样)，第三个参数"0p0"在这个采样方式下用不到
  -  "DDIM_50_0p0": 加速的DDIM采样，采样步数为50，eta参数为0p0。

  上述两种形式都以"_"为界限分别指定"采样方式"、"迭代步数"、"eta"，可自行修改参数查看相关效果

其他注意点:在使用test_mode="interpolate"时候，最好使用sample_info="DDPM_1000_0p0"

# 生成效果

| 数据集                                                       | 去噪过程可视化                                               | 最终去噪效果    | 插值                                              |
| ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |------------------------------------------------------------ |
| Mnist                                                        | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221225095139_mnist_latest_progress.jpg" alt="20221225095139_mnist_latest_progress" style="zoom:2000%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221225095139_mnist_latest_final.jpg" alt="20221225095139_mnist_latest_final" style="zoom:2000%;" /> |<img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226174548_mnist_latest_interpolate.jpg" alt="20221226174548_mnist_latest_interpolate" style="zoom:150%;" />  |
| Fashion_Mnist                                                | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221225200037_fashion_mnist_latest_progress.jpg" alt="20221225200037_fashion_mnist_latest_progress" style="zoom:1500%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221225200037_fashion_mnist_latest_final.jpg" alt="20221225200037_fashion_mnist_latest_final" style="zoom:1500%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226175012_fashion_mnist_latest_interpolate.jpg" alt="20221226175012_fashion_mnist_latest_interpolate" style="zoom:155%;" />|
| Cifar10                                                      | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221225200604_cifar10_latest_progress.jpg" alt="20221225200604_cifar10_latest_progress" style="zoom:200%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221225200604_cifar10_latest_final.jpg" alt="20221225200604_cifar10_latest_final" style="zoom:200%;" /> |<img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226174822_cifar10_latest_interpolate.jpg" alt="20221226174822_cifar10_latest_interpolate" style="zoom:1500%;" /> |
| [Flower102](https://www.robots.ox.ac.uk/~vgg/data/flowers/102/) | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226111542_flower102_size64_progress.jpg" alt="20221226111542_flower102_size64_progress" style="zoom:100%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226111542_flower102_size64_final.jpg" alt="20221226111542_flower102_size64_final" style="zoom:100%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226173958_flower102_size64_858_interpolate.jpg" alt="20221226173958_flower102_size64_858_interpolate" style="zoom:120%;" />|
| StyleGAN2人脸                                                | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226130201_StyleGAN_face_size128_progress.jpg" alt="20221226130201_StyleGAN_face_size128_progress" style="zoom:100%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226130201_StyleGAN_face_size128_final.jpg" alt="20221226130201_StyleGAN_face_size128_final" style="zoom:100%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/20221226174754_StyleGAN_face_size128_225_interpolate.jpg" alt="20221226174754_StyleGAN_face_size128_225_interpolate" style="zoom:101%;" />|

上述训练数据集和已经训练好的模型放在[这里](https://drive.google.com/drive/folders/1yInbcK5pq9qMhkl9ES3QIZr69LXkeeQK).

- 去噪过程可视化中，如果在采样step内均匀采样时间戳，会发现前面的去噪过程过于缓慢而后面会突然“有效果”，所以这里对于时间戳采用了一个“先粗后细”的采样trick用于可视化(具体见test_solver.py的test_random_generate函数)。
- 上述插值方法采用[原DDPM](https://arxiv.org/pdf/2006.11239.pdf)论文中提出，本仓库采用其默认配置: `扩散步数=500，插值系数从0~1均匀采样11次`
其中第一行和最后一行分别为两个原始的插值图片，中间11行为插值的结果，并且第二行和倒数第二行插值系数分别为0和1，可以看成是对原始两个插值图片的重构。
实验下来：感觉插值的结果并非是平缓的，而是中间会有一个比较“陡峭”的突变过程.....暂时没找到这种突变的解释~



# 效果提升

这里简单实验和讨论了在训练DDPM过程中使用的损失函数(L1或L2损失)，并且实验了[Improved DDPM](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) 中提到的Cosine Beta Schedule带来的效果提升:

- 单通道较简单的数据集(如Mnist, Fashion_Mnist等)可以直接使用Linear的Beta采样，与Cosine采样无大区别
- 3通道相对复杂的数据集(如Cifar10, Flower102, StyleGAN_Face等)的Beta采样最好使用[Improved DDPM](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)中提出的Cosine schedule,不然会导致最终生成的图片偏白。
- L1损失和L2损失对比，L2损失下生成效果会更加"尖锐"有时候会稍显乱+脏，而L1损失则显得更加平滑。

上述提到的L1/L2损失、Linear Beta Schedule/Cosine Beta Schedule的效果对比如下:

| 配置      | L1 loss、Cosine Beta                                         | L2 loss、Cosine Beta                                         | L1 loss、Linear Beta                                         |
| --------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Cifar10   | ![](https://github.com/qzq2514/Diffusion/blob/main/images/cifar10_epoch27.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/cifar10_L2loss_epoch27.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/cifar10_linearBeta_epoch27.jpg) |
| Flower102 | ![](https://github.com/qzq2514/Diffusion/blob/main/images/Flower102_epoch_520.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/Flower102_L2loss_epoch_520.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/FLower102_linearBeta_epoch_520.jpg) |

  

# 采样加速

推理加速主要使用[DDIM](https://arxiv.org/abs/2010.02502)  中的算法

| 配置        | 50步                                                         | 100步                                                        | 200步                                                        | 500步                                                        | 800步                                                        | 1000步                                                       |
| ----------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ | ------------------------------------------------------------ |
| DDPM        | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231153752_StyleGAN_face_size128_DDPM_50_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231153902_StyleGAN_face_size128_DDPM_100_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231153924_StyleGAN_face_size128_DDPM_200_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154000_StyleGAN_face_size128_DDPM_500_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154126_StyleGAN_face_size128_DDPM_800_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154212_StyleGAN_face_size128_DDPM_1000_0p0_final.jpg) |
| DDIM(eta=1) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154809_StyleGAN_face_size128_DDIM_50_1p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154843_StyleGAN_face_size128_DDIM_100_1p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154953_StyleGAN_face_size128_DDIM_200_1p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231155049_StyleGAN_face_size128_DDIM_500_1p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231155129_StyleGAN_face_size128_DDIM_800_1p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231155212_StyleGAN_face_size128_DDIM_1000_1p0_final.jpg) |
| DDIM(eta=0) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154412_StyleGAN_face_size128_DDIM_50_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154734_StyleGAN_face_size128_DDIM_100_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154454_StyleGAN_face_size128_DDIM_200_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154518_StyleGAN_face_size128_DDIM_500_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154641_StyleGAN_face_size128_DDIM_800_0p0_final.jpg) | ![](https://github.com/qzq2514/Diffusion/blob/main/images/DDIM/20221231154551_StyleGAN_face_size128_DDIM_1000_0p0_final.jpg) |

上面的DDPM在采样步数小于训练步数(1000)的时候，使用的是[Improved DDPM](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf) 中的间隔步长采样。可以看到上面的结果：

- 常规的DDPM采样结果在步数比较小时候基本仍然为纯噪声，虽然在步数逐渐增大(如500和800步)后会逐渐出现生成主体，但是仍然有一层噪声，直到采样步数等于训练时候的去噪步数(1000步)才能生成比较好完全无噪声的图像。
- 使用DDIM算法时候即便在部署很小(50步)时，也已经具有了比较好的生成效果，生成速度能提升至少20倍。
- 从当前实验结果看下来，在DDIM采样中eta=0和eta=1好像并无太大的效果差异。



# 其他Tricks

此外又使用[lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch) 的代码验证了一下其中比较重要配置(如EMA、P2 loss、Clip_denoised)的必要性，对比效果如下:

| 配置            | EMA✅                      P2 loss✅              Clip denoised✅ | EMA❌                       P2 loss✅             Clip denoised✅ |
| --------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| 效果(26K steps) | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/StyleGANFace_128-sample-130.png" alt="StyleGANFace_128-sample-130" style="zoom:101%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/StyleGANFace_128_noEMA-sample-130.png" alt="StyleGANFace_128_noEMA-sample-130" style="zoom:101%;" /> |
| **配置**        | **EMA✅                        P2 loss❌            Clip denoised✅** | **EMA✅                      P2 loss✅              Clip denoised❌** |
| 效果(26K steps) | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/StyleGANFace_128_noP2weight-sample-130.png" alt="StyleGANFace_128_noP2weight-sample-130" style="zoom:101%;" /> | <img src="https://github.com/qzq2514/Diffusion/blob/main/images/StyleGANFace_128_noClipDenoised-sample-130.png" alt="StyleGANFace_128_noClipDenoised-sample-130" style="zoom:101%;" /> |

上面看来无明显的大差别，所以在本仓库中未使用EMA和P2 Loss，但是用了Clip denoised。

其他重要的配置则使用[lucidrains](https://github.com/lucidrains/denoising-diffusion-pytorch)原代码仓库中默认的:

```
loss_type='l1'     beta_schedule='cosine'     objective='pred_noise'   self_condition=False
```




# 参考文献/Code:

[DDPM](https://arxiv.org/pdf/2006.11239.pdf)  
[Improved DDPM](http://proceedings.mlr.press/v139/nichol21a/nichol21a.pdf)  
[What are Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/)  
[lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch)  
[zoubohao/DenoisingDiffusionProbabilityModel-ddpm](https://github.com/zoubohao/DenoisingDiffusionProbabilityModel-ddpm-)  
[LianShuaiLong/CV_Applications](https://github.com/LianShuaiLong/CV_Applications)  
[yiyixuxu/denoising-diffusion-flax](https://github.com/yiyixuxu/denoising-diffusion-flax)   
