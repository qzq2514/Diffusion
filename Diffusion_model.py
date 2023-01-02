import math
import torch
import numpy as np
from torch import nn
from tqdm import tqdm
import torch.nn.functional as F


# 定义与beta有关的variace shedule(又称noise shedule)
# DDPM论文中默认从0.0001均匀增加到0.02,并且时间戳总步数默认为1000
def linear_beta_shedule(start=0.0001, end=0.02, timestep_num=1000):
    return torch.linspace(start=start, end=end, steps=timestep_num)


def cosine_beta_schedule(timestep_num, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timestep_num + 1
    x = torch.linspace(0, timestep_num, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timestep_num) + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class Diffusion:
    # timestep_num: 前向去噪声的轮数
    # beta_shedule: 噪声方差的指定方案，这里使用线性增加
    def __init__(self, model,
                 timestep_num=1000,
                 beta_shedule="cosine",
                 loss_type="l1",
                 device="cpu"):
        self.device = device
        self.loss_type = loss_type

        # 制定beta_shedule
        self.timestep_num = timestep_num
        self.model = model

        if beta_shedule == "linear":
            self.betas = linear_beta_shedule(timestep_num=timestep_num)
        elif beta_shedule == "cosine":
            self.betas = cosine_beta_schedule(timestep_num=timestep_num)
        else:
            raise NotImplementedError("Unknow beta shedule~")

        # 计算alpha相关的变量
        self.alphas = 1.0 - self.betas
        self.overline_alphas = torch.cumprod(self.alphas, dim=0)  # \overline \alpha
        self.overline_alphas_prev = F.pad(self.overline_alphas[:-1], (1, 0), value=1.0)  # \overline \alpha_{t-1}
        self.sqrt_overline_alphas = torch.sqrt(self.overline_alphas)  # \sqrt {\overline \alpha}
        self.sqrt_one_minus_overline_alphas = torch.sqrt(1.0 - self.overline_alphas)  # \sqrt { 1 - \overline \alpha}
        # 紧接着的两个是根据公式3从x_t和噪声反推得到x_0时候使用到的系数，稍微推一下就能得到下面的结果了
        self.sqrt_recip_overline_alphas = torch.sqrt(1.0 / self.overline_alphas)
        self.sqrt_recip_overline_alphas_minus_one = torch.sqrt(1.0 / self.overline_alphas - 1)
        # 不知道是在哪里用到的，先算出来再说吧
        self.log_one_minus_overline_alphas = torch.log(1.0 - self.overline_alphas)


        # 计算逆向过程使用到的参数
        # 公式(14-2)中q(x_{t-1} | x_t, x_0)的方差
        self.reverse_variance = self.betas * (1 - self.overline_alphas_prev) / (1 - self.overline_alphas)
        # 对方差求一个log，可能是为了数值上的溢出和截断，本质没有引入原DDPM论文以外的变量
        self.reverse_log_variance_clipped = torch.log(self.reverse_variance.clamp(min=1e-20))
        # q(x_{t-1} | x_t, x_0)的均值系数:
        # 公式(14-2)中x_0的系数
        self.reverse_mean_coef1 = self.betas * torch.sqrt(self.overline_alphas_prev) / (1.0-self.overline_alphas)
        # 公式(14-2)中x_t的系数
        self.reverse_mean_coef2 = torch.sqrt(self.alphas) * (1.0 - self.overline_alphas_prev) / (1 - self.overline_alphas)

    # 从参数para中选择指定时间t对应的元素
    def _extract(self, para, time_sequence, x_shape):
        '''
        :param para: (self.timesteps, )大小
        :param timesteps: (batch_size, )大小
        :param x_shape: 图像的大小，一般是[N, C, H, W]
        :return: [batch_size, 1, 1, 1] 大小, 并且其第i个元素就是para[timesteps[i]]
        '''
        batch_size = time_sequence.shape[0]
        # 在dim=0时候gather可以理解为将每一列中指定的元素取出来
        # 其实就是切片操作，这里直接直接para[time_sequence]也能得到相同的内容
        out = para.to(time_sequence.device).gather(0, time_sequence).float()
        out = out.reshape(batch_size, *((1,) * (len(x_shape) - 1)))  # [batch_size,]-->[batch_size,1,1,1]
        return out

    # 根据公式3和4进行前向过程-diffusion从x_0加噪直接得到x_t:  q(x_t|x_0)
    def q_sample(self, x_0, time_sequence, noise=None):
        if noise is None:
            noise = torch.randn_like(x_0)

        # 计算公式3中的两个系数
        sqrt_overline_alphas_t = self._extract(self.sqrt_overline_alphas, time_sequence, x_0.shape)
        sqrt_one_minus_overline_alphas_t = self._extract(self.sqrt_one_minus_overline_alphas, time_sequence, x_0.shape)
        # 使用公式3计算x_t
        '''
        print(sqrt_overline_alphas_t.shape, x_0.shape, sqrt_one_minus_overline_alphas_t.shape, noise.shape)
        torch.Size([4, 1, 1, 1]) torch.Size([4, 3, 256, 256]) torch.Size([4, 1, 1, 1]) torch.Size([4, 3, 256, 256])
        '''
        #
        x_t = sqrt_overline_alphas_t * x_0 + sqrt_one_minus_overline_alphas_t * noise
        return x_t

    # 根据公式4得到q(x_t|x_0)过程的均值和方差
    def q_mean_variance(self, x_0, time_sequence):
        shape = x_0.shape
        mean_t = self._extract(self.sqrt_overline_alphas, time_sequence, shape) * x_0
        variance_t = self._extract(1 - self.overline_alphas, time_sequence, shape)
        log_variance_t = self._extract(self.log_one_minus_overline_alphas, time_sequence, shape)
        return mean_t, variance_t, log_variance_t

    # 根据公式(14-2)计算q(x_{t-1}|x_t,x_0)的均值和方差
    def q_reverse_mean_variance(self, x_0, x_t, time_sequence):
        shape = x_0.shape
        reverse_mean_t = self._extract(self.reverse_mean_coef1, time_sequence, shape) * x_0 \
                         + self._extract(self.reverse_mean_coef2, time_sequence, shape) * x_t
        reverse_variance_t = self._extract(self.reverse_variance, time_sequence, shape)
        reverse_log_variance_clipped_t = self._extract(self.reverse_log_variance_clipped, time_sequence, shape)
        return reverse_mean_t, reverse_variance_t, reverse_log_variance_clipped_t

    # q_sample的逆向过程:即根据公式3从x_t和噪声反推得到x_0
    def predict_start_from_noise(self, x_t, time_sequence, noise):
        shape = x_t.shape
        x_0_pred = self._extract(self.sqrt_recip_overline_alphas, time_sequence, shape) * x_t \
                   - self._extract(self.sqrt_recip_overline_alphas_minus_one, time_sequence, shape) * noise
        return x_0_pred

    # 上面的反例，从x_0和x_t计算噪声
    def predict_noise_from_start(self, x_t, time_sequence, x_0):
        shape = x_t.shape
        x_0_pred = (self._extract(self.sqrt_recip_overline_alphas, time_sequence, shape) * x_t - x_0) / \
                    - self._extract(self.sqrt_recip_overline_alphas_minus_one, time_sequence, shape)
        return x_0_pred

    # 公式14-1和14-3得到q(x_{t-1}|x_t)的噪声均值和方差
    # 根据公式5和预测噪声的模型得到后面用于从x_t预测x_{t-1}即p(x_{t-1}|x_t)的均值和方差
    def p_mean_variance(self, x_t, time_sequence, clip_denoised=True):
        # 根据当前状态x_t和时间戳t预测噪声
        # (其实从总结和反思中可以看见这个噪声其实就是x_0到x_t使用的噪声,也就是q_sample中的noise)
        pred_noise = self.model(x_t, time_sequence)
        # 根据公式4的反推和上面预测得到的噪声从x_t重构x_0
        x_0_reco = self.predict_start_from_noise(x_t, time_sequence, pred_noise)
        if clip_denoised:
            # 注意这里有一个技巧, 因为默认图像是ToTensor归一化到0~1后又(x-0.5)/0.5最终缩放到-1~1,
            # 所以也希望重构的x_0在-1~1的范围,所以这里做了个clip截断
            x_0_reco = torch.clamp(x_0_reco, min=-1.0, max=1.0)
        # 根据公式(14-2)从x_t、重构的x_0和时间戳来计算q(x_{t-1}|x_t)的方差和均值
        # 此外要注意的是这里的均值计算其实用的是公式14-2，而不是公式14-3，所以其还要有一个从x_t和噪声重构x_0的过程
        # TODO 后面可以考虑使用公式(14-3)绕过x_0直接使用使用公式14-3计算均值？
        #  (其实是可以的，但是这里为了做一个x_0重构的clip，所以使用了x_0作为中介得到噪声均值和方差，即走的是公式14-2而非14-3)，
        #  其实就是加了一个x_0自身的约束保证其在原始处理后数据的(-1,1)范围
        model_mean, reverse_variance, reverse_log_variance = \
            self.q_reverse_mean_variance(x_0_reco, x_t, time_sequence)
        return model_mean, reverse_variance, reverse_log_variance

    # DDPM原始单个的去噪生成step，根据公式5从x_t预测x_{t-1},其实就是DDPM的sampling过程
    @torch.no_grad()
    def p_sample(self, x_t, time_sequence, clip_denoised=True):
        model_mean, _, reverse_log_variance = self.p_mean_variance(x_t, time_sequence, clip_denoised)

        noise = torch.randn_like(x_t)
        # 这里要注意当t=0的时候已经就是最原始的图了，这里不使用任何噪声进行去噪
        nonzero_mask = (time_sequence != 0).float().view(-1, *([1] * (len(x_t.shape) - 1)))
        # TODO 这里为什么要乘以0.5没有很理解，其理论依据来源于哪里？
        #  破案了sqrt(var) == exp(0.5*log(var)), 其实就是方差变标准差
        x_tm1 = model_mean + nonzero_mask * (0.5 * reverse_log_variance).exp() * noise
        return x_tm1

    @torch.no_grad()
    # DDIM的核心公式:从x_0和x_t预测x_{t-1}
    def p_sample_ddim(self, x_t, time, time_prev, eta, clip_denoised=True):
        batch_size = x_t.shape[0]
        time_sequence = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
        pred_noise = self.model(x_t, time_sequence)  # 根据x_t和当前时间戳预测用于去噪的噪声
        x_0_reco = self.predict_start_from_noise(x_t, time_sequence, pred_noise)  # 根据公式3反推得到x_0
        if clip_denoised:
            x_0_reco = torch.clamp(x_0_reco, min=-1.0, max=1.0)
        if time_prev < 0:  # 这里相当于去噪过程结束
            return x_0_reco
        overline_alphas_t = self.overline_alphas[time]
        overline_alphas_t_prev = self.overline_alphas[time_prev]
        sigma = eta * torch.sqrt((1 - overline_alphas_t / overline_alphas_t_prev) *
                                 (1 - overline_alphas_t_prev) / (1 - overline_alphas_t))
        pred_noise_coef = torch.sqrt(1 - overline_alphas_t_prev - sigma ** 2)
        noise = torch.randn_like(x_t, device=self.device)
        # DDIM的核心:从x_0和x_t预测x_{t-1}
        x_tm1 = torch.sqrt(overline_alphas_t_prev) * x_0_reco + pred_noise_coef * pred_noise + sigma * noise
        return x_tm1

    @torch.no_grad()
    # sample函数中是从T-->0的采样time下标,并且一开始去噪的慢，后面去噪的快
    # ddim_sample: "DDIM_50_0p0"/"DDPM_500_0p0"
    def sample(self, image_size, batch_size=8, channels=3, clip_denoised=True, sample_info="DDPM_{}_1p0"):
        sample_info = sample_info.format(self.timestep_num)   # 默认使用DDPM采样，采样步数为训练时候的加噪步数
        sample_shape = (batch_size, channels, image_size, image_size)
        sample_mode, iternum, eta = sample_info.split("_")
        sample_stepnum, eta = int(iternum), float(eta.replace("p", "."))
        assert sample_stepnum <= self.timestep_num, "Sampling number must less than timestep_num in training!"

        times = torch.linspace(-1, self.timestep_num-1, steps=sample_stepnum+1)
        times = list(reversed(times.int().tolist()))
        images = []  # 保存逆向过程中的生成的中间图像用于后面可视化
        # 初始化从纯的高斯随机噪声开始进行去噪生成,就相当于x_T,这里原来犯过一个很低级的错误，
        # 就是不小心手误写成了torch.rand的0~1均匀采样，在DDIM且eta=0的时候效果差异非常大以至于无法生成图像，
        # 而在DDPM或者DDIM且eta=1.0的时候基本没有影响
        image = torch.randn(sample_shape, device=self.device)

        for tau in tqdm(range(0, len(times)-1), desc='Sampling loop time step....', total=sample_stepnum):
            # 在迭代的过程中，image就相当于x_t
            time, time_prev = times[tau], times[tau+1]   # t和t_next分别对应于公式中的tau_i和tau_{i-1}表示采样过程中相邻的两个时间戳下标

            if sample_mode.lower() == "ddim":
                # 1.使用ddim的方式进行采样
                image = self.p_sample_ddim(image, time, time_prev, eta, clip_denoised)
            elif sample_mode.lower() == "ddpm":
                # 2.使用DDPM进行采样生成仍然使用time作为下标进行一般的ddpm采样，如果sample_stepnum小于timestep_num，
                # 就默认表示在DDPM下使用间隔指定长度步长进行采样，当sample_stepnum=timestep_num的时候，
                # 就真的变成了原始无加速的DDPM算法，可以在这里减小sample_stepnum看一下间隔步长导致的DDPM降噪效果退化
                time_sequence = torch.full((batch_size,), time, device=self.device, dtype=torch.long)
                image = self.p_sample(image, time_sequence, clip_denoised)
            else:
                raise NotImplementedError("Sample_mode:{} is not Implemented".format(sample_mode))

            images.append(image.cpu())
        return images

    def train_loss(self, x_0, time_sequence):
        # 计算损失的过程非常简单:
        # 1.从x_0根据公式(3)从x_0生成x_t
        noise = torch.randn_like(x_0)
        x_t = self.q_sample(x_0, time_sequence, noise=noise)
        # 2.网络预测噪声(预测的噪声就是用于原本从x_0加噪到x_t的噪声)
        pred_noise = self.model(x_t, time_sequence)
        # 3.预测噪声和真实噪声计算MSE
        if self.loss_type.lower() == "l1":
            loss = F.l1_loss(noise, pred_noise)
        elif self.loss_type.lower() == "l2":
            loss = F.mse_loss(noise, pred_noise)
        else:
            raise NotImplementedError("Loss type of {} is not supportd now~".format(self.loss_type))
        return loss


if __name__ == '__main__':

    diff_model = Diffusion(model=None)
    time_sequence = np.array([6, 2, 3, 5])
    diff_model.q_sample(torch.randn(4, 1, 28, 28), torch.from_numpy(time_sequence))
