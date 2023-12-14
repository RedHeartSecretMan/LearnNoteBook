# 本周任务：GAN&Transformer

## GAN

### **1. *Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network***

#### **本文亮点**

1. **第一个实现单图像 4 倍超分辨的神经网络，提供啦一个高效的框架**
2. **提出感知相似意分数评价网络，远超现有模型，接近真实结果**



#### 单图像 4 倍超分辨的难点

1. **The ill-posed nature of the underdetermined SR problem is particularly pronounced for high upscaling factors, for which texture detail in the reconstructed SR images is typically absent 欠定的SR问题的不适定性质在高尺度因子时尤其明显，即重构的SR图像中通常缺乏纹理细节**
2. **现有损失函数 MSE 与 PSNR 捕捉感知相关的差异（高纹理细节）能力非常有限，例如 MSE 鼓励寻找合理解决方案的像素级的平均，导致结果过于平滑**

<img src="./assets/Photo-Realistic%20Single%20Image%20Super-Resolution%20Using%20a%20Generative%20Adversarial%20Network/image-20220504122607048.png" alt="image-20220504122607048" style="zoom: 67%;" />

3. **SR 问题的相关方法起源于压缩感知（寻找欠定线性系统的稀疏解的技术，在远小于Nyquist采样率的条件下，用随机采样获取信号的离散样本，然后通过非线性重建算法完美的重建信号），单图像的超分辨相比从多幅低分辨率图像恢复高分辨率图像的可用信息更少，需要更为强大的建模能力**



***Batch Normalization***
$$
input = tensor_{(B,C,H,W)}\\
x = tensor_{(B,H,W)}\\
\hat{x}_\text{new} = (1 - \text{momentum}) \times \hat{x} + \text{momentum} \times x_t\\
y = \frac{x - \mathrm{E}[x]}{ \sqrt{\mathrm{Var}[x] + \epsilon}} * \gamma + \beta\\
output = y
$$
