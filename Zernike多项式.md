# ***Zernike多项式***

[toc]



## **表达式**

$$
\varphi^k(x,y)=Z_n^m(\rho,\theta)=N_n^mR_n^m(\rho)\Theta_m(\theta)\\
\\
\pmb{其中}
\begin{cases}
\rho=\sqrt{x^2+y^2} \;\in (0,\quad 1)\\
\theta=arctan(\frac{y}{x}) \in (0,\quad 2\pi)\\
|Z^{m}_n(\rho,\theta)| \le 1
\end{cases}
$$

- **标准化系数**

$$
N_n^m=
\begin{cases}\sqrt{\frac{2(1+n)}{1+\sigma}}\Rightarrow N_n^m(\pi)\\
\sqrt{\frac{2(1+n)}{\pi(1+\sigma)}}\Rightarrow N_n^m(1)
\end{cases}
\quad \pmb{and}\quad 
\sigma=
\begin{cases}1,&m = 0\\
0,&m\not=0 
\end{cases}
$$

- **径向表达式** 

$$
R_n^m(\rho)=\sum_{k=0}^{\tfrac{n-m}{2}}{\frac{(-1)^k(n-k)!}{k!(\frac{n+m}{2}-k)!(\frac{n-m}{2}-k)!}}\rho^{n-2k}\\
\pmb{or}\\
R_n^m(\rho)=\sum_{k=0}^{\tfrac{n-m}{2}}(-1)^k \binom{n-k}{k} \binom{n-2k}{\tfrac{n-m}{2}-k} \rho^{n-2k}\\
\pmb{其中}\\
k={\frac {n(n+1)}{2}}+|m|+\left\{{\begin{array}{ll}0,&m>0\land n\equiv \{0,1\}{\pmod {4}}\\0,&m<0\land n\equiv \{2,3\}{\pmod {4}}\\1,&m\geq 0\land n\equiv \{2,3\}{\pmod {4}}\\1,&m\leq 0\land n\equiv \{0,1\}{\pmod {4}}\end{array}}\right.\\\\
$$

$$
\int_{0}^{2\pi }\int_{0}^{1}Z_n^m(\rho,\theta)^{2}\cdot \rho \,d\rho \,d\theta =\pi\\
$$



- **弧度表达式**

$$
\Theta_m(\theta)=
\begin{cases}
\cos(|m|\theta) &m\geq 0\\
\sin(|m|\theta) &m<0
\end{cases}
$$



- **多项式积分与方差**

$$
N_n^m(\pi)\Rightarrow
\begin{cases}
\int_{0}^{2\pi }\int_{0}^{1}Z_n^m(\rho,\theta)^{2}\cdot \rho \,d\rho \,d\theta =\pi\\
{Var} (Z_n^m(\rho,\theta))_{\text{unit circle}}=1
\end{cases}
$$



## **应用**

### **光学像差**

**带像差的点扩散函数**
$$
h(u,v;\xi,\eta)=\frac{1}{\lambda^2z^2_i}\cdot\iint^{+\infty}_{-\infty}P(x,y)\cdot e^{\frac{j2\pi\varphi^k(x,y)}{\lambda}}\cdot e^{-j\frac{2\pi}{\lambda z_i}[(u-M\xi)x+(v-M\eta)y]}\,dxdy
$$
- 响应 $h$ 是从光瞳汇聚到理想像点 $(u=M\xi,v=M\eta)$ 球面波产生，$z_i$ 是出瞳到像面的距离，$\lambda$ 是光的波长，$P$ 是光瞳函数是对于出瞳坐标系 $(x,y)$ 孔径内为 $1$ 外为 $0$



### **图像特征**

**一般将图像的几何矩作为特征，将 $2、3$ 阶的几何矩归一化，并施加旋转不变性的要求，得到 *Hu* 矩**  
- ***Hu*** **矩**是局部拟合，容易受到噪声影响 

**按泰勒多项式正交化为勒让德多项式的思路，将几何矩正交化为 *Zernike* 多项式，即 *Zernike* 矩**
- ***Zernike* 多项式**是正交完备基，其主要分量集中在前几项，可以用于当作特征，任何单位圆内的图片可以靠无穷多的 ***Zernike* 多项式**表示  
- ***Zernike* 多项式**前几项集中了图像的一些特征信息，类似傅里叶变换的频谱承载频率信息，工程上通常取前几项就描述特征

**提取图像的 *Zernike* 特征，可以实现目标识别、重建等任务**

