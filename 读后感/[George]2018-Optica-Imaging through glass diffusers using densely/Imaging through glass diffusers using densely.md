# 	**Optical configuration.**

- **Experimental arrangement.** 

  <img src="./assets/Imaging%20through%20glass%20diffusers%20using%20densely/Experimental%20arrangement.png" alt="Experimental arrangement" width="800”;" />

- **Detail of the telescopic imaging system.** 

  <img src="./assets/Imaging%20through%20glass%20diffusers%20using%20densely/Detail%20of%20the%20telescopic%20imaging%20system.png" width="800”;" />

  > **Spatial Light Modulator** ***SLM*** **(Holoeye, LC-R 720, Reflective)** 
  > 分辨率*1280×768*。像素大小*20μm*，填充因子*92％*，可见光反射率约*70％*，响应时间小于*3*毫秒。LC-R720可以被用于相位调制，提供约$1\pi$可见光的相移。例如~$1.2\pi$*@532nm*，此处∼17的最大强度调制比， 相位调制深度~$0.6\pi$.

  > **Complementary Metal-Oxide Semiconductor** ***CMOS*** **(Camera Basler, A504k)** 
  > 分辨率*1280×1024*，像素大小*12μm*。

  > **Lens *L1* 和 *L2* 焦距** 
  > *f~1~=250 mm* 和 *f~2~=150 mm*，缩放因子*0.6*

  > **不同目数毛玻璃的** **Bidirectional Scattering Distribution Function**
  > **双向散射分布函数** ***BSDF*** **是一个超集，是[双向反射率分布函数](https://en.wikipedia.org/wiki/Bidirectional_reflectance_distribution_function)** ***BRDF*** **和双向透射率分布函数** ***BTDF* 的泛化** 
  > 所有 ***BxDF*** 函数都可以描述为一个黑盒，输入是任意两个角度，一个用于入射（入射）光线，第二个用于在表面给定点的出射（反射或透射）光线。
  > 黑盒的输出是定义给定角度的入射和出射光能量之间的比率的值。黑匣子的内容可以是的对实际行为进行近似建模的数学公式，或者是基于测量数据从而描述实际行为的算法。

  <img src="./assets/Imaging%20through%20glass%20diffusers%20using%20densely/Bidirectional%20Scattering%20Distribution%20Function.png" alt="Bidirectional Scattering Distribution Function" width="500”;" />

  该函数是 **4(+1) 维的包括 2 个 3D 角度的 4 个值 + 1 个可选的光波长** 
  
  <img src="./assets/Imaging%20through%20glass%20diffusers%20using%20densely/BRDF&BTDF.png" alt="BRDF&BTDF" width="500”;" />
