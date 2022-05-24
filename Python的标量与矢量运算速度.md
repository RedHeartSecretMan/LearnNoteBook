# ***Python的标量与矢量运算速度***

[toc]



## **技术背景**

**Python 的官方解释器是用 *C* 实现的 *CPython* 。其他解释器有用 *Java* 实现的 *Jython*、*C#* 实现的 *IronPython* 和 *Python* 自身实现的 *PyPy* 。用 *C* 语言编写的 *CPython* 利于使用 C 语言接口的外部库。一定程度可以解决 *Python* 执行慢的问题，但是用 *C* 编写必要的代码与 *Python* 通信相当麻烦**



### **NumPy**

***NumPy* 是 *Python* 中科学计算的基础包。它是一个 *Python* 库，提供多维数组对象、各种派生对象（例如掩码数组和矩阵）以及用于对数组进行快速操作的各种例程，包括数学、逻辑、形状操作、排序、选择、I/O 、离散傅里叶变换、基本线性代数、基本统计运算、随机模拟等等**



### **Pytorch**

***Pytorch* 是一个基于 *Python* 的数学计算库，是 *Numpy* 的超集，具有强大的 *GPU* 加速的张量计算能力**



### **TensorFlow**

***TensorFlow* 是一个基于数据流编程的符号数学系统，是 *Numpy* 的超集，支持 *GPU* 高性能张量运算**



### **Numba**

***Numba* 是适用于 *Python* 的即时编译器*（JIT）*，最适用于使用 *NumPy* 数组和函数以及循环的代码。使用 *Numba* 最常见的方法是使用装饰器 *`@jit(nopython=True)` `@njit(parallel=True)`* 等，当调用一个 *Numba* 装饰的函数时，它会将被装饰的代码编译为即时的机器代码，执行时比解释速度要快**

- ***Numba* 可以加速循环但是循环状态必须是 *int32 int64 uint64* 等**

- ***Numba* 可以加速 *NumPy function* 加法、相乘和平方等**

  > *Numpy 的归约函数 `sum`, `prod`, `min`, `max`, `argmin` 和 `argmax` 以及数组函数 `mean`、`var` 和 `std`，数组创建函数 `zeros`, `ones`, `arange`, `linspace` 和一些随机函数（rand、randn、ranf、random_sample、sample、random、standard_normal、chisquare、weibull、power、geometric、exponential、poisson、rayleigh、normal、uniform、beta、binomial、f , 伽马, 对数正态, 拉普拉斯, 三角函数）以及矩阵乘法函数 `dot`*

- ***Numba* 可以加速 *NumPy broadcasting***

- ***Numba* 对代码中的变量类型与函数类型都有要求，当无法静态确定函数的返回类型时无法正常工作，目前兼容性一般**

- ***Numba* 性能 *jit(nopython=True)=njit>>jit(nopython=False)* 首次执行后不在需要编译速度更快**

> ```py
> from numba import njit 
> 
> @njit
> def numba_div(a=np.random.rand(10), b=2, i=1e6):
>   	# 返回值 c 与 d 需要定义变量类型，并且最后类型能够被Numba推断
>     c = np.random.rand(10)
>     d = 1.0
>     # 循环状态 i 需要使用限定的类型
>     i = int(i)
>     if mode == "for":
>         for _ in range(m):
>             c = np.divide(a, b)
>             c = np.tanh(c)
>     return c + d
> ```



### **Cython**

***Cython* 是一门编程语言但几乎是 *Python* 的超集，是具有 *C* 数据类型的 *Python*，几乎任何一段 *Python* 代码也是有效的 *Cython* 代码。*Cython* 编译器会将 *Cython* 源代码优化成 *C/C++* 代码并编译为 *Python/C API* 进行等效调用的 *C* 代码，操作 *Python* 值和 *C* 值的代码可以自由混合，使 *Python* 编写 *C* 扩展就像编写 *Python* 本身一样容易**

- **使用 *Cython* 编写可被 *Python* 调用的库**

> ***Cython 基础语法，几乎是 C 和 Python 的融合***
>
> ```python
> """Cython定义静态类型"""
> # 整型
> cdef int i
> cdef long j 
> # 浮点
> cdef float a 
> # 对象
> cdef object ftang(object int):
>   ...
> # 指针
> cdef float *h = &a
> # 定义数组
> cdef double arr[10]
> cdef double arr[5][2]
> # 定义Numpy数组，必然会调用Python解释器，但从十分便捷性
> arr = np.zeros(10, dtype=np.float64)
> # 类型强制转化
> cdef int a = 0
> cdef float b
> b = <float> a
> 
> """Cython定义函数的api"""
> # def函数在Cython中执行的Python函数，Python调用接口
> def function(a, b):
>     return 0
> # cdef函数 - 支持C-only类型的低开销C级函数，只能在cython文件内部调用
> cdef int function(int a, int b):
>     return 0
> # 支持对类扩展，加速类的执行
> cdef class Shrubbery:
>     cdef int width
>     cdef int height
> 
>     def __init__(self, w, h):
>         self.width = w
>         self.height = h
> 
>     def describe(self):
>         print("This shrubbery is", self.width,
>               "by", self.height, "cubits.")
> 
> # 混合cpdef函数 - 具有自动生成的Python兼容性包装器的C级函数
> cpdef double function(double a, double b):
>     return 0.0
> ```
>
> ***在项目文件夹下新建一个 Cython 文件（后缀名为.pyx），在其中编写 Cython 代码***
>
> ```python
> # 例如矩阵乘法 DotCython.pyx
> import numpy as np
> # numpy给cython留了调用的c-level的接口使用cimport导入
> cimport numpy as np  
> cimport cython
> 
> @cython.boundscheck(False)
> @cython.wraparound(False)
> cdef np.ndarray[np.float32_t, ndim=2] _cython_dot(np.ndarray[np.float32_t, ndim=2] a, np.ndarray[np.float32_t, ndim=2] b):
>     cdef np.ndarray[np.float32_t, ndim=2] c
>     cdef int n, p, m
>     cdef np.float32_t s
>     if a.shape[1] != b.shape[0]:
>         raise ValueError('shape not matched')
>     n, p, m = a.shape[0], a.shape[1], b.shape[1]
>     c = np.zeros((n, m), dtype=np.float32)
>     for i in range(n):
>         for j in range(m):
>             s = 0
>             for k in range(p):
>                 s += a[i, k] * b[k, j]
>             c[i, j] = s
>     return c
> 
> def cython_dot(a, b):
>     return _cython_dot(a, b)
> ```
>
> ***在项目文件夹下新建一个名为 setup.py 的文件，类似于 python Makefile 的文件构建 Cython 模块***
>
> ```py
> # 必须部分「接口配置」
> from distutils.core import setup, Extension
> from Cython.Build import cythonize
> # 可选部分「用到的库」
> import numpy as np  
> 
> # 调用numpy所以添加include_dirs参数，没有则可以去掉
> ext_modules = [Extension("DotCython", ["DotCython.pyx"], include_dirs=[np.get_include()]), ]
> setup(ext_modules=cythonize(Extension(
>     'DotCython',
>     sources=['DotCython.pyx'],
>     language='c',
>     include_dirs=[np.get_include()],
>     library_dirs=[],
>     libraries=[],
>     extra_compile_args=[],
>     extra_link_args=[]
> )))
> # 'DotCython' 是我们要生成的动态链接库的名字
> # sources 里面可以包含 .pyx 文件，以及后面如果我们要调用 C/C++ 程序的话，还可以往里面加 .c / .cpp 文件
> # language 其实默认就是 c，如果要用 c++，就改成 c++ 
> # include_dirs 这个就是传给 gcc 的 -I 参数
> # library_dirs 这个就是传给 gcc 的 -L 参数
> # libraries 这个就是传给 gcc 的 -l 参数
> # extra_compile_args 就是传给 gcc 的额外的编译参数，比方说你可以传一个 -std=c++11
> # extra_link_args 就是传给 gcc 的额外的链接参数
> 
> # 也可简单直接定义构建代码
> from setuptools import setup
> from Cython.Build import cythonize
> 
> setup(
>     ext_modules = cythonize("filename.pyx")
> )
> ```
>
> ***在 `shell` 中执行 `pythonX.X setup.py build_ext --inplace` 成功运行后会生成 build文件夹、.c文件、.so文件，其中 .so文件里是 .pyx编译好的可供python调用的文件，调用格式是 `from filename import functionname`，通过执行 `cython <option> filename.pyx` 其中 option 是传给 Cython 编译器的参数，使用 `-a` 参数可以查看代码的优化细节*** 
>
> ```shell
> Options:
>   -V, --version                  Display version number of cython compiler
>   -l, --create-listing           Write error messages to a listing file
>   -I, --include-dir <directory>  Search for include files in named directory
>                                  (multiple include directories are allowed).
>   -o, --output-file <filename>   Specify name of generated C file
>   -t, --timestamps               Only compile newer source files
>   -f, --force                    Compile all source files (overrides implied -t)
>   -v, --verbose                  Be verbose, print file names on multiple compilation
>   -p, --embed-positions          If specified, the positions in Cython files of each
>                                  function definition is embedded in its docstring.
>   --cleanup <level>              Release interned objects on python exit, for memory debugging.
>                                  Level indicates aggressiveness, default 0 releases nothing.
>   -w, --working <directory>      Sets the working directory for Cython (the directory modules
>                                  are searched from)
>   --gdb                          Output debug information for cygdb
>   --gdb-outdir <directory>       Specify gdb debug information output directory. Implies --gdb.
> 
>   -D, --no-docstrings            Strip docstrings from the compiled module.
>   -a, --annotate                 Produce a colorized HTML version of the source.
>   --annotate-coverage <cov.xml>  Annotate and include coverage information from cov.xml.
>   --line-directives              Produce #line directives pointing to the .pyx source
>   --cplus                        Output a C++ rather than C file.
>   --embed[=<method_name>]        Generate a main() function that embeds the Python interpreter.
>   -2                             Compile based on Python-2 syntax and code semantics.
>   -3                             Compile based on Python-3 syntax and code semantics.
>   --3str                         Compile based on Python-3 syntax and code semantics without
>                                  assuming unicode by default for string literals under Python 2.
>   --lenient                      Change some compile time errors to runtime errors to
>                                  improve Python compatibility
>   --capi-reexport-cincludes      Add cincluded headers to any auto-generated header files.
>   --fast-fail                    Abort the compilation on the first error
>   --warning-errors, -Werror      Make all warnings into errors
>   --warning-extra, -Wextra       Enable extra warnings
>   -X, --directive <name>=<value>[,<name=value,...] Overrides a compiler directive
>   -E, --compile-time-env name=value[,<name=value,...] Provides compile time env like DEF would do.
> ```

- ***Cython* 可在 *Python* 中导入 Cython 库使用，在 *Jupyter* 中使用只需通过行魔法（本行有效）声明 *`%load_ext Cython` 与单元魔法（本单元有效）声明 `%%cython <option>` 有序优化执行***

  > ***在 Python 中直接使用 Cython 库直接实现具有 Cython 特性的 Python 代码***
  >
  > ```python
  > """定义变量"""
  > # 导入Cython
  > import cython
  > # 整型
  > a: cython.int = 10
  > # 浮点
  > b: cython.float = 2.5
  > # 数组
  > c: cython.int[4] = [1, 2, 3, 4]
  > # 指针p_type表示某种类型的指针 
  > d: cython.p_float = cython.address(c)
  > e: cython.pointer(cython.int) = cython.address(a)
  > # 字符
  > f: cython.char = 'a'
  > # 类型强制转换
  > p: cython.p_char
  > q: cython.p_float
  > g = cython.cast(cython.p_char, q)
  > ......
  > # 结构体
  > Grail = cython.struct(
  >     age=cython.int,
  >     volume=cython.float)
  > # 公用体
  > Food = cython.union(
  >     spam=cython.p_char,
  >     eggs=cython.p_float)
  > # 类型命名
  > ULong = cython.typedef(cython.ulong)
  > IntPtr = cython.typedef(cython.p_int)
  > # 通过装饰器将Python函数包装成C函数以及包装类
  > @cython.cfunc
  > def eggs(l: cython.ulong, f: cython.float) -> cython.int:
  >     ...
  >     
  > @cython.cclass
  > class Shrubbery:
  >     width: cython.int
  >     height: cython.int
  > 
  >     def __init__(self, w, h):
  >         self.width = w
  >         self.height = h
  > 
  >     def describe(self):
  >         print("This shrubbery is", self.width,
  >               "by", self.height, "cubits.")
  >         
  > # 声明一个名为的参数int，它是一个Python对象。且object用作函数的显式返回类型
  > @cython.cfunc
  > def ftang(int: object) -> object:
  >     ... 
  > 
  > # 声明整数for-loop的整数循环变量为Cython整数类型就能将for-loop优化为Cython        
  > # Python中大量检查会严重影响性能可以使用Cython关闭
  > # cython: initializedcheck=False 内存视图是否初始化
  > # cython: cdivision=True 负索引
  > # cython: boundscheck=False 数组下标越界
  > # cython: wraparound=False 负索引
  > # 例如
  > @cython.cfunc
  > @cython.boundsback(False)
  > n: cython.int = 10
  > i: cython.int = 0
  > for i in range(n):
  >     def function(...):
  >       ...
  > ```

## **标量运算**

```py
import time
import numpy as np
from numba import jit
import torch
import tensorflow as tf


def execute_time(func):
    def func_new(*args, **kwargs):
        time_start = time.time_ns()
        func(*args, **kwargs)
        time_end = time.time_ns()
        sum_time = (time_end - time_start) / 1e9
        print(f"运行总时间{sum_time}秒")
    return func_new


# 加法
@execute_time
def python_add(a=1, b=2, m=1e6, mode="for"):
    m = int(m)
    c = 0
    i = 0
    if mode == "for":
        for _ in range(m):
            c = a + b
    else:
        while i < m:
            c = a + b
            i += 1
    return c


@execute_time
def numpy_add(a=np.array(1), b=np.array(2), m=int(1e6), mode="for"):
    m = np.array(m, dtype=np.int64)
    c = np.array(0)
    i = np.array(0)
    if mode == "for":
        for _ in range(m):
            c = np.add(a, b)
    else:
        while i < m:
            c = np.add(a, b)
            i = np.add(i, np.array(1))
    return c


# 减法
@execute_time
def python_sub(a=1, b=2, m=1e6, mode="for"):
    m = int(m)
    c = 0
    i = 0
    if mode == "for":
        for _ in range(m):
            c = a - b
    else:
        while i < m:
            c = a - b
            i += 1
    return c


@execute_time
def numpy_sub(a=np.array(1), b=np.array(2), m=int(1e6), mode="for"):
    m = np.array(m, dtype=np.int64)
    c = np.array(0)
    i = np.array(0)
    if mode == "for":
        for _ in range(m):
            c = np.subtract(a, b)
    else:
        while i < m:
            c = np.subtract(a, b)
            i = np.add(i, np.array(1))
    return c


# 乘法
@execute_time
def python_mul(a=1, b=2, m=1e6, mode="for"):
    m = int(m)
    c = 0
    i = 0
    if mode == "for":
        for _ in range(m):
            c = a * b
    else:
        while i < m:
            c = a * b
            i += 1
    return c


@execute_time
def numpy_mul(a=np.array(1), b=np.array(2), m=int(1e6), mode="for"):
    m = np.array(m, dtype=np.int64)
    c = np.array(0)
    i = np.array(0)
    if mode == "for":
        for _ in range(m):
            c = np.multiply(a, b)
    else:
        while i < m:
            c = np.multiply(a, b)
            i = np.add(i, np.array(1))
    return c


# 除法
@execute_time
def python_div(a=1, b=2, m=1e6, mode="for"):
    m = int(m)
    c = 0
    i = 0
    if mode == "for":
        for _ in range(m):
            c = a / b
    else:
        while i < m:
            c = a / b
            i += 1
    return c


@execute_time
def numpy_div(a=np.array(1), b=np.array(2), m=int(1e6), mode="for"):
    m = np.array(m, dtype=np.int64)
    c = np.array(0)
    i = np.array(0)
    if mode == "for":
        for _ in range(m):
            c = np.divide(a, b)
    else:
        while i < m:
            c = np.divide(a, b)
            i = np.add(i, np.array(1))
    return c


@execute_time
def torch_div(a=torch.tensor(1), b=torch.tensor(2), m=int(1e6), mode="for"):
    m = torch.tensor(m, dtype=torch.int64)
    c = torch.tensor(0)
    i = torch.tensor(0)
    if mode == "for":
        for _ in range(m):
            c = torch.divide(a, b)
    else:
        while i < m:
            c = torch.divide(a, b)
            i = torch.add(i, torch.tensor(1))
    return c


@execute_time
def tensorflow_div(a=tf.constant(1), b=tf.constant(2), m=int(1e6), mode="for"):
    m = tf.constant(m, dtype=tf.int64)
    c = tf.constant(0)
    i = tf.constant(0)
    if mode == "for":
        for _ in range(m):
            c = tf.divide(a, b)
    else:
        while i < m:
            c = tf.divide(a, b)
            i = tf.add(i, tf.constant(1))
    return c


@execute_time
@jit(nopython=True)
def numba_div(a=1, b=2, m=int(1e6), mode="for"):
    i = 0
    c = 0
    if mode == "for":
        for _ in range(m):
            c = np.divide(a, b)
    else:
        while i < m:
            c = np.divide(a, b)
            i = np.add(i, np.array(1))
    return c


if __name__ == "__main__":
    # 标量计算
    python_add()  # 运行总时间0.020576秒
    numpy_add()  # 运行总时间0.273121秒
    python_sub()  # 运行总时间0.02032秒
    numpy_sub()  # 运行总时间0.278431秒
    python_mul()  # 运行总时间0.021092秒
    numpy_mul()  # 运行总时间0.272244秒
    python_div()  # 运行总时间0.023047秒
    numpy_div()  # 运行总时间0.452664秒
    torch_div()  # 运行总时间1.878211秒
    tensorflow_div()  # 运行总时间29.529989秒
    numba_div()  # 首次运行总时间0.169949秒，之后运行总时间5.1e-05秒（可能是从缓存直接拿结果而并未计算）
```

```python
"""Jupyter"""
%load_ext Cython
%%cython -a
import time
import cython
time_start = time.time_ns()

def cython_div(a=1, b=2, m=int(1e6), mode="for"):
    a: cython.int = a
    b: cython.int = b
    m: cython.int = m
    i: cython.int = 0
    c: cython.int = 0
    if mode == "for":
        for i in range(m):
            c = a / b
    else:
        while i < m:
            c = a / b
            i += 1
    return c

cython_div()
time_end = time.time_ns()
sum_time = (time_end - time_start) / 1e9
print(f"运行总时间{sum_time}秒")
# 运行总时间0.016209秒
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

float div(int a, int b, int m, char *mode);

int main()
{   
    clock_t start, finish;
    double  duration; 
    int a = 2;
    int b = 2;
    int m = 1e6;
    char *mode = "for";
    float c;
    start = clock(); 
    c = loopdiv(a, b, m, mode);
    finish = clock();  
    duration = (double)(finish - start) / CLOCKS_PER_SEC;
    printf("运行总时间%f秒\n", duration); 
    return 0;
}

float div(int a, int b, int m, char *mode)
{
    float c;
    int i = 0;
    if(strcmp(mode, "for")==0)
    {
        for(i=0; i < m; i++)
        {
            c = a / b;
        }
    }
    else
    {
        while(i<m)
        {
            c = a / b;
            i++;
        }
    }
    return c;
}
// 运行总时间0.001738秒
```

**标量运算的取决于机器码的翻译效率、执行 *for-loop* 的遍历效率和四则运算的计算效率，对于纯暴力的模式最底层的 *C* 性能最好，速度最快，*Pytorch* 与 *TensorFlow* 封装太高，需要初始化再调用底层实现的 *C/C++* 代码，整个过程代价太高，速度最慢**

- **第一梯队是 *C* 处于 $10^{-3}$ 量级**

- **第二梯队是 *Cython* 和 *Python* 内建方法处于 $10^{-2}$ 量级，*Cython* 更快**

  > *标量四则运算 Python 的优化很好* 

- **第三梯队是 *Numba* 和 *Numpy* 处于 $10^{-1}$ 量级，*Numba* 更快**

- **第四梯队是 *PyTorch* 处于 $10^{0}$ 量级**

  >  *PyTorch 暂无法调用 M1 的 GPU 而使用 CPU*

- **第五梯队是 *TensorFlow* 处于 $10^{1}$ 量级**

  > *TensorFlow 可能启动 M1 的 GPU 失败仍然使用 CPU*



## **矢量运算**

```python
import time
import numpy as np
import torch
import tensorflow as tf
from numba import jit, njit
import 随想.DotCython.DotCython as DotCython


def execute_time(func):
    def func_new(*args, **kwargs):
        time_start = time.time_ns()
        func(*args, **kwargs)
        time_end = time.time_ns()
        sum_time = (time_end - time_start) / 1e9
        print(f"运行总时间{sum_time}秒")
    return func_new


# 矩阵乘法
@execute_time
def python_dot(a, b):
    if len(a) != len(b[0]):
        raise ValueError('shape not matched')
    n, p, m = len(a), len(a[0]), len(b[0])
    c = [[0 for i in range(n)] for j in range(m)]
    for i in range(m):
        for j in range(n):
            s = 0
            for k in range(p):
                s += a[i][k] * b[k][j]
            c[i][j] = s
    return c


@execute_time
def numpy_dot(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    c = np.matmul(a, b)
    return c


@execute_time
def torch_dot(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    c = torch.matmul(a, b)
    return c


@execute_time
def tensorflow_dot(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    c = tf.matmul(a, b)
    return c


@execute_time
@jit(nopython=True)
def numba_dot_1(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    n, p, m = a.shape[0], a.shape[1], b.shape[1]
    c = np.zeros((n, m), dtype=np.float32)
    for i in range(n):
        for j in range(m):
            s = 0
            for k in range(p):
                s += a[i, k] * b[k, j]
            c[i, j] = s
    return c


@execute_time
@njit
def numba_dot_2(a, b):
    if a.shape[1] != b.shape[0]:
        raise ValueError('shape not matched')
    c = np.dot(a, b)
    return c

dc.cython_dot(a, b)

if __name__ == "__main__":
    a = [[0 for i in range(1000)] for j in range(500)]
    b = [[0 for i in range(500)] for j in range(1000)]
    # 通过暴力循环方法实现
    python_dot(a, b)  # 运行总时间17.02168秒
		
    a = np.random.rand(1000, 500)
    b = np.random.rand(500, 1000)
    # 通过Numpy矢量化方法实现
    numpy_dot(a, b)  # 运行总时间0.032583秒

    a = torch.rand(1000, 500)
    b = torch.rand(500, 1000)
    # 通过PyTorch矢量化方法实现
    torch_dot(a, b)  # 运行总时间0.002913秒

    a = tf.random.normal((1000, 500))
    b = tf.random.normal((500, 1000))
    # 通过TensorFlow矢量化方法实现
    tensorflow_dot(a, b)  # 运行总时间0.007627秒

    a = np.ones((1000, 500))  
    b = np.ones((500, 1000))
    # 通过Numpy矢量化方法实现
    numba_dot_1(a, b)  # 首次运行总时间0.845412秒，之后运行总时间0.458038秒
    # 通过Numpy矢量化方法实现
    numba_dot_2(a, b)  # 首次运行总时间0.202812秒，之后运行总时间0.012645秒  
    # 使用事先编译的Cython函数DotCython通过暴力循环实现
    @execute_time
		DotCython.cython_dot(a, b)
		# 运行总时间0.437443秒
```

```py
"""Jupyter"""
%load_ext Cython
%%cython -a
import time
import numpy as np
import cython
time_start = time.time_ns()

def cython_mul(a, b):
    a: cython.double = a
    b: cython.double = b

    c = np.matmul(a, b)

    return c


a = np.random.rand(1000, 500)
b = np.random.rand(500, 1000)
# 通过Numpy矢量化方法实现
cython_mul(a, b)
time_end = time.time_ns()
sum_time = (time_end - time_start) / 1e9
print(f"运行总时间{sum_time}秒")
# 运行总时间0.040552秒
```

```cpp
#include <ctime>
#include <cstdlib>
#include <chrono>
#include <iostream>

class Matrix
{
    float *data;

public:
    size_t n, m;
    Matrix(size_t r, size_t c) : data(new float[r * c]), n(r), m(c) {}
    ~Matrix() { delete[] data; }
    float &operator()(size_t x, size_t y) { return data[x * m + y]; }
    float operator()(size_t x, size_t y) const { return data[x * m + y]; }
};

float dot(const Matrix &a, const Matrix &b)
{
    Matrix c(a.n, b.m);
    for (size_t i = 0; i < a.n; ++i)
        for (size_t j = 0; j < b.m; ++j)
        {
            float s = 0;
            for (size_t k = 0; k < a.m; ++k)
                s += a(i, k) * b(k, j);
            c(i, j) = s;
        }
    return c(0, 0);
}

void fill_rand(Matrix &a)
{
    for (size_t i = 0; i < a.n; ++i)
        for (size_t j = 0; j < a.m; ++j)
            a(i, j) = rand() / static_cast<float>(RAND_MAX) * 2 - 1;
}

int main()
{
    srand((unsigned)time(NULL));
    const int n = 1000, p = 500, m = 1000, T = 3;
    Matrix a(n, p), b(p, m);
    fill_rand(a);
    fill_rand(b);
    auto st = std::chrono::system_clock::now();
    float s = 0;
  	// 通过暴力循环方法实现
    for (int i = 0; i < T; ++i)
    {
        s += dot(a, b);
    }
    auto ed = std::chrono::system_clock::now();
    std::chrono::duration<double> diff = ed - st;
    std::cout << "运行总时间" << diff.count() / T << "秒" << std::endl;
  }
  // 运行总时间2.02449秒
```

**矢量运算其最底层的逻辑仍是 *for-loop* 和四则运算，只是庞杂繁复很多，*Numpy*、*PyTorch* 和 *TensorFlow* 等高级封装库的矢量化算法十分强大，性价比就很高啦，是进行矢量运算最优选择，尤其是可以充分使用 *GPU* 下 *PyTorch* 和 *TensorFlow* 的可并行计算速度会更快，相比之下通过 *Numba、Cython* 提升解释器性能性价比不高**

- **第一梯队是基于自身矢量化函数的 *PyTorch* 和 *TensorFlow* 处于 $10^{-3}$ 量级，*PyTorch* 更快**

- **第二梯队是基于 *Numpy* 矢量化函数的 *Numpy*、*Numba* 和 *Cython* 处于 $10^{-2}$ 量级**

- **第三梯队是暴力循环的 *Numba* 和 *Cython* 处于 $10^{-1}$ 量级**

- **第四梯队是暴力循环的 *C++* 处于 $10^{0}$ 量级**

  > *可能写的代码逻辑不行拖后腿了*

- **第五梯队是暴力循环的 *Python* 内建方法**



## **经验总结**

**标量运算使用 *Python*、*Cython* 和 *Jupyter* 的组合最佳（*Numba* 学习成本高，使用限制多，我实验中经常编译错误，没有预想方便好用），矢量运算使用 *Pytorch* 即可（语法和 *Numpy* 相近）**

