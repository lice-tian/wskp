from setuptools import setup
import torch
import os
import glob
from torch.utils.cpp_extension import (CUDAExtension, BuildExtension)


def get_extensions():
    extensions = []
    ext_name = 'wskp_auto_kernel'  # 编译后保存的文件前缀名称及其位置
    os.environ.setdefault('MAX_JOBS', '4')
    # define_macros = []

    if torch.cuda.is_available():
        print(f'Compiling {ext_name} with CUDA')
        # define_macros += [('WITH_CUDA', None)]
        # 宏处理，会在每个.h/.cpp/.cu/.cuh源文件前添加 #define WITH_CUDA！！这个操作很重要
        # 这样在拓展的源文件中就可以通过#ifdef WITH_CUDA来判断是否编译代码
        op_files = glob.glob('./*.cu')
        extension = CUDAExtension  # 如果cuda可用，那么extension类型为CUDAExtension
    else:
        # print(f'Compiling {ext_name} without CUDA')
        raise Exception('无法用CUDA编译')

    include_path = os.path.abspath('./')
    ext_ops = extension(  # 返回setuptools.Extension类
        name=ext_name,
        sources=op_files,
        # language='c',
        # define_macros=define_macros,
        include_dirs=[include_path]
    )
    extensions.append(ext_ops)
    return extensions  # 由setuptools.Extension组成的list


setup(
    name='wskp_auto_kernel',
    ext_modules=get_extensions(),
    cmdclass={'build_ext': BuildExtension},  # BuildExtension代替setuptools.command.build_ext
)
