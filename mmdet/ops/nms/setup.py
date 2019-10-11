import os
import platform
import subprocess
import time
from setuptools import Extension, dist, find_packages, setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

dist.Distribution().fetch_build_eggs(['Cython', 'numpy>=1.11.1'])
import numpy as np  # noqa: E402
from Cython.Build import cythonize  # noqa: E414


nvcc_ARCH  = ['-arch=sm_52']
nvcc_ARCH += ["-gencode=arch=compute_75,code=\"compute_75\""]
nvcc_ARCH += ["-gencode=arch=compute_75,code=\"sm_75\""]
nvcc_ARCH += ["-gencode=arch=compute_70,code=\"sm_70\""]
nvcc_ARCH += ["-gencode=arch=compute_61,code=\"sm_61\""]
nvcc_ARCH += ["-gencode=arch=compute_52,code=\"sm_52\""]
extra_compile_args = {
            'cxx': ['-Wno-unused-function', '-Wno-write-strings'],
            'nvcc': nvcc_ARCH,}

if __name__ == "__main__":
    setup(
        name='nms_cuda',
        ext_modules=[
            CUDAExtension('nms_cuda', [
                'src/nms_cuda.cpp',
                'src/nms_kernel.cu',
            ],
                          extra_compile_args=extra_compile_args,
                          ),
            CUDAExtension('nms_cpu', [
                'src/nms_cpu.cpp',
            ]),
        ],
        cmdclass={'build_ext': BuildExtension})
