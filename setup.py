from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='torchcheck',
    version='1.0.5',
    author='fleonce',
    package_data={
        'torchcheck': ['py.typed', 'C/__init__.pyi']
    },
    packages=["torchcheck"],
    ext_modules=[
        CppExtension('torchcheck.C', [
            'torchcheck/csrc/main.cpp',
            'torchcheck/csrc/asserts.cpp',
            'torchcheck/csrc/index.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
