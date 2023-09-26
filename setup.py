from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='torchcheck',
    version='1.0.1',
    author='fleonce',
    package_data={
        'torchcheck': ['py.typed', '__init__.pyi']
    },
    packages=["torchcheck"],
    ext_modules=[
        CppExtension('torchcheck', [
            'torchcheck/main.cpp',
            'torchcheck/asserts.cpp',
            'torchcheck/index.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
