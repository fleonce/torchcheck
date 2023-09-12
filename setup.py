from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CppExtension

setup(
    name='torch_check',
    package_data={
        'torch_check': ['py.typed', '__init__.pyi']
    },
    packages=["torch_check"],
    ext_modules=[
        CppExtension('torch_check', [
            'torch_check/shape.cpp',
        ]),
    ],
    cmdclass={
        'build_ext': BuildExtension
    })
