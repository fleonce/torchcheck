from setuptools import setup, find_packages

setup(
    name='torchcheck',
    version='1.0.7',
    author='fleonce',
    package_data={
        'torchcheck': ['py.typed']
    },
    packages=find_packages()
)
