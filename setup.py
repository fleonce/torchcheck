from setuptools import setup, find_packages

setup(
    name='torchcheck',
    version='1.0.8rc2',
    author='fleonce',
    package_data={
        'torchcheck': ['py.typed']
    },
    packages=find_packages()
)
