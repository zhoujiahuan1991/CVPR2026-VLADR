import numpy as np
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize


def numpy_include():
    try:
        return np.get_include()
    except AttributeError:
        return np.get_numpy_include()


ext_modules = [
    Extension(
        'reid.evaluation.rank_cylib.rank_cy',
        ['reid/evaluation/rank_cylib/rank_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]

__version__ = '1.0.0'

setup(
    name='VLADR',
    version=__version__,
    description='Vision-Language Attribute Disentanglement and Reinforcement for Lifelong Person Re-Identification',
    author='Kunlun Xu, Haotong Cheng, Jiangmeng Li, Xu Zou, Jiahuan Zhou',
    license='MIT',
    packages=find_packages(),
    keywords=['Person Re-Identification', 'Lifelong Learning', 'CLIP', 'Computer Vision'],
    ext_modules=cythonize(ext_modules),
    python_requires='>=3.7',
)
