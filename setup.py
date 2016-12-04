from setuptools import setup
from setuptools import find_packages

setup(name='EasyNN',
      version='0.0.1',
      description='Theano-based Deep Learning',
      author='zpchen',
      author_email='zpchen@iflytek.com',
      url='https://github.com/zhipengChen/EasyNN',
      license='HIT',
      install_requires=['theano'],
      packages=find_packages(),
)
