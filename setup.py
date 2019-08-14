from setuptools import find_packages
from setuptools import setup

setup(name='rainbow-iqn-apex',
      install_requires=['atari-py', 'redlock-py',
                        'plotly', 'opencv-python'],
      packages=find_packages())
