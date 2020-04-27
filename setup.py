from setuptools import find_packages
from distutils.core import setup

setup(
    name='cgpip',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'numpy',
        'cv2'
    ],
    license='MIT',
)
