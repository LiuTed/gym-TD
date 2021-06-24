from setuptools import setup

setup(
    name="gym_TD",
    version="0.3.2",
    url="https://github.com/liuted/gym-TD",
    author="Taide Liu",
    packages=["gym_TD", "gym_TD.envs"],
    install_requires = ["gym", "numpy"],
    dependency_links = ["https://mirrors.ustc.edu.cn/pypi/web/simple"]
)
