from setuptools import setup

setup(
    name="gym_TD",
    version="0.1",
    url="https://github.com/liuted/gym-TD",
    author="Taide Liu",
    packages=["gym_TD", "gym_TD.envs"],
    install_requires = ["gym", "numpy"]
)
