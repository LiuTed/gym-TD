from setuptools import setup

setup(
    name="gym_TD",
    version="0.6.0",
    url="https://github.com/liuted/gym-TD",
    author="Taide Liu",
    packages=["gym_TD", "gym_TD.envs", "gym_TD.utils", "gym_toys", "gym_toys.envs"],
    install_requires = ["gym", "numpy", "scipy"]
)
