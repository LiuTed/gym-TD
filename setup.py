from setuptools import setup

setup(
    name="gym_TD",
    version="0.5.1",
    url="https://github.com/liuted/gym-TD",
    author="Taide Liu",
    packages=["gym_TD", "gym_TD.envs", "gym_TD.utils"],
    install_requires = ["gym", "numpy>=1.9.0"]
)
