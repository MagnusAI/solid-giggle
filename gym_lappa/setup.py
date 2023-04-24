from setuptools import setup

setup(
    name="gym_lappa",
    version="0.0.1",
    install_requires=["gymnasium==0.28.1", "mujoco-py<2.2,>=2.1", "mujoco==2.3.4"],
)