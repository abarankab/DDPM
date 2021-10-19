from setuptools import setup

setup(
    name="ddpm",
    py_modules=["ddpm"],
    install_requires=["torch", "torchvision", "einops", "wandb", "joblib"],
)