from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="vinsgrad",
    version="0.1.0",
    author="Vincent Amato",
    author_email="vincentaamato@gmail.com",
    description="A lightweight deep learning library inspired from Karpathy's micrograd, smolorg's smolgrad, and PyTorch",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/vincentamato/vinsgrad",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.20.0",
    ],
)