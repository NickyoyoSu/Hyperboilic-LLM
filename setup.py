from setuptools import setup, find_packages

setup(
    name="hypercore",
    version="0.1.0",
    packages=find_packages(),
    description="HyperCore - Hyperbolic Deep Learning Library",
    author="HyperCore Team",
    author_email="example@example.com",
    python_requires=">=3.6",
    install_requires=[
        "torch>=1.8.0",
        "numpy>=1.18.0",
        "scikit-learn>=0.22.0",
    ],
)