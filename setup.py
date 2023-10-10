from setuptools import find_packages, setup

setup(
    name="peoplenet_torch",
    version="0.0.6",
    description="A Python package that allows you to run NVIDIA's PeopleNet using PyTorch",
    author="Kanat Alimanov",
    author_email="7l21@pm.me",
    url="https://github.com/paindefender/peoplenet_torch",
    packages=["peoplenet_torch"],
    install_requires=[
        "onnx",
        "onnx2torch",
        "torch",
        "torchvision",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "Topic :: Software Development :: Libraries :: Python Modules",
        "Topic :: Scientific/Engineering :: Image Processing",
        "License :: OSI Approved :: GNU Affero General Public License v3 or later (AGPLv3+)",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
    ],
    python_requires=">=3.8",
)
