from setuptools import setup, find_packages

setup(
    name="semantic_colorizer",
    version="0.1.0",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        "torch>=1.12.0",
        "torchvision>=0.13.0",
        "numpy",
        "Pillow",
        "matplotlib",
    ],
    entry_points={
        "console_scripts": [
            "train=src.train:main",
            "evaluate=src.evaluate:main",
            "gui=src.gui:main"
        ]
    }
)
