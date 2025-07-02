from setuptools import setup, find_packages

setup(
    name="battery-prediction",
    version="1.0.0",
    description="Battery State of Health and State of Charge Prediction System",
    author="Shishir Kafle",
    author_email="shishirkafle44@gmail.com",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    install_requires=[
        line.strip()
        for line in open("requirements.txt")
        if line.strip() and not line.startswith("#")
    ],
    python_requires=">=3.9",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    entry_points={
        "console_scripts": [
            "battery-predict=battery_predictor:main",
        ],
    },
)
