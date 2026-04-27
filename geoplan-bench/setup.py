from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="geoplan-bench",
    version="0.1.0",
    author="Anonymity",
    description="A benchmark for evaluating agent architectures in remote sensing task planning",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="Anonymity",  
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    entry_points={
        "console_scripts": [
            "geoplan-generate=scripts.generate_tasks:main",
            "geoplan-evaluate=scripts.evaluate:main",
            "geoplan-filter=scripts.filter_tasks:main",
        ],
    },
)
