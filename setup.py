import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name="scmappy",
    version="0.2",
    author='Gabriel Torregrosa Cortes',
    author_email="g.torregrosa@outlook.com",
    description="Cell annotation function",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gatocor/scmappy",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={'': 'src'},
    packages=setuptools.find_packages(where="src"),
    python_requires = ">=3.5",
    install_requires = ["numpy>=1.17.5","scanpy>=1.5.0","pandas>=0.25.0","scikit-learn"]
)
