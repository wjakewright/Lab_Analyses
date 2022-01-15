from setuptools import find_packages, setup


def readme():
    with open("README.md") as f:
        return f.read()


setup(
    name="Lab_Analyses",
    version="0.0.1",
    description="Code used for behavioral and imaging data",
    long_description=readme(),
    long_description_content_type="text/markdown",
    url="https://github.com/wjakewright/Lab_Analyses",
    author="William (Jake) Wright",
    license="",
    packages=find_packages(),
)
