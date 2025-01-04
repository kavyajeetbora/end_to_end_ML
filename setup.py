from setuptools import find_packages, setup
from typing import List

HYPHEN_DOT = "-e ."


def get_requirements(file_path: str) -> List[str]:
    """
    This function will return the list of requirements
    """
    requirements = []
    with open(file_path, "r") as file_handler:
        requirements = [i.strip() for i in file_handler.readlines() if i != HYPHEN_DOT]

    return requirements


setup(
    name="END to END Machine Learning Project",
    version="0.0.1",
    author="Kavyajeet Bora",
    author_email="1234@gmail.com",
    packages=find_packages(),
    install_requires=get_requirements("requirements.txt"),
)
