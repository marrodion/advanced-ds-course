from pathlib import Path

from setuptools import find_packages, setup

NAME = "fraud"
DESCRIPTION = "Example model for ML pipelines"
URL = "https://github.com/marrodion/ml-eng"
EMAIL = "mar.rodion@gmail.com"
AUTHOR = ""
REQUIRES_PYTHON = ">=3.6"
VERSION = "0.1"
BASE_PATH = Path(__file__).parent


try:
    long_description = (BASE_PATH / "README.md").read_text()
except FileNotFoundError:
    long_description = DESCRIPTION


setup(
    name=NAME,
    version=VERSION,
    description=DESCRIPTION,
    long_description=long_description,
    long_description_content_type="text/markdown",
    author=AUTHOR,
    author_email=EMAIL,
    python_requires=REQUIRES_PYTHON,
    url=URL,
    install_requires=[],
    include_package_data=True,
    package_dir={"": "src"},
    packages=find_packages("src"),
)
