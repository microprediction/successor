import pathlib
from setuptools import setup, find_packages

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text()

setup(
    name="successor",
    version="0.1.2",
    description="Predict the next number in a sequence, or the next k",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://github.com/microprediction/successor",
    author="microprediction",
    author_email="peter.cotton@microprediction.com",
    license="MIT",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
    ],
    packages=["successor","successor.skaters","successor.skaters.scalarskaters","successor.extension","successor.interpolation"],
    test_suite='pytest',
    tests_require=['pytest'],
    include_package_data=True,
    install_requires=["wheel","pathlib","getjson","numpy","tensorflow","momentum>=0.2.1"],
    entry_points={
        "console_scripts": [
            "successor=successor.__main__:main",
        ]
    },
)
