# -*- coding: utf-8 -*-
import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setuptools.setup(
    name='SharjahArabicLibraryCode',
    version='0.0.1',
    author='Faisal Alshargi',
    author_email='alshargi@hotmail.de',
    description='Arabic Library of congress code',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url='https://github.com/alshargi/SharjahArabicLibraryCode',
    project_urls={
        "Bug Tracker": "https://github.com/alshargi/SharjahArabicLibraryCode/issues"
    },
    license='MIT',
    packages=['SharjahArabicLibraryCode'],
    package_data={'SharjahArabicLibraryCode': ['models/*.sav']},
    install_requires=['requests'],
)

