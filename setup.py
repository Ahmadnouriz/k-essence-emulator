from setuptools import setup, find_packages

setup(
    name="CLODE",
    version="0.2",
    packages=find_packages(),
    author="Ahmadreza NOurizonoz",
    author_email="Ahmadreza.Nourizonoz@unige.ch",
    description="k-essence python library",
    long_description=open('README.md').read(),
    long_description_content_type="text/markdown",
    url="http://github.com/yourusername/my_library",
    classifiers=[
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
    ],
)
