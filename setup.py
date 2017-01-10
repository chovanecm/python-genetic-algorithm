from setuptools import setup, find_packages

with open('README.md') as f:
    long_description = ''.join(f.readlines())


def get_requirements():
    with open("requirements.txt") as f:
        return f.readlines()


setup(
    author="Martin Chovanec",
    author_email="chovanecm@gmail.com",
    classifiers=[
        "Development Status :: 3 - Alpha",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.4",
        "Programming Language :: Python :: 3.5",
    ],
    description="Genetic algorithm library",
    long_description=long_description,
    license="MIT License",
    url="https://github.com/chovanecm/python-genetic-algorithm",
    name="mchgenalg",
    keywords="genetic algorithm",
    packages=find_packages(),
#    entry_points={
#        "console_scripts": [
#            "sacredboard = sacredboard.webapp:run"
#        ]
#    },
    install_requires=get_requirements(),
    #    setup_requires=["numpy"],
    tests_require=["pytest"],
    version="0.1.1"
)
