from setuptools import setup, find_packages, find_namespace_packages
import os
import pathlib

def package_files(directory):
    directory = str(pathlib.Path(__file__).parent.absolute()) + str(pathlib.Path(directory))
    paths = []
    for (path, directories, filenames) in os.walk(directory):
        for filename in filenames:
            paths.append(os.path.join('..', path, filename))
    return paths

extra_files = package_files('./modular_drl_env/assets')
print(extra_files)

"""
setup(
    name="modular_drl_env",
    version="0.1.0",
    author="Benno",
    author_email="your.email@example.com",
    description="A package for IRDRL project",
    packages=["modular_drl_env." + i for i in find_packages("modular_drl_env")],
    package_data={
        "modular_drl_env" : ["assets/*"],
    },
    include_package_data=True,
    #packages=["modular_drl_env.robot", "modular_drl_env.goal","modular_drl_env.gym_env","modular_drl_env.sensor", "modular_drl_env.util", "modular_drl_env.world"],
    package_dir={'modular_drl_env': 'modular_drl_env'},
    install_requires=[
        "numpy>=1.18.0",
        "tensorflow>=2.4.0",
        "gym>=0.18.3",
        "stable-baselines3>=1.1",
        "matplotlib>=3.2.2",
        "scipy>=1.4.1"
    ],

    
)
"""
setup(
    name="modular_drl_env",
    version="0.1.0",
    author="Benno",
    author_email="your.email@example.com",
    description="A package for IRDRL project",
    packages=find_namespace_packages(where="."),
    package_dir={"": "."},
    package_data={
        "modular_drl_env.assets": ["*","*/*","*/*/*", "*/*/*/*", "*/*/*/*/*", "*/*/*/*/*/*", "*/*/*/*/*/*/*", "*/*/*/*/*/*/*/*/*"]
    }
)