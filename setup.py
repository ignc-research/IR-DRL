from setuptools import setup, find_namespace_packages
setup(
    name="modular_drl_env",
    version="0.1.0",
    description="A package for the IR-DRL project.",
    packages=find_namespace_packages(where="."),
    package_dir={"": "."},
    package_data={
        "modular_drl_env.assets": ["*","*/*","*/*/*", "*/*/*/*", "*/*/*/*/*", "*/*/*/*/*/*", "*/*/*/*/*/*/*", "*/*/*/*/*/*/*/*", "*/*/*/*/*/*/*/*/*"]
    }
)