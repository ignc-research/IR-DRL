from setuptools import setup, find_namespace_packages
setup(
    name="modular_drl_env",
    version="0.1.0",
    author="Benno",
    author_email="your.email@example.com",
    description="A package for IRDRL project",
    packages=find_namespace_packages(where="."),
    package_dir={"": "."},
    package_data={
        "modular_drl_env.assets": ["*","*/*","*/*/*", "*/*/*/*", "*/*/*/*/*", "*/*/*/*/*/*", "*/*/*/*/*/*/*", "*/*/*/*/*/*/*/*", "*/*/*/*/*/*/*/*/*"]
    }
)