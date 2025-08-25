from setuptools import setup, find_packages

# TODO add requirements to the package
setup(
    name = "jfk_taxis",
    version = "0.1",
    packages = find_packages(where = "src"),
    package_dir={"": "src"}
)