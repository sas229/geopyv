from skbuild import setup

setup(
    name="geopyv",
    version="0.0.1",
    description="A PIV/DIC analysis package for Python.",
    author="Sam Stanier",
    packages=["geopyv"],
    package_dir={"": "src"},
    cmake_install_dir="src/geopyv",
)
