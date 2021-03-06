import sys
from skbuild import setup

setup(
    name="geopyv",
    version="0.0.1",
    description="A PIV/DIC analysis package for Python.",
    author="Sam Stanier",
    license="GNU GPLv3",
    packages=["geopyv", "geopyv.image", "geopyv.subset", "geopyv.mesh", "geopyv.particle", "geopyv.sequence", "geopyv.gui", "geopyv.geometry"],
    package_dir={"": "src"},
    cmake_install_dir="src/geopyv",
)
