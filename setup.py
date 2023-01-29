from skbuild import setup

setup(
    packages=["geopyv", "geopyv.geometry", "geopyv.gui", "geopyv.gui.selectors"],
    package_dir={"": "src"},
    cmake_install_dir="src/geopyv",
    include_package_data=True,
)
