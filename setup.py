from setuptools import setup

setup(
    setup_requires=["cffi ~= 1.12"],
    cffi_modules=["bindings/python/ffi.py:ffi"],
    install_requires=["cffi ~= 1.12"],
)
