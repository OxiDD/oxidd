from setuptools import setup

setup(
    setup_requires=["cffi ~= 1.12"],
    cffi_modules=["bindings/python/build/ffi.py:ffi"],
    # CFFI >= 1.12 uses `#define Py_LIMITED_API` on all platforms, so we only
    # need to build one wheel to target multiple CPython versions.
    # https://cffi.readthedocs.io/en/latest/cdef.html#ffibuilder-compile-etc-compiling-out-of-line-modules
    options={"bdist_wheel": {"py_limited_api": "cp39"}},
)
