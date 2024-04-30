# Developing the Python Bindings

To start developing, create a virtual environment (`python -m venv .venv`) and run:

    just devtools-py

This will install a few development tools and run `pip install --editable .`. The latter makes all your edits in Python files available in the virtual environment without the need to run any install commands again. Behind the scenes, the `pip install` command builds the `oxidd-ffi` crate in release mode, generates `target/include/oxidd.h` using `cbindgen`, and compiles a Python extension module called `_oxidd` via CFFI. This also means that if you change the Rust code, you need to run `pip install --editable .` again.


## Common Actions

Most actions can be run using `just` (a tool similar to `make`, available on `crates.io` and via the respective package manager on most systems):

- `just fmt-py`: Run `ruff` as formatter
- `just lint-py`: Run `ruff` as linter and `pyright` as static type checker
- `just doc-py`: Build documentation using Sphinx. The output is placed in `target/python/doc`.
- `just test-py`: Run `pytest` and report test coverage to `target/python/coverage`.


## Build Process Internals

We currently use `setuptools` for the build process. Therefore, the entrypoint is `setup.py` in the project root. `setuptools` is in the progress of migrating away from `setup.py`, so most settings are already taken from `pyproject.toml`. The remaining settings in `setup.py` cannot yet be specified in `pyproject.toml`. `setup.py` mainly registers a CFFI module with `bindings/python/build/ffi.py` as the build script. The latter invokes `cargo`, `cbindgen`, and compilation via CFFI.

There are different ways to link the OxiDD library and the Python extension module together. To control this, there is the `OXIDD_PYFFI_LINK_MODE` environment variable, which can be:
- `static`: Statically link against `liboxidd_ffi.a` or `oxidd_ffi.lib` in the `target/<profile>` directory.

    This is the mode we will be using for shipping via PyPI since we do not need to mess around with the RPath (which does not exist on Windows anyway). It is also the default mode on Windows.
- `shared-system`: Dynamically link against a system-installed `liboxidd.so`, `liboxidd.dylib`, or `oxidd.dll`, respectively.

    This mode is useful for packaging in, e.g., Linux distributions. With this mode, we do not need to ship the main OxiDD library in both packages, `liboxidd` and `python3-oxidd`. We can furthermore decouple updates of the two packages.
- `shared-dev`: Dynamically link against `liboxidd_ffi.so`, `liboxidd_ffi.dylib`, or `oxidd_ffi.dll` in the `target/<profile>` directory.

    This mode is the default for developing on Unix systems. When tuning heuristics for instance, a simple `cargo build --release` suffices, no `pip install --editable .` is required before re-running the Python script. On Windows, setting this mode up requires extra work, since there is no RPath like on Unix systems.

Building Python wheels is possible using [`cibuildwheel`](https://cibuildwheel.pypa.io/en/stable/). To build for Linux, we use a wrapper script (`./build/linux-buildwheel.py`) around `cibuildwheel` to avoid installing a Rust toolchain in the containers. It will both cross-compile the `oxidd-ffi` crate for all the specified target architectures and run `cbindgen` on the host system. Only compiling the Python module and packaging is left to the `cibuildwheel` containers. We pass the environment variable `OXIDD_PYFFI_CONTAINER_BUILD=1` to the build script to indicate that it should run neither `cargo build` nor `cbindgen` but look for `liboxidd_ffi.a` in the respective target subdirectory (e.g., `target/aarch64-unknown-linux-gnu/release`). `OXIDD_PYFFI_CONTAINER_BUILD=1` also implies `OXIDD_PYFFI_LINK_MODE=static`. Note that building for non-native architectures requires [emulation](https://cibuildwheel.pypa.io/en/stable/faq/#emulation).
