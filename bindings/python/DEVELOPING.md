To start developing, create a virtual environment and run:

  pip install --editable .

This will build the `oxidd-ffi` crate in release mode, generate `target/include/oxidd.h` using `cbindgen`, and compile a Python extension module called `_oxidd` via CFFI.

The following environment variables are relevant for release builds:

- `CFLAGS` to set `-O3` for compiling the extension module
- `OXIDD_PYFFI_BUILD_MODE`, can be
  * `static`: statically link `liboxidd_ffi.a` into the extension module
  * `shared-system`: dynamically link the system-installed `liboxidd`
  * `shared-dev`: dynamically link `liboxidd_ffi` from `target/release` and set the `rpath` accordingly
