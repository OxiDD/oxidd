<!-- spell-checker:ignore PYFFI -->

To start developing, create a virtual environment and run:

    pip install --editable .

This will build the `oxidd-ffi` crate in release mode, generate `target/include/oxidd.h` using `cbindgen`, and compile a Python extension module called `_oxidd` via CFFI.

There are different ways to link the OxiDD library and the Python extension module together. To control this, there is the `OXIDD_PYFFI_LINK_MODE` environment variable, which can be:
- `static`: Statically link against `liboxidd_ffi.a` or `oxidd_ffi.lib` in the `target/<profile>` directory.
  
    This is the mode we will be using for shipping via PyPI since we do not need to mess around with the RPath (which does not exist on Windows anyway). It is also the default mode on Windows.
- `shared-system`: Dynamically link against a system-installed `liboxidd.so`, `liboxidd.dylib`, or `oxidd.dll`, respectively.

    This mode is useful for packaging in, e.g., Linux distributions. With this mode, we do not need to ship the main OxiDD library in both packages, `liboxidd` and `python3-oxidd`. We can furthermore decouple updates of the two packages.
- `shared-dev`: Dynamically link against `liboxidd_ffi.so`, `liboxidd_ffi.dylib`, or `oxidd_ffi.dll` in the `target/<profile>` directory.

    This mode is the default for developing on Unix systems. When tuning heuristics for instance, a simple `cargo build --release` suffices, no `pip install --editable .` is required before re-running the Python script. On Windows, setting this mode up requires extra work, since there is no RPath like on Unix systems.
