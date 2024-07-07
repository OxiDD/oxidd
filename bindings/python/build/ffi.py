"""This script is run form setup.py (in the project root)"""

import os
import platform
import sys
from enum import Enum
from pathlib import Path

# Make relative imports work
sys.path.append(str(Path(__file__).parent.parent))
__package__ = "build"

from cffi import FFI

from .util import fatal, linux_targets, repo_dir, run, which

# spell-checker:ignore AUDITWHEEL,cdef,cdefs,liboxidd


if os.name != "posix" and os.name != "nt":
    fatal("Error: only Unix and Windows systems are currently supported")

target_dir = repo_dir / "target"
include_dir = target_dir / "include"
oxidd_h = include_dir / "oxidd.h"
profile = "release"
target = os.environ.get("CARGO_BUILD_TARGET")
lib_dir = target_dir / target / profile if target else target_dir / profile

include_dir.mkdir(parents=True, exist_ok=True)

container_build = os.environ.get("OXIDD_PYFFI_CONTAINER_BUILD", "") == "1"


class LinkMode(Enum):
    """Modes to build OxiDD, set via the environment variable OXIDD_PYFFI_LINK_MODE"""

    STATIC = 0
    """
    Statically link against `liboxidd_ffi.a` or `oxidd_ffi.lib` in the
    `target/<profile>` directory.

    This is the mode we will be using for shipping via PyPI since we do not need
    to mess around with the RPath (which does not exist on Windows anyway). It
    is also the default mode on Windows.
    """

    SHARED_SYSTEM = 1
    """
    Dynamically link against a system-installed `liboxidd.so`, `liboxidd.dylib`,
    or `oxidd.dll`, respectively.

    This mode is useful for packaging in, e.g., Linux distributions. With this
    mode, we do not need to ship the main OxiDD library in both packages,
    `liboxidd` and `python3-oxidd`. We can furthermore decouple updates of the
    two packages.
    """

    SHARED_DEV = 2
    """
    Dynamically link against `liboxidd_ffi.so`, `liboxidd_ffi.dylib`, or
    `oxidd_ffi.dll` in the `target/<profile>` directory.

    This mode is the default for developing on Unix systems. When tuning
    heuristics for instance, a simple `cargo build --release` suffices, no
    `pip install --editable .` is required before re-running the Python script.
    On Windows, setting this mode up requires extra work, since there is no
    RPath like on Unix systems.
    """

    @staticmethod
    def from_env() -> "LinkMode":
        """Read from the OXIDD_PYFFI_LINK_MODE environment variable"""
        if container_build:
            return LinkMode.STATIC

        key = "OXIDD_PYFFI_LINK_MODE"
        s = os.environ.get(key, None)
        if s is None:
            # Since Windows does not have an RPath, static linking yields the
            # better developer experience.
            return LinkMode.SHARED_DEV if os.name != "nt" else LinkMode.STATIC
        s = s.lower()
        if s == "static":
            return LinkMode.STATIC
        if s == "shared-system":
            return LinkMode.SHARED_SYSTEM
        if s == "shared-dev":
            return LinkMode.SHARED_DEV
        fatal(
            f"Error: unknown build mode '{s}', supported values for {key} are "
            "`static`, `shared-system`, and `shared-dev`.",
        )

    def crate_type(self) -> str:
        """Associated Rust crate type"""
        if self == LinkMode.STATIC:
            return "staticlib"
        return "cdylib"


build_mode = LinkMode.from_env()

if container_build:
    lib_dir = target_dir / linux_targets[os.environ["AUDITWHEEL_PLAT"]] / profile
else:
    cbindgen_bin = which("cbindgen")

    if build_mode != LinkMode.SHARED_SYSTEM:
        cargo_bin = which("cargo")

        # Fix win32 build on win64
        if (
            platform.system() == "Windows"
            and platform.machine() == "AMD64"
            and sys.maxsize <= 0x1_0000_0000
            and not target
        ):
            target = "i686-pc-windows-msvc"
            lib_dir = target_dir / target / profile
            os.environ["CARGO_BUILD_TARGET"] = target

        print(f"building crates/oxidd-ffi ({target if target else 'native'}) ...")
        # Use --crate-type=... to avoid missing linker errors when cross-compiling
        run(
            cargo_bin,
            "rustc",
            f"--profile={profile}",
            "--package=oxidd-ffi",
            f"--crate-type={build_mode.crate_type()}",
        )

    print("running cbindgen ...")
    run(
        cbindgen_bin,
        f"--config={repo_dir / 'crates' / 'oxidd-ffi' / 'cbindgen.toml'}",
        f"--output={oxidd_h}",
        str(repo_dir / "crates" / "oxidd-ffi"),
    )


def read_cdefs(header: Path) -> str:
    """Remove C macros and include directives from the include header since CFFI
    cannot deal with them.
    """

    res = ""
    # encoding="utf-8" seems to be important on Windows
    with header.open("r", encoding="utf-8") as f:
        lines = iter(f)
        while True:
            line = next(lines, None)
            if line is None:
                break
            if all(c == "\n" or c == "\r" for c in line):
                continue
            if line[0] == "#":
                if line.startswith("#if") and not line.startswith(
                    "#ifndef __cplusplus"
                ):
                    while True:
                        line = next(lines, None)
                        if line is None:
                            fatal(
                                f"Error parsing {header}: reached end of file while"
                                "searching for #endif",
                            )
                        if line.startswith("#endif"):
                            break
                continue
            res += line
    return res


cdefs = read_cdefs(oxidd_h)

flags: dict[str, list[str]] = {
    "libraries": []  # for `+=` (MSVC)
}

if build_mode == LinkMode.STATIC:
    flags["include_dirs"] = [str(include_dir)]

    if os.name == "posix":
        flags["extra_link_args"] = [str(lib_dir / "liboxidd_ffi.a")]
    elif os.name == "nt":
        # TODO: This should be derived from
        #       `cargo rustc -q -- --print=native-static-libs`, but without the
        #       `windows.lib` and without duplicates?
        # spell-checker:ignore advapi,ntdll,userenv
        flags["libraries"] += [
            "kernel32",
            "advapi32",
            "bcrypt",
            "ntdll",
            "userenv",
            "ws2_32",
            "msvcrt",
        ]
        flags["extra_link_args"] = [str(lib_dir / "oxidd_ffi.lib")]
elif build_mode == LinkMode.SHARED_SYSTEM:
    if os.name == "posix":
        flags["libraries"] = ["oxidd"]
    elif os.name == "nt":
        flags["libraries"] = ["oxidd.dll"]
elif build_mode == LinkMode.SHARED_DEV:
    flags["include_dirs"] = [str(include_dir)]
    flags["library_dirs"] = [str(lib_dir)]
    if os.name == "posix":
        flags["libraries"] = ["oxidd_ffi"]
        flags["runtime_library_dirs"] = flags["library_dirs"]
    elif os.name == "nt":
        flags["libraries"] = ["oxidd_ffi.dll"]
        # There is no RPath on Windows :(

flags["extra_compile_args"] = ["-O2" if os.name != "nt" else "/O2"]


ffi = FFI()
ffi.set_source("_oxidd", "#include <oxidd.h>\n", ".c", **flags)
ffi.cdef(cdefs)

if __name__ == "__main__":
    ffi.compile(verbose=True)
