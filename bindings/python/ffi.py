import shlex
import shutil
import subprocess
import sys
from enum import Enum
from os import environ
from pathlib import Path
from typing import Dict, List, NoReturn

import setuptools.command.build_ext
from cffi import FFI
from setuptools import Distribution, Extension

# spell-checker:ignore cdef,cdefs,liboxidd,PYFFI

repo_dir = Path(__file__).parent.parent.parent
target_dir = repo_dir / "target"
include_dir = target_dir / "include"
oxidd_h = include_dir / "oxidd.h"
profile = "release"


def get_compiler() -> str:
    """Returns an instance of the `build_ext` compiler to do variant detection"""

    build_ext = Distribution().get_command_obj("build_ext")
    assert isinstance(build_ext, setuptools.command.build_ext.build_ext)
    build_ext.finalize_options()
    # register an extension to ensure a compiler is created
    build_ext.extensions = [Extension("ignored", ["ignored.c"])]
    # disable building fake extensions
    build_ext.build_extensions = lambda: None
    # run to populate self.compiler
    build_ext.run()
    return build_ext.compiler.compiler_type


compiler = get_compiler()

liboxidd_ffi_a = (
    target_dir / profile / ("oxidd_ffi.lib" if compiler == "msvc" else "liboxidd_ffi.a")
)


include_dir.mkdir(parents=True, exist_ok=True)


def fatal(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    exit(1)


class BuildMode(Enum):
    """These are the build modes explained in DEVELOPING.md"""

    STATIC = 0
    SHARED_SYSTEM = 1
    SHARED_DEV = 2

    @staticmethod
    def from_env() -> "BuildMode":
        key = "OXIDD_PYFFI_BUILD_MODE"
        s = environ.get(key, None)
        if s is None:
            return BuildMode.SHARED_DEV
        if s == "static":
            return BuildMode.STATIC
        if s == "shared-system":
            return BuildMode.SHARED_SYSTEM
        if s == "shared-dev":
            return BuildMode.SHARED_DEV
        fatal(
            f"Error: unknown build mode '{s}', supported values for {key} are "
            "`static`, `shared-system`, and `shared-dev`.",
        )


build_mode = BuildMode.from_env()


def which(bin: str) -> str:
    res = shutil.which(bin)
    if res is None:
        fatal(f"Error: could not find {bin}")
    return res


cbindgen_bin = which("cbindgen")


def run(*args: str):
    """Runs the script and checks the return code"""
    res = subprocess.run(args, cwd=repo_dir, check=False)
    if res.returncode != 0:
        fatal(f"Error: {shlex.join(args)} failed")


if build_mode != BuildMode.SHARED_SYSTEM:
    cargo_bin = which("cargo")
    print("building crates/oxidd-ffi ...")
    run(cargo_bin, "build", f"--profile={profile}", "--package=oxidd-ffi")

print("running cbindgen ...")
run(
    cbindgen_bin,
    f"--config={repo_dir / 'crates' / 'oxidd-ffi' / 'cbindgen.toml'}",
    f"--output={oxidd_h}",
    str(repo_dir / "crates" / "oxidd-ffi"),
)


def read_cdefs(header: Path) -> str:
    """
    Remove C macros and include directives from the include header since CFFI cannot
    deal with them.
    """

    res = ""
    with header.open("r") as f:
        lines = iter(f)
        while True:
            line = next(lines, None)
            if line is None:
                break
            if not line:
                continue
            if line[0] == "#":
                if line.startswith("#if"):
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

flags: Dict[str, List[str]] = {
    "libraries": []  # for `+=` (MSVC)
}

if build_mode == BuildMode.STATIC:
    flags["include_dirs"] = [str(include_dir)]
    flags["extra_link_args"] = [str(liboxidd_ffi_a)]
elif build_mode == BuildMode.SHARED_SYSTEM:
    flags["libraries"] = ["oxidd"]
elif build_mode == BuildMode.SHARED_DEV:
    flags["include_dirs"] = [str(include_dir)]
    flags["libraries"] = ["oxidd_ffi"]
    flags["runtime_library_dirs"] = flags["library_dirs"] = [str(target_dir / profile)]

if compiler == "msvc":
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
    flags["runtime_library_dirs"] = []

flags["extra_compile_args"] = ["/O2" if compiler == "msvc" else "-O2"]


ffi = FFI()
ffi.set_source("_oxidd", "#include <oxidd.h>\n", ".c", **flags)
ffi.cdef(cdefs)

if __name__ == "__main__":
    ffi.compile(verbose=True)
