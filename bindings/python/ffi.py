from typing import List, NoReturn
from cffi import FFI
from enum import Enum
from os import environ
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

from setuptools import Distribution, Extension

# spell-checker:ignore cdef,cdefs,liboxidd,PYFFI

repo_dir = Path(__file__).parent.parent.parent
target_dir = repo_dir / "target"
include_dir = target_dir / "include"
oxidd_h = include_dir / "oxidd.h"
profile = "release"

def get_compiler():
    """ Returns an instance of the build_ext compiler to do variant detection """

    build_ext = Distribution().get_command_obj("build_ext")
    build_ext.finalize_options()
    # register an extension to ensure a compiler is created
    build_ext.extensions = [Extension("ignored", ["ignored.c"])]
    # disable building fake extensions
    build_ext.build_extensions = lambda: None
    # run to populate self.compiler
    build_ext.run()
    return build_ext.compiler

if get_compiler().compiler_type == "msvc":
    liboxidd_ffi_a = target_dir / profile / "oxidd_ffi.lib"
else:
    liboxidd_ffi_a = target_dir / profile / "liboxidd_ffi.a"


include_dir.mkdir(parents=True, exist_ok=True)


def fatal(msg: str) -> NoReturn:
    print(msg, file=sys.stderr)
    exit(1)


class BuildMode(Enum):
    """ These are the build modes explained in DEVELOPING.md """
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
            f"Error: unknown build mode '{s}', supported values for {key} are `static`, `shared-system`, and `shared-dev`.",
        )


build_mode = BuildMode.from_env()


def which(bin: str) -> str:
    res = shutil.which(bin)
    if res is None:
        fatal(f"Error: could not find {bin}")
    return res


cbindgen_bin = which("cbindgen")


def run(*args: str):
    """ Runs the script and checks the return code """
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
    """ Removes C macros and include directives from the include header since cffi cannot deal with them. """
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
                                f"Error parsing {header}: reached end of file while searching for #endif",
                            )
                        if line.startswith("#endif"):
                            break
                continue
            res += line
    return res


cdefs = read_cdefs(oxidd_h)

include_dirs: List[str] = []
libraries: List[str] = []
library_dirs: List[str] = []
runtime_library_dirs: List[str] = []
extra_link_args: List[str] = []

if build_mode == BuildMode.STATIC:
    include_dirs = [str(include_dir)]

    extra_link_args = [str(liboxidd_ffi_a)]
    if get_compiler().compiler_type == "msvc":
        # TODO: This should be derived from 'cargo rustc -q -- --print=native-static-libs', but without the windows.lib and without duplicates?
        libraries = ["kernel32", "advapi32", "bcrypt", "ntdll", "userenv", "ws2_32", "msvcrt"]

elif build_mode == BuildMode.SHARED_SYSTEM:
    libraries = ["oxidd"]

elif build_mode == BuildMode.SHARED_DEV:
    include_dirs = [str(include_dir)]
    libraries = ["oxidd_ffi"]
    library_dirs = [str(target_dir / profile)]
    runtime_library_dirs = library_dirs


ffi = FFI()
ffi.set_source(
    "_oxidd",
    "#include <oxidd.h>\n",
    include_dirs=include_dirs,
    libraries=libraries,
    library_dirs=library_dirs,
    runtime_library_dirs=runtime_library_dirs,
    extra_link_args=extra_link_args,
)
ffi.cdef(cdefs)

if __name__ == "__main__":
    ffi.compile(verbose=True)
