from typing import List
from cffi import FFI
from enum import Enum
from os import environ
from pathlib import Path
import shlex
import shutil
import subprocess
import sys

# spell-checker:ignore cdef,cdefs,liboxidd,PYFFI

repo_dir = Path(__file__).parent.parent.parent
target_dir = repo_dir / "target"
include_dir = target_dir / "include"
oxidd_h = include_dir / "oxidd.h"
profile = "release"
liboxidd_ffi_a = target_dir / profile / "liboxidd_ffi.a"
include_dir.mkdir(parents=True, exist_ok=True)


class BuildMode(Enum):
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
        print(
            f"Error: unknown build mode '{s}', supported values for {key} are `static`, `shared-system`, and `shared-dev`.",
            file=sys.stderr,
        )
        exit(1)


build_mode = BuildMode.from_env()


def which(bin: str) -> str:
    res = shutil.which(bin)
    if res is None:
        print(f"Error: could not find {bin}", file=sys.stderr)
        exit(1)
    return res


cbindgen_bin = which("cbindgen")


def run(*args: str):
    res = subprocess.run(args, cwd=repo_dir)
    if res.returncode != 0:
        print(f"Error: {shlex.join(args)} failed", file=sys.stderr)
        exit(1)


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
                            print(
                                f"Error parsing {header}: reached end of file while searching for #endif",
                                file=sys.stderr,
                            )
                            exit(1)
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
