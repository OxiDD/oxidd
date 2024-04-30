#!/usr/bin/env python3

import argparse
import os
import sys
from pathlib import Path
from typing import Optional

# Make relative imports work
sys.path.append(str(Path(__file__).parent.parent))
__package__ = "build"

import cibuildwheel.__main__
import cibuildwheel.options

from .util import linux_targets, repo_dir, run, which

target_dir = repo_dir / "target"
include_dir = target_dir / "include"
oxidd_h = include_dir / "oxidd.h"
profile = "release"
lib_dir = target_dir / profile
wheelhouse_dir = repo_dir / "wheelhouse"

include_dir.mkdir(parents=True, exist_ok=True)


def install_targets(triples: list[str], toolchain: Optional[str]) -> None:
    """Install Rust targets using `rustup target add`"""
    rustup = which("rustup")
    toolchain_args = []
    if toolchain:
        toolchain_args = [f"--toolchain={toolchain}"]
    run(rustup, "target", "add", *toolchain_args, *triples)


def build(archs: str, triples: list[str], toolchain: Optional[str]) -> None:
    """Build wheels

    - `archs` is passed to cibuildwheel
    - `triples` are the Rust target triples to build
    - `toolchain` is the Rust toolchain to use

    Must be run from the repository root directory.
    """
    cargo = which("cargo")
    cbindgen = which("cbindgen")

    toolchain_args = []
    if toolchain:
        toolchain_args = [f"+{toolchain}"]

    for target in triples:
        print(f"building crates/oxidd-ffi for {target} ...")
        run(
            cargo,
            *toolchain_args,
            "rustc",
            f"--profile={profile}",
            f"--target={target}",
            "--package=oxidd-ffi",
            "--crate-type=staticlib",
        )

    print("running cbindgen ...")
    run(
        cbindgen,
        f"--config={repo_dir / 'crates' / 'oxidd-ffi' / 'cbindgen.toml'}",
        f"--output={oxidd_h}",
        str(repo_dir / "crates" / "oxidd-ffi"),
    )

    args = cibuildwheel.options.CommandLineArguments.defaults()
    args.platform = "linux"
    args.archs = archs
    args.output_dir = wheelhouse_dir

    cibuildwheel.__main__.build_in_directory(args)


def get_triples(*archs: str) -> list[str]:
    """Get the list of Rust triples"""
    arch_set = set(archs)
    # We don't handle auto* and native specifically but over-approximate
    all_triples = ["auto", "auto64", "auto32", "native", "all"]
    for k in all_triples:
        if k in arch_set:
            triples = list(linux_targets.values())
            break
    else:
        triples = []
        for triple in linux_targets.values():
            arch = triple.split("-")[0]
            if arch == "ppc64le":
                arch = "powerpc64le"
            if arch in arch_set:
                triples.append(triple)
    return triples


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build Python wheels for Linux using Rust cross compilation on the "
            "host and cibuildwheel with OCI containers afterwards"
        ),
    )
    parser.add_argument(
        "--install-targets",
        action="store_true",
        help="Run `rustup target add` for the respective targets",
    )
    parser.add_argument(
        "--archs",
        default="auto",
        help=(
            "Comma-separated list of CPU architectures to build for. When set "
            "to 'auto', builds the architectures natively supported on this "
            "machine. Set this option to build an architecture via emulation, "
            "for example, using binfmt_misc and QEMU. Default: auto. Choices: "
            "auto, auto64, auto32, native, all, x86_64, i686, aarch64, "
            "ppc64le, s390x"
        ),
    )
    parser.add_argument(
        "--toolchain",
        help=(
            "Which Rust toolchain to use (stable, nightly, 1.77.2, ...). If "
            "omitted, the default Rust toolchain will be picked."
        ),
    )

    args = parser.parse_args()
    archs = str(args.archs)
    toolchain = None
    if args.toolchain:
        toolchain = str(args.toolchain)

    os.chdir(repo_dir)

    triples = get_triples(*(a.strip() for a in archs.split(",")))
    if args.install_targets:
        print("installing targets ...")
        install_targets(triples, toolchain)
    build(archs, triples, toolchain)


if __name__ == "__main__":
    main()
