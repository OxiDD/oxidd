# spell-checker:ignore autodoc,pydata,stubtest,Werror

# Print available recipes
help:
    @just --list

# Check spelling in almost all files
spellcheck:
    ./.local/cspell/bin/cspell --quiet --unique --gitignore --dot --cache '**'

# Format Rust code
fmt-rust:
    cargo +nightly fmt

# Lint Rust code
lint-rust:
    cargo +nightly clippy --all-targets
    cargo +nightly fmt --check
    cargo doc --no-deps

# Format C++ code using clang-format
fmt-cpp:
    find bindings/cpp -name '*.[ch]pp' -print0 | xargs -0 clang-format -i

# Lint C++ code using clang-format and clang-tidy
lint-cpp:
    @{{ if path_exists("build/compile_commands.json") == "false" { error("Could not find `build/compile_commands.json`. You must run CMake first.") } else { "" } }}
    find bindings/cpp -name '*.[ch]pp' -print0 | xargs -0 clang-format -n -Werror
    find bindings/cpp -name '*.[ch]pp' -print0 | xargs -0 clang-tidy -p build --warnings-as-errors='*'

# Format Python code using ruff
fmt-py:
    ruff format

# Lint Python code using ruff, mypy, and pyright
lint-py:
    ruff check
    ruff format --check
    mypy
    python3 -m mypy.stubtest oxidd._oxidd
    basedpyright

# Generate documentation for the Python bindings using Sphinx (output: `target/python/doc`)
doc-py:
    @# Generate _oxidd.pyi
    cargo check -p oxidd-ffi-python
    @# We want Sphinx autodoc to look at the type annotations of the .pyi file
    @# (we cannot provide the annotations in the compiled module). Hence, we
    @# make a fake oxidd package without the compiled module and `_oxidd.pyi`
    @# as `_oxidd.py`.
    mkdir -p target/python/autodoc/oxidd
    cp bindings/python/oxidd/*.py target/python/autodoc/oxidd
    cp bindings/python/oxidd/_oxidd.pyi target/python/autodoc/oxidd/_oxidd.py
    PYTHONPATH=target/python/autodoc sphinx-build bindings/python/doc target/python/doc

# Test Python code using pytest
test-py:
    pytest --verbose

# `fmt-rust`
fmt: fmt-rust

# `spellcheck` and `lint-rust`
lint: spellcheck lint-rust

# `fmt-rust`, `fmt-cpp`, and `fmt-py`
fmt-all: fmt-rust fmt-cpp fmt-py

# `spellcheck`, `lint-rust`, `lint-cpp`, and `lint-py`
lint-all: spellcheck lint-rust lint-cpp lint-py

# Create a Python virtual environment in `.venv`
devtools-py-venv:
    {{ if path_exists(".venv") == "false" { "python3 -m virtualenv .venv" } else { "" } }}

[private]
devtools-py-venv-warn:
    @{{ if env("VIRTUAL_ENV", "") == "" { "echo; echo 'WARNING: Run `source .venv/bin/activate` to activate the virtual environment'" } else { "" } }}

# Install Python development tools
devtools-py: devtools-py-venv && devtools-py-venv-warn
    #!/bin/sh
    if [ "$VIRTUAL_ENV" = '' ]; then source .venv/bin/activate; fi
    pip3 install --editable '.[dev,docs,test]'

# Install all development tools
devtools: devtools-py
    cargo install mdbook
    npm install --prefix=.local/cspell -g cspell
