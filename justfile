# spell-checker:ignore Werror,pydata

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

# Lint Python code using ruff and pyright
lint-py:
    ruff check
    ruff format --check
    pyright

# Generate documentation for the Python bindings using Sphinx (output: `target/python/doc`)
doc-py:
    sphinx-build bindings/python/doc target/python/doc

# Test Python code using pytest and generate a coverage report in `target/python/coverage`
test-py:
    pytest -v --cov=oxidd --cov-report=html:target/python/coverage

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
    pip3 install --upgrade pip pyright ruff sphinx pydata-sphinx-theme pytest-cov
    pip3 install --editable .

# Install all development tools
devtools: devtools-py
    cargo install mdbook
    npm install --prefix=.local/cspell -g cspell
