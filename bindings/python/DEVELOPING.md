# Developing the Python Bindings

To start developing, create a virtual environment (`python -m venv .venv`) and run:

    just devtools-py

This essentially executes `pip3 install --editable '.[dev,docs,test]'`. Here, `--editable` means that if you edit Python files, you do not need to re-install the package. This does not apply to Rust code, however. The current build backend is [maturin](https://www.maturin.rs/), and instead of `pip3 install --editable .`, you can also run `maturin develop` (or `maturin develop --release` for an optimized build). During the development process, this is more handy, since pip captures all the output of the build tool and does not present errors as nicely. Note that your working directory needs to be in the project root (and not `crates/oxidd-ffi-python`), otherwise maturin does not pick up the correct configuration.

To assist users of OxiDD's Python bindings with PEP 484 type hints, there is a build script (in `crates/oxidd-ffi-python/{build.rs,stub_gen}`) that parses the Rust source code and extracts the type hints from the docstrings. To that end, the docstrings need to follow the [Google Python Style Guide](https://google.github.io/styleguide/pyguide.html) (see also https://www.sphinx-doc.org/en/master/usage/extensions/napoleon.html).


## Common Actions

Most actions can be run using `just` (a tool similar to `make`, available on crates.io and via the respective package manager on most systems):

- `just fmt-py`: Run `ruff` as formatter
- `just lint-py`: Run `ruff` as linter and `pyright` as static type checker
- `just doc-py`: Build documentation using Sphinx. The output is placed in `target/python/doc`.
- `just test-py`: Run `pytest`
