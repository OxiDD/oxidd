# spell-checker:ignore Werror

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

# `fmt-rust`
fmt: fmt-rust

# `spellcheck` and `lint-rust`
lint: spellcheck lint-rust

# Install development tools
devtools:
    cargo install mdbook
    npm install --prefix=.local/cspell -g cspell
