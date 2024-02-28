# Print available recipes
help:
    @just --list

# Format code
fmt:
    cargo +nightly fmt

# Lint code
lint:
    cargo +nightly clippy
    cargo +nightly fmt --check
    ./target/cspell/bin/cspell --quiet --unique --gitignore --dot --cache '**'

# Install development tools
install:
    cargo install mdbook x-launcher
    npm install --prefix=target/cspell -g cspell
