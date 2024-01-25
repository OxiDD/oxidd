cmake_minimum_required(VERSION 3.15)

project(OxiDD)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

include(CTest)

# Use FetchContent to acquire Corrosion for the cmake integration of Rust projects.
include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.4.7 # Optionally specify a commit hash, version tag or branch here
)
FetchContent_MakeAvailable(Corrosion)

# This could be derived from the $<CONFIG> if desired.
set(OXIDD_CARGO_PROFILE "release" CACHE STRING "Sets the build profile used for Rust dependencies (`dev`, `release`, or `release-with-debug`)")
set_property(CACHE OXIDD_CARGO_PROFILE PROPERTY STRINGS "dev" "release" "release-with-debug")

if(BUILD_SHARED_LIBS)
    set(crate_type "cdylib")
else()
    set(crate_type "staticlib")
endif()

# Import the oxidd-ffi crate specifically
corrosion_import_crate(
    MANIFEST_PATH Cargo.toml
    PROFILE ${OXIDD_CARGO_PROFILE}
    CRATES oxidd-ffi
    CRATE_TYPES ${crate_type})

# Generate the cbindgen headers automatically after compilation.
find_program(CBINDGEN_EXECUTABLE NAMES cbindgen DOC "Requires cbindgen to generate the OxiDD headers")

add_custom_command(TARGET cargo-build_oxidd-ffi
    POST_BUILD
    COMMAND ${CBINDGEN_EXECUTABLE}
    ARGS --output ${CMAKE_CURRENT_BINARY_DIR}/include/oxidd.h ${CMAKE_CURRENT_SOURCE_DIR}/crates/oxidd-ffi)

target_include_directories(oxidd-ffi INTERFACE
    ${CMAKE_CURRENT_BINARY_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/bindings/cpp/include)

# Install the shared objects generated.
if(BUILD_SHARED_LIBS)
    corrosion_install(TARGETS RUNTIME oxidd-ffi)
endif()

add_library(oxidd ALIAS oxidd-ffi)

add_subdirectory(bindings/cpp/tests EXCLUDE_FROM_ALL)


# Mark some cmake variables as advanced since they don't have to be exposed to the user.
mark_as_advanced(FORCE
    CORROSION_BUILD_TESTS
    CORROSION_DEV_MODE
    CORROSION_NATIVE_TOOLING
    CORROSION_NO_WARN_PARSE_TARGET_TRIPLE_FAILED
    CORROSION_RESPECT_OUTPUT_DIRECTORY
    CORROSION_VERBOSE_OUTPUT
    Rust_RESOLVE_RUSTUP_TOOLCHAINS)
