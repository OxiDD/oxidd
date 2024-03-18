cmake_minimum_required(VERSION 3.15)

project(OxiDD
    VERSION 0.1.0
    DESCRIPTION "Concurrent Decision Diagram Library"
    HOMEPAGE_URL "https://oxidd.net"
    LANGUAGES CXX)


# --- Settings -----------------------------------------------------------------

# This could be derived from the $<CONFIG> if desired.
set(OXIDD_CARGO_PROFILE "release" CACHE STRING "Sets the build profile used for Rust dependencies (`dev`, `release`, or `release-with-debug`)")
set_property(CACHE OXIDD_CARGO_PROFILE PROPERTY STRINGS "dev" "release" "release-with-debug")

option(OXIDD_CXX_DOCS "Build C/C++ API documentation for OxiDD" ${PROJECT_IS_TOP_LEVEL})

# Mark some cmake variables as advanced since they don't have to be exposed to the user.
mark_as_advanced(FORCE
    CORROSION_BUILD_TESTS
    CORROSION_DEV_MODE
    CORROSION_NATIVE_TOOLING
    CORROSION_NO_WARN_PARSE_TARGET_TRIPLE_FAILED
    CORROSION_RESPECT_OUTPUT_DIRECTORY
    CORROSION_VERBOSE_OUTPUT
    Rust_RESOLVE_RUSTUP_TOOLCHAINS)


# --- Build Dependencies -------------------------------------------------------

include(CTest)

# Use FetchContent to acquire Corrosion for the cmake integration of Rust projects.
include(FetchContent)

FetchContent_Declare(
    Corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.4.7 # Optionally specify a commit hash, version tag or branch here
)
FetchContent_MakeAvailable(Corrosion)

# Generate the cbindgen headers automatically after compilation.
find_program(CBINDGEN_EXECUTABLE NAMES cbindgen DOC "Requires cbindgen to generate the OxiDD headers")


# --- Main Targets -------------------------------------------------------------

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


# --- Dev tools ----------------------------------------------------------------

if(OXIDD_CXX_DOCS)
  add_subdirectory(bindings/cpp/docs)
endif(OXIDD_CXX_DOCS)

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

    # Get all C++ source files
    file(GLOB_RECURSE CXX_SRC_FILES ${CMAKE_CURRENT_SOURCE_DIR}/bindings/cpp/*.[ch]pp)

    # Add `clang-format` target if executable is found
    find_program(CLANG_FORMAT "clang-format")
    if(CLANG_FORMAT)
        add_custom_target(clang-format
            COMMAND ${CLANG_FORMAT} ${CXX_SRC_FILES} -i)
    endif()

    # Add `clang-tidy` target if executable is found
    find_program(CLANG_TIDY "clang-tidy")
    if(CLANG_TIDY)
        add_custom_target(clang-tidy
            COMMAND ${CLANG_TIDY} -p ${CMAKE_BINARY_DIR} ${CXX_SRC_FILES})
    endif()
endif()
