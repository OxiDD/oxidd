cmake_minimum_required(VERSION 3.15)

project(OxiDD
    VERSION 0.11.1
    DESCRIPTION "Concurrent Decision Diagram Library"
    HOMEPAGE_URL "https://oxidd.net"
    LANGUAGES CXX)

# spell-checker:ignore CTEST,INCLUDEDIR,STREQUAL,LIBDIR

include(CTest)
include(FetchContent) # to acquire corrosion and tl-expected
include(GNUInstallDirs)

# --- Settings -----------------------------------------------------------------

if(PROJECT_IS_TOP_LEVEL)
    set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
endif()

# This could be derived from the $<CONFIG> if desired.
set(OXIDD_CARGO_PROFILE "release" CACHE STRING
    "Sets the build profile used for Rust dependencies (`dev`, `release`, or `release-with-debug`)")
set_property(CACHE OXIDD_CARGO_PROFILE PROPERTY STRINGS
    "dev" "release" "release-with-debug")

option(OXIDD_ENABLE_CXX "Enable C++ bindings" ON)

option(OXIDD_COMPAT_ENABLE_TL_EXPECTED
    "Enable the tl-expected library for compatibility purposes. `std::expected` was introduced in C++23. When using an older C++ standard or standard library, the tl-expected library will be used as a fallback."
    ON)
option(OXIDD_COMPAT_EXTERNAL_TL_EXPECTED
    "Use the system-installed version of tl-expected instead of fetching it from the Git repository. Only effective if OXIDD_COMPAT_ENABLE_TL_EXPECTED is set."
    OFF)

# Mark some cmake variables as advanced since they don't have to be exposed to the user.
mark_as_advanced(FORCE
    CORROSION_BUILD_TESTS
    CORROSION_DEV_MODE
    CORROSION_NATIVE_TOOLING
    CORROSION_NO_WARN_PARSE_TARGET_TRIPLE_FAILED
    CORROSION_RESPECT_OUTPUT_DIRECTORY
    CORROSION_VERBOSE_OUTPUT
    Rust_RESOLVE_RUSTUP_TOOLCHAINS)

# Generate config.h
configure_file(
    ${PROJECT_SOURCE_DIR}/bindings/cpp/config.h.in
    ${PROJECT_BINARY_DIR}/include/oxidd/config.h)


# --- Build Dependencies -------------------------------------------------------

FetchContent_Declare(corrosion
    GIT_REPOSITORY https://github.com/corrosion-rs/corrosion.git
    GIT_TAG v0.5.2)
FetchContent_MakeAvailable(corrosion)

find_program(CBINDGEN_EXECUTABLE
    NAMES cbindgen
    DOC "Requires cbindgen to generate the OxiDD C API header"
    REQUIRED)

# --- Dependencies -------------------------------------------------------------

if(OXIDD_COMPAT_ENABLE_TL_EXPECTED)
    set(EXPECTED_BUILD_TESTS OFF)
    set(EXPECTED_BUILD_PACKAGE OFF)
    if(OXIDD_COMPAT_EXTERNAL_TL_EXPECTED)
        find_package(tl-expected REQUIRED)
    else()
        FetchContent_Declare(tl-expected
            GIT_REPOSITORY https://github.com/TartanLlama/expected.git
            GIT_TAG v1.3.1
            EXCLUDE_FROM_ALL)
        FetchContent_MakeAvailable(tl-expected)
    endif()
endif()


# --- Main Targets -------------------------------------------------------------

# Generate oxidd/capi.h using cbindgen
set(oxidd_capi_include ${PROJECT_BINARY_DIR}/include)
set(oxidd_capi_h ${oxidd_capi_include}/oxidd/capi.h)
file(GLOB_RECURSE oxidd_ffi_c_sources CONFIGURE_DEPENDS
    ${PROJECT_SOURCE_DIR}/crates/oxidd-core/*.rs
    ${PROJECT_SOURCE_DIR}/crates/oxidd-ffi-c/*.rs)
add_custom_command(
    OUTPUT ${oxidd_capi_h}
    COMMAND ${CBINDGEN_EXECUTABLE}
    ARGS --output ${oxidd_capi_h} ${PROJECT_SOURCE_DIR}/crates/oxidd-ffi-c
    DEPENDS ${oxidd_ffi_c_sources} ${PROJECT_SOURCE_DIR}/crates/oxidd-ffi-c/cbindgen.toml
    COMMENT "Generating oxidd/capi.h using cbindgen"
    VERBATIM)
# Custom target such that targets in other directories can depend on the header
add_custom_target(oxidd-capi-header ALL DEPENDS ${oxidd_capi_h})

# Import the oxidd-ffi-c crate specifically
corrosion_import_crate(
    MANIFEST_PATH Cargo.toml
    PROFILE ${OXIDD_CARGO_PROFILE}
    CRATES oxidd-ffi-c
    FEATURES $<$<BOOL:${OXIDD_ENABLE_CXX}>:cpp>
    CRATE_TYPES cdylib staticlib)
add_dependencies(oxidd_ffi_c oxidd-capi-header)
if(OXIDD_COMPAT_ENABLE_TL_EXPECTED)
    target_link_libraries(oxidd_ffi_c INTERFACE tl::expected)
endif()
target_include_directories(oxidd_ffi_c INTERFACE
    ${oxidd_capi_include}
    ${PROJECT_SOURCE_DIR}/bindings/cpp/include)

add_library(oxidd ALIAS oxidd_ffi_c)

set(CMAKE_SKIP_TEST_ALL_DEPENDENCY FALSE)
add_subdirectory(bindings/cpp/tests)
add_subdirectory(bindings/cpp/examples)

# --- Installation -------------------------------------------------------------

# C++ headers
install(DIRECTORY ${PROJECT_SOURCE_DIR}/bindings/cpp/include/oxidd
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
# C header
install(FILES ${oxidd_capi_h}
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/oxidd)

if(Rust_CARGO_TARGET_OS STREQUAL "windows")
    if(Rust_CARGO_TARGET_ENV STREQUAL "msvc")
        set(lib_prefix "")
        set(lib_suffix "lib")
    elseif(Rust_CARGO_TARGET_ENV STREQUAL "gnu")
        set(lib_prefix "lib")
        set(lib_suffix "a")
    else()
        message(FATAL_ERROR "Unknown Windows environment")
    endif()

    # dynamic
    install(FILES ${PROJECT_BINARY_DIR}/oxidd_ffi_c.dll
        TYPE LIB
        RENAME oxidd.dll)
    # dynamic implib
    install(FILES ${PROJECT_BINARY_DIR}/${lib_prefix}oxidd_ffi_c.dll.${lib_suffix}
        TYPE LIB
        RENAME ${lib_prefix}oxidd.dll.${lib_suffix})
    # static
    install(FILES ${PROJECT_BINARY_DIR}/${lib_prefix}liboxidd_ffi_c.${lib_suffix}
        TYPE LIB
        RENAME ${lib_prefix}liboxidd.${lib_suffix})
else()
    if(Rust_CARGO_TARGET_OS STREQUAL "darwin")
        set(so_suffix "dylib")
        set(so_ver_suffix "${PROJECT_VERSION}.dylib")
        set(so_ver_major_suffix "${PROJECT_VERSION_MAJOR}.dylib")
    else()
        set(so_suffix "so")
        set(so_ver_suffix "so.${PROJECT_VERSION}")
        set(so_ver_major_suffix "so.${PROJECT_VERSION_MAJOR}")
    endif()

    # dynamic
    install(FILES ${PROJECT_BINARY_DIR}/liboxidd_ffi_c.${so_suffix}
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RENAME liboxidd.${so_ver_suffix})
    install(CODE "file(CREATE_LINK \"liboxidd.${so_ver_suffix}\" \"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/liboxidd.${so_suffix}\" SYMBOLIC)")
    install(CODE "file(CREATE_LINK \"liboxidd.${so_ver_suffix}\" \"${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}/liboxidd.${so_ver_major_suffix}\" SYMBOLIC)")
    # static
    install(FILES ${PROJECT_BINARY_DIR}/liboxidd_ffi_c.a
        DESTINATION ${CMAKE_INSTALL_LIBDIR}
        RENAME liboxidd.a)
endif()

# --- Dev tools ----------------------------------------------------------------

add_subdirectory(bindings/cpp/doc)
