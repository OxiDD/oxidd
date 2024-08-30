set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Workaround for https://corrosion-rs.github.io/corrosion/common_issues.html#linking-debug-cc-libraries-into-rust-fails-on-windows-msvc-targets
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreadedDLL")

add_executable(boolean-function boolean-function.cpp)
target_link_libraries(boolean-function oxidd)
add_test(NAME boolean-function COMMAND boolean-function)
