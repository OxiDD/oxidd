set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_executable(boolean-function boolean-function.cpp)
target_link_libraries(boolean-function oxidd)
add_test(NAME boolean-function COMMAND boolean-function)
