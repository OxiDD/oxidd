add_executable(boolean-function boolean-function.cpp)
target_link_libraries(boolean-function oxidd)
add_test(NAME boolean-function COMMAND boolean-function)
