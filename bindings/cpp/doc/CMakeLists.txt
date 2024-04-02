find_package(Doxygen)

if(DOXYGEN)
    # configure doxygen files
    set(DOXYGEN_IN ${CMAKE_CURRENT_SOURCE_DIR}/Doxyfile.in)
    set(DOXYGEN_OUT ${CMAKE_CURRENT_BINARY_DIR}/Doxyfile)

    configure_file(${DOXYGEN_IN} ${DOXYGEN_OUT} @ONLY)

    # add target
    add_custom_target(oxidd-doc
        COMMAND ${DOXYGEN_EXECUTABLE} ${DOXYGEN_OUT}
        WORKING_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}
        COMMENT "Generating API documentation with Doxygen"
        VERBATIM)
    add_dependencies(oxidd-doc oxidd-capi-header)
else(DOXYGEN)
    message(STATUS "Doxygen not found, OxiDD API documentation cannot be generated.")
endif(DOXYGEN)
