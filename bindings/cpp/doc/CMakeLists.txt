find_package(Doxygen)

# spell-checker:ignore TAGFILE,cppreference

if(DOXYGEN)
    include(FetchContent)

    # TAGFILE for linking to cppreference.com (see https://stackoverflow.com/a/44743279)
    FetchContent_Declare(cppreference_tags
        # The following URL is (currently) not working, so we (temporarily) use our mirror below
        #URL https://upload.cppreference.com/mwiki/images/f/f8/cppreference-doxygen-web.tag.xml
        URL https://oxidd.net/api/cpp/cppreference-doxygen-web.tag.xml
        #DOWNLOAD_DIR ${CMAKE_CURRENT_BINARY_DIR}
        DOWNLOAD_NO_EXTRACT ON)

    FetchContent_Declare(doxygen_awesome_css
        GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
        GIT_TAG v2.3.3)

    FetchContent_MakeAvailable(cppreference_tags doxygen_awesome_css)

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
