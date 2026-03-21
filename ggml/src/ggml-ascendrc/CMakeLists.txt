find_library(ASCEND_BLAS
    NAMES libascblas.a
    PATHS "/usr/local/lib"
    NO_DEFAULT_PATH
)
set(ASCEND_BLAS_INCLUDE_DIR /usr/local/include)

if ("cann${CANN_INSTALL_DIR}" STREQUAL "cann" AND DEFINED ENV{ASCEND_TOOLKIT_HOME})
    set(CANN_INSTALL_DIR $ENV{ASCEND_TOOLKIT_HOME})
    message(STATUS "CANN: updated CANN_INSTALL_DIR from ASCEND_TOOLKIT_HOME=$ENV{ASCEND_TOOLKIT_HOME}")
endif()

set(CANN_INCLUDE_DIRS
    ${CANN_INSTALL_DIR}/include
    ${CANN_INSTALL_DIR}/include/aclnn
    ${CANN_INSTALL_DIR}/acllib/include
)

ggml_add_backend_library(ggml-ascendrc
                             ggml-ascendrc.cpp
                            )

find_library(TILING_API NAMES libtiling_api.a PATHS "${CANN_INSTALL_DIR}/aarch64-linux/lib64" NO_DEFAULT_PATH)
target_link_libraries(ggml-ascendrc PRIVATE ${ASCEND_BLAS} ${TILING_API} ascendcl platform)
target_include_directories(ggml-ascendrc PRIVATE ${ASCEND_BLAS_INCLUDE_DIR} ${CANN_INCLUDE_DIRS})
target_include_directories(ggml-ascendrc PRIVATE ${CANN_INCLUDE_DIRS})
target_link_directories(ggml-ascendrc PRIVATE ${CANN_INSTALL_DIR}/lib64 ${CANN_INSTALL_DIR}/runtime/lib64 ${CANN_INSTALL_DIR}/aarch64-linux/lib64)
