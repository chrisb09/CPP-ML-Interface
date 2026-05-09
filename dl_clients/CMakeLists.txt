cmake_minimum_required(VERSION 3.10)

project(PhyDLLDlClient LANGUAGES CXX)

option(PHYDLL_DL_WITH_TORCH "Build DL client with libtorch" ON)
set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/libtorch" CACHE PATH "Path to libtorch root (contains share/cmake/Torch)")

add_executable(phydll_dl_client dl_client.cpp)

if(NOT PHYDLL_BUILD_DIR)
    set(PHYDLL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/phydll/build" CACHE PATH "PhyDLL build prefix")
endif()

target_include_directories(phydll_dl_client PRIVATE ${PHYDLL_BUILD_DIR}/include)
target_link_directories(phydll_dl_client PRIVATE ${PHYDLL_BUILD_DIR}/lib)
target_link_libraries(phydll_dl_client PRIVATE phydll)

if(PHYDLL_DL_WITH_TORCH)
    if(LIBTORCH_DIR AND EXISTS "${LIBTORCH_DIR}/share/cmake/Torch/TorchConfig.cmake")
        list(APPEND CMAKE_PREFIX_PATH "${LIBTORCH_DIR}")
    endif()
    find_package(Torch QUIET)
    if(Torch_FOUND)
        target_link_libraries(phydll_dl_client PRIVATE ${TORCH_LIBRARIES})
        target_compile_definitions(phydll_dl_client PRIVATE PHYDLL_DL_USE_TORCH)
        message(STATUS "Torch found; building DL client with libtorch.")
    else()
        message(WARNING "Torch not found; building DL client without libtorch.")
    endif()
endif()

find_package(MPI QUIET)

if(MPI_FOUND)
    target_link_libraries(phydll_dl_client PRIVATE MPI::MPI_CXX)
endif()
