cmake_minimum_required(VERSION 3.10)

# Honor <Package>_ROOT variables like CUDA_ROOT.
cmake_policy(SET CMP0074 NEW)

project(PhyDLLDlClient LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(PHYDLL_DL_WITH_TORCH "Build DL client with libtorch" ON)
set(LIBTORCH_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/libtorch" CACHE PATH "Path to libtorch root (contains share/cmake/Torch)")

if(NOT PHYDLL_BUILD_DIR)
    set(PHYDLL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/../extern/phydll/build" CACHE PATH "PhyDLL build prefix")
endif()

add_library(phydll_dl_runtime SHARED
    phydll_dl_runtime.cpp
)

target_include_directories(phydll_dl_runtime PRIVATE ${PHYDLL_BUILD_DIR}/include)
target_link_directories(phydll_dl_runtime PRIVATE ${PHYDLL_BUILD_DIR}/lib)
target_link_libraries(phydll_dl_runtime PRIVATE phydll)

add_executable(phydll_dl_client dl_client.cpp)


target_include_directories(phydll_dl_client PRIVATE ${PHYDLL_BUILD_DIR}/include)
target_link_directories(phydll_dl_client PRIVATE ${PHYDLL_BUILD_DIR}/lib)
target_link_libraries(phydll_dl_client PRIVATE phydll phydll_dl_runtime)

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

if(WITH_SCOREP)
    scorep_instrument_target(phydll_dl_runtime USER ON COMPILER OFF MEMORY ON MPP mpi)
    scorep_instrument_target(phydll_dl_client USER ON COMPILER OFF MEMORY ON MPP mpi)
    target_compile_definitions(phydll_dl_runtime PRIVATE USE_SCOREP)
    target_compile_definitions(phydll_dl_client PRIVATE USE_SCOREP)
endif()
