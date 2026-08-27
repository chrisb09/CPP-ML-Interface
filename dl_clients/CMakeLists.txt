cmake_minimum_required(VERSION 3.10)

# Prevent CUDA stubs from being added to the RUNPATH/RPATH of binaries compiled on CPU login nodes
list(APPEND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/lib/stubs"
  "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib/stubs"
)
list(APPEND CMAKE_CXX_IMPLICIT_LINK_DIRECTORIES
  "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/lib/stubs"
  "/cvmfs/software.hpc.rwth.de/Linux/RH9/x86_64/intel/sapphirerapids/software/CUDA/12.4.0/targets/x86_64-linux/lib/stubs"
)

# Honor <Package>_ROOT variables like CUDA_ROOT.
cmake_policy(SET CMP0074 NEW)

if(WITH_SCOREP OR (DEFINED ENV{USE_SCOREP} AND NOT "$ENV{USE_SCOREP}" STREQUAL "" AND NOT "$ENV{USE_SCOREP}" STREQUAL "0"))
    set(CMAKE_CXX_COMPILER scorep-mpicxx)
    set(CMAKE_C_COMPILER scorep-mpicc)
    set(CMAKE_CXX_COMPILER_ARG1 "--nocompiler --user")
    set(CMAKE_C_COMPILER_ARG1 "--nocompiler --user")
endif()

project(PhyDLLDlClient LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CPP_MODULE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/..")

option(PHYDLL_DL_WITH_TORCH "Build DL client with libtorch" ON)
option(WITH_SCOREP "Enable Score-P instrumentation for the PhyDLL DL client" OFF)
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
target_compile_options(phydll_dl_client PRIVATE -frtti)


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

# MPMD regression harness: 2 solver ranks + 1 DL rank. Header-only: the PhyDLL
# provider and MLCouplingData are template headers, so no CPP-ML-Interface
# shared library is required. This keeps the target buildable when dl_clients is
# configured standalone (as the module_test matrix does).
add_executable(phydll_phy_test
    ../test/phydll_phy_test.cpp
)

target_include_directories(phydll_phy_test PRIVATE
    ${CPP_MODULE_DIR}/include
    ${PHYDLL_BUILD_DIR}/include
)
target_link_directories(phydll_phy_test PRIVATE
    ${PHYDLL_BUILD_DIR}/lib
)
target_link_libraries(phydll_phy_test PRIVATE phydll MPI::MPI_CXX)
target_compile_definitions(phydll_phy_test PRIVATE WITH_PHYDLL MPI_FOUND)

if(WITH_SCOREP)
    list(APPEND CMAKE_MODULE_PATH "${CPP_MODULE_DIR}/extern/AIxeleratorService/cmake")
    find_package(Scorep REQUIRED)
    if(SCOREP_FOUND)
        if(NOT CPPML_SCOREP_MPP)
            set(CPPML_SCOREP_MPP "mpi")
        endif()
        # scorep_instrument_target is NOT called here: when building with the
        # scorep-mpicxx compiler wrapper, the wrapper already injects scorep_init.o
        # at link time. Calling scorep_instrument_target additionally would produce
        # a duplicate-symbol linker error (scorep_subsystems defined twice).
        target_include_directories(phydll_dl_runtime PRIVATE ${SCOREP_ROOT_DIR}/include ${SCOREP_ROOT_DIR}/include/scorep)
        target_include_directories(phydll_dl_client PRIVATE ${SCOREP_ROOT_DIR}/include ${SCOREP_ROOT_DIR}/include/scorep)
        if(TARGET phydll_phy_test)
            target_include_directories(phydll_phy_test PRIVATE ${SCOREP_ROOT_DIR}/include ${SCOREP_ROOT_DIR}/include/scorep)
        endif()
        target_link_libraries(phydll_dl_runtime PRIVATE Scorep::Plugin)
        target_link_libraries(phydll_dl_client PRIVATE Scorep::Plugin)
        target_link_options(phydll_dl_client PRIVATE -no-pie)
    endif()
    target_compile_definitions(phydll_dl_runtime PRIVATE USE_SCOREP)
    target_compile_definitions(phydll_dl_client PRIVATE USE_SCOREP)
    if(TARGET phydll_phy_test)
        target_compile_definitions(phydll_phy_test PRIVATE USE_SCOREP)
    endif()
endif()
