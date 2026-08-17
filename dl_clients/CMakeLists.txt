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

if(WITH_SCOREP)
    list(APPEND CMAKE_MODULE_PATH "${CPP_MODULE_DIR}/extern/AIxeleratorService/cmake")
    find_package(Scorep REQUIRED)
    if(SCOREP_FOUND)
        if(NOT CPPML_SCOREP_MPP)
            set(CPPML_SCOREP_MPP "mpi")
        endif()
        execute_process(COMMAND ${SCOREP_CONFIG_EXECUTABLE} --prefix OUTPUT_VARIABLE SCOREP_PREFIX_OUT OUTPUT_STRIP_TRAILING_WHITESPACE)
        set(SCOREP_INC_DIR "${SCOREP_PREFIX_OUT}/include")
        link_directories("${SCOREP_PREFIX_OUT}/lib" "${SCOREP_PREFIX_OUT}/lib/scorep" "${SMARTSIM_PAPI_ROOT}/lib")
        set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -L${SCOREP_PREFIX_OUT}/lib -L${SCOREP_PREFIX_OUT}/lib/scorep")
        set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -L${SCOREP_PREFIX_OUT}/lib -L${SCOREP_PREFIX_OUT}/lib/scorep")
    endif()
endif()

add_library(phydll_dl_runtime SHARED
    phydll_dl_runtime.cpp
)

target_include_directories(phydll_dl_runtime PRIVATE ${PHYDLL_BUILD_DIR}/include)
if(WITH_SCOREP AND SCOREP_FOUND)
    target_include_directories(phydll_dl_runtime BEFORE PRIVATE "${SCOREP_INC_DIR}")
endif()
target_link_directories(phydll_dl_runtime PRIVATE ${PHYDLL_BUILD_DIR}/lib)
target_link_libraries(phydll_dl_runtime PRIVATE phydll)

add_executable(phydll_dl_client dl_client.cpp)


target_include_directories(phydll_dl_client PRIVATE ${PHYDLL_BUILD_DIR}/include)
if(WITH_SCOREP AND SCOREP_FOUND)
    target_include_directories(phydll_dl_client BEFORE PRIVATE "${SCOREP_INC_DIR}")
endif()
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

if(MPI_FOUND)
    # MPMD regression harness: 2 solver ranks + 1 DL rank. Links the full
    # CPP-ML-Interface library so the PhyDLL provider template is instantiated.
    add_executable(phydll_phy_test EXCLUDE_FROM_ALL
        ../test/phydll_phy_test.cpp
    )

    target_include_directories(phydll_phy_test PRIVATE
        ${CPP_MODULE_DIR}/include
        ${CPP_MODULE_DIR}/build
        ${PHYDLL_BUILD_DIR}/include
    )
    target_link_directories(phydll_phy_test PRIVATE
        ${PHYDLL_BUILD_DIR}/lib
        ${CPP_MODULE_DIR}/build
    )
    target_link_libraries(phydll_phy_test PRIVATE phydll cpp_ml_interface_library MPI::MPI_CXX)
    target_compile_definitions(phydll_phy_test PRIVATE WITH_PHYDLL MPI_FOUND)
endif()

if(WITH_SCOREP AND SCOREP_FOUND)
    # Keep MPI interception consistent with the solver binary.
    scorep_instrument_target(phydll_dl_runtime USER ON COMPILER OFF MEMORY ON MPP ${CPPML_SCOREP_MPP})
    scorep_instrument_target(phydll_dl_client USER ON COMPILER OFF MEMORY ON MPP ${CPPML_SCOREP_MPP})
    target_compile_options(phydll_dl_runtime PRIVATE "-I${SCOREP_PREFIX_OUT}/include")
    target_compile_options(phydll_dl_client PRIVATE "-I${SCOREP_PREFIX_OUT}/include")
    target_link_directories(phydll_dl_runtime PRIVATE "${SCOREP_PREFIX_OUT}/lib" "${SMARTSIM_PAPI_ROOT}/lib")
    target_link_directories(phydll_dl_client PRIVATE "${SCOREP_PREFIX_OUT}/lib" "${SMARTSIM_PAPI_ROOT}/lib")
    target_link_libraries(phydll_dl_runtime PRIVATE Scorep::Plugin)
    target_link_libraries(phydll_dl_client PRIVATE Scorep::Plugin)
    target_link_options(phydll_dl_client PRIVATE -no-pie)
    target_compile_definitions(phydll_dl_runtime PRIVATE USE_SCOREP)
    target_compile_definitions(phydll_dl_client PRIVATE USE_SCOREP)
    if(TARGET phydll_phy_test)
        scorep_instrument_target(phydll_phy_test USER ON COMPILER OFF MEMORY ON MPP ${CPPML_SCOREP_MPP})
        target_compile_options(phydll_phy_test PRIVATE "-I${SCOREP_PREFIX_OUT}/include")
        target_link_directories(phydll_phy_test PRIVATE "${SCOREP_PREFIX_OUT}/lib" "${SMARTSIM_PAPI_ROOT}/lib")
        target_link_libraries(phydll_phy_test PRIVATE Scorep::Plugin)
        target_compile_definitions(phydll_phy_test PRIVATE USE_SCOREP)
    endif()
endif()
