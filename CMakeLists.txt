# Basic Defitions and Version(s)
cmake_minimum_required(VERSION 3.10)
project(HPC_Registry_Project)

# Set C++ standard to C++17
set(CMAKE_CXX_STANDARD 17)

include(CTest)

# Options
option(CPPML_RUN_REGISTRY_TESTS "Run Python registry parser unittests during build" OFF)
option(CPPML_DISABLE_REGISTRY_GEN "Disable registry generation (use externally generated registry)" OFF)

option(WITH_SMARTSIM "Enable SmartSim provider (requires SmartSim dependencies)" ON)
option(WITH_AIX "Enable AIXelerator provider (requires AIXelerator dependencies)" ON)
option(AIX_USE_PREBUILT "Link against a prebuilt AIxelerator library instead of building from source" ON)
option(WITH_PHYDLL "Enable using phyDLL for physics coupling (requires phyDLL dependencies)" OFF)

option(WITH_TORCH "Enable Torch support in AIxelerator (only applies if building AIxelerator from source or using a prebuilt with Torch support)" ON)
option(WITH_TENSORFLOW "Enable TensorFlow support in AIxelerator (only applies if building AIxelerator from source or using a prebuilt with TF support)" OFF)
option(WITH_ONNX "Enable ONNX support in AIxelerator (only applies if building AIxelerator from source or using a prebuilt with ONNX support)" OFF)
option(FORCE_AIX_REBUILD "Force rebuilding AIxelerator from source even if a prebuilt library is found (only applies if AIX_USE_PREBUILT is ON)" OFF)
option(WITH_SCOREP "Enable Score-P manual instrumentation" OFF)
set(CPPML_SCOREP_MPP "mpi" CACHE STRING "Score-P MPP subsystem for CMI targets")

# Set Variables
set(EXECUTABLE_NAME "cpp_ml_interface_executable")
set(LIBRARY_NAME "cpp_ml_interface_library")
set(SMARTREDIS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/SmartRedis")
set(AIXELERATOR_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/AIxeleratorService")
set(AIXELERATOR_PREBUILT_INSTALL_PREFIX "${AIXELERATOR_DIR}/INSTALL-SCOREP" CACHE PATH "Install prefix used for prebuilt AIxelerator")
set(AIXELERATOR_PREBUILT_LIB_DIR "${AIXELERATOR_PREBUILT_INSTALL_PREFIX}/lib" CACHE PATH "Library directory containing prebuilt AIxelerator .so")
set(AIXELERATOR_CMAKE_ARGS "-DWITH_TORCH=ON" CACHE STRING "Additional CMake args used when auto-building prebuilt AIxelerator")
set(PHYDLL_BUILD_DIR "${CMAKE_CURRENT_SOURCE_DIR}/extern/phydll/build" CACHE PATH "PhyDLL build prefix")

message("WITH_SMARTSIM: ${WITH_SMARTSIM}")
message("WITH_AIX: ${WITH_AIX}")
message("AIX_USE_PREBUILT: ${AIX_USE_PREBUILT}")
message("WITH_PHYDLL: ${WITH_PHYDLL}")
message("WITH_TORCH: ${WITH_TORCH}")
message("WITH_TENSORFLOW: ${WITH_TENSORFLOW}")
message("WITH_ONNX: ${WITH_ONNX}")
message("Using SmartRedis directory: ${SMARTREDIS_DIR}")
message("Using AIXelerator directory: ${AIXELERATOR_DIR}")


include(FetchContent)

FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)

FetchContent_MakeAvailable(tomlplusplus)

# Optional: Find and include MPI if available
find_package(MPI)

if(MPI_FOUND)
    message(STATUS "MPI found: ${MPI_CXX_COMPILER}")
    add_compile_definitions(MPI_FOUND)
else()
    message(STATUS "MPI not found, proceeding without MPI support.")
endif()

# 1. Alle Header-Dateien finden, die gescannt werden sollen
file(GLOB_RECURSE ALL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp")

# 2. Pfad für die generierte Datei definieren
set(GENERATED_SRC "${CMAKE_CURRENT_BINARY_DIR}/generated_registry.hpp")

if(NOT CPPML_DISABLE_REGISTRY_GEN)
    # 3. Custom Command für Code-Generierung
    add_custom_target(generate_registry
        COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_registry.py" 
                "${GENERATED_SRC}" "MLCouplingProvider,MLCouplingNormalization,MLCouplingBehavior,MLCouplingApplication" ${ALL_HEADERS}
        DEPENDS ${ALL_HEADERS} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_registry.py"
        COMMENT "Generating registry from headers..."
    )

    # Python regression tests for registry parser (optional)
    if(CPPML_RUN_REGISTRY_TESTS)
        add_custom_target(registry_parser_tests
            COMMAND python3 -m unittest scripts.test_generate_registry_defaults scripts.test_generate_registry_abstract_detection scripts.test_generate_registry_constructor_visibility
            WORKING_DIRECTORY "${CMAKE_CURRENT_SOURCE_DIR}"
            DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_registry.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_generate_registry_defaults.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_generate_registry_abstract_detection.py"
                    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/test_generate_registry_constructor_visibility.py"
            COMMENT "Running registry parser regression tests..."
        )
        add_dependencies(generate_registry registry_parser_tests)
    endif()
endif()

# 4.1 SHARED Library mit tatsächlichen Quelldateien
file(GLOB_RECURSE LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main\\.cpp$")  # main.cpp ausschließen

add_library(${LIBRARY_NAME} SHARED 
    ${LIB_SOURCES}
)

# CMI uses typeid in generated registry (get_type_name functions),
# so it needs RTTI even if the consuming project compiles with -fno-rtti.
target_compile_options(${LIBRARY_NAME} PUBLIC -frtti)

if(NOT CPPML_DISABLE_REGISTRY_GEN)
    # Abhängigkeit zur Code-Generierung
    add_dependencies(${LIBRARY_NAME} generate_registry)
endif()

target_include_directories(${LIBRARY_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include" "${CMAKE_CURRENT_BINARY_DIR}")

# Header-only interface target for consumers that do not link the .so
add_library(cpp_ml_interface_headers INTERFACE)
target_include_directories(cpp_ml_interface_headers INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_link_libraries(cpp_ml_interface_headers INTERFACE tomlplusplus::tomlplusplus)

if (WITH_SCOREP)
    list(APPEND CMAKE_MODULE_PATH "${AIXELERATOR_DIR}/cmake")
    find_package(Scorep REQUIRED)
    if(SCOREP_FOUND)
        scorep_instrument_target(${LIBRARY_NAME} USER ON COMPILER OFF MEMORY ON MPP ${CPPML_SCOREP_MPP})
        scorep_instrument_target(cpp_ml_interface_headers INTERFACE USER ON COMPILER OFF MPP ${CPPML_SCOREP_MPP})
        target_compile_definitions(${LIBRARY_NAME} PUBLIC USE_SCOREP)
        target_compile_definitions(cpp_ml_interface_headers INTERFACE USE_SCOREP)
    endif()
endif()


# Manually set SmartRedis include and library paths
if(WITH_SMARTSIM)
    target_compile_definitions(${LIBRARY_NAME} PUBLIC WITH_SMARTSIM)
    target_compile_definitions(cpp_ml_interface_headers INTERFACE WITH_SMARTSIM)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${SMARTREDIS_DIR}/include)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${SMARTREDIS_DIR}/install/include)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${SMARTREDIS_DIR}/extern/SmartRedis/include)
    target_link_directories(${LIBRARY_NAME} PRIVATE ${SMARTREDIS_DIR}/install/lib64)
    target_link_libraries(${LIBRARY_NAME} PRIVATE smartredis)
    target_include_directories(cpp_ml_interface_headers INTERFACE ${SMARTREDIS_DIR}/include ${SMARTREDIS_DIR}/install/include ${SMARTREDIS_DIR}/extern/SmartRedis/include)
endif()

if(WITH_AIX)
    target_compile_definitions(${LIBRARY_NAME} PUBLIC WITH_AIX)
    target_compile_definitions(cpp_ml_interface_headers INTERFACE WITH_AIX)

    target_include_directories(cpp_ml_interface_headers INTERFACE ${AIXELERATOR_DIR}/src)

    message(STATUS "AIXelerator support enabled.")
    message(STATUS "AIxelerator backend flags: WITH_TORCH=${WITH_TORCH}, WITH_TENSORFLOW=${WITH_TENSORFLOW}, WITH_ONNX=${WITH_ONNX}")

    if(NOT WITH_TORCH AND NOT WITH_TENSORFLOW AND NOT WITH_ONNX)
        message(FATAL_ERROR
            "WITH_AIX is ON, but no AIxelerator backend is enabled. "
            "Enable at least one backend: -DWITH_TORCH=ON or -DWITH_TENSORFLOW=ON or -DWITH_ONNX=ON."
        )
    endif()

    if(AIX_USE_PREBUILT)
        find_library(AIXELERATOR_PREBUILT_LIB
            NAMES AIxeleratorService aixelerator
            PATHS ${AIXELERATOR_PREBUILT_LIB_DIR}
            NO_DEFAULT_PATH
        )

        if(NOT AIXELERATOR_PREBUILT_LIB OR FORCE_AIX_REBUILD)
            if (NOT AIXELERATOR_PREBUILT_LIB)
                message(STATUS "Prebuilt AIxelerator library not found in ${AIXELERATOR_PREBUILT_LIB_DIR}.")
            endif()
            if (FORCE_AIX_REBUILD)
                message(STATUS "FORCE_AIX_REBUILD is ON, forcing rebuild of AIxelerator from source.")
            endif()

            set(AIXELERATOR_PREBUILT_BUILD_DIR "${CMAKE_BINARY_DIR}/aixelerator-prebuilt-build")
            file(MAKE_DIRECTORY "${AIXELERATOR_PREBUILT_BUILD_DIR}")

            set(_aixelerator_cmake_args "${AIXELERATOR_CMAKE_ARGS}")
            separate_arguments(_aixelerator_cmake_args NATIVE_COMMAND "${_aixelerator_cmake_args}")

            message(STATUS "Configuring AIxelerator prebuilt bootstrap build in ${AIXELERATOR_PREBUILT_BUILD_DIR}...")
            execute_process(
                COMMAND ${CMAKE_COMMAND}
                    -S ${AIXELERATOR_DIR}
                    -B ${AIXELERATOR_PREBUILT_BUILD_DIR}
                    -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
                    -DCMAKE_INSTALL_PREFIX=${AIXELERATOR_PREBUILT_INSTALL_PREFIX}
                    -DUSE_PYTHON_TORCH_CMAKE_PREFIX=${USE_PYTHON_TORCH_CMAKE_PREFIX}
                    -DTORCH_VERSION=${TORCH_VERSION}
                    ${_aixelerator_cmake_args}
                RESULT_VARIABLE AIXELERATOR_CONFIGURE_RESULT
            )
            if(NOT AIXELERATOR_CONFIGURE_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to configure AIxelerator prebuilt bootstrap build.")
            endif()

            execute_process(
                COMMAND ${CMAKE_COMMAND} --build ${AIXELERATOR_PREBUILT_BUILD_DIR} -j
                RESULT_VARIABLE AIXELERATOR_BUILD_RESULT
            )
            if(NOT AIXELERATOR_BUILD_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to build AIxelerator prebuilt bootstrap build.")
            endif()

            execute_process(
                COMMAND ${CMAKE_COMMAND} --install ${AIXELERATOR_PREBUILT_BUILD_DIR}
                RESULT_VARIABLE AIXELERATOR_INSTALL_RESULT
            )
            if(NOT AIXELERATOR_INSTALL_RESULT EQUAL 0)
                message(FATAL_ERROR "Failed to install AIxelerator prebuilt bootstrap build.")
            endif()

            find_library(AIXELERATOR_PREBUILT_LIB
                NAMES AIxeleratorService aixelerator
                PATHS ${AIXELERATOR_PREBUILT_LIB_DIR}
                NO_DEFAULT_PATH
            )
        endif()

        if(NOT AIXELERATOR_PREBUILT_LIB)
            message(FATAL_ERROR "AIxelerator prebuilt library not found after bootstrap in ${AIXELERATOR_PREBUILT_LIB_DIR}.")
        endif()

        target_include_directories(${LIBRARY_NAME} PUBLIC ${AIXELERATOR_DIR}/src)
        target_include_directories(${LIBRARY_NAME} PUBLIC ${AIXELERATOR_PREBUILT_INSTALL_PREFIX}/include)
        target_link_libraries(${LIBRARY_NAME} PUBLIC ${AIXELERATOR_PREBUILT_LIB})
        target_include_directories(cpp_ml_interface_headers INTERFACE ${AIXELERATOR_PREBUILT_INSTALL_PREFIX}/include)
        target_link_libraries(cpp_ml_interface_headers INTERFACE ${AIXELERATOR_PREBUILT_LIB})
    else()
        target_include_directories(${LIBRARY_NAME} PUBLIC ${AIXELERATOR_DIR}/src)
        if(NOT TARGET AIxeleratorServiceLib)
            add_subdirectory(${AIXELERATOR_DIR} ${CMAKE_BINARY_DIR}/aixelerator-build)
        endif()
        target_link_libraries(${LIBRARY_NAME} PUBLIC AIxeleratorServiceLib)
        target_link_libraries(cpp_ml_interface_headers INTERFACE AIxeleratorServiceLib)
    endif()
endif()

if(WITH_PHYDLL)
    target_compile_definitions(${LIBRARY_NAME} PUBLIC WITH_PHYDLL)
    target_compile_definitions(cpp_ml_interface_headers INTERFACE WITH_PHYDLL)
    target_include_directories(${LIBRARY_NAME} PUBLIC ${PHYDLL_BUILD_DIR}/include)
    target_link_directories(${LIBRARY_NAME} PUBLIC ${PHYDLL_BUILD_DIR}/lib)
    target_link_libraries(${LIBRARY_NAME} PUBLIC phydll)
    target_include_directories(cpp_ml_interface_headers INTERFACE ${PHYDLL_BUILD_DIR}/include)
    target_link_directories(cpp_ml_interface_headers INTERFACE ${PHYDLL_BUILD_DIR}/lib)
    target_link_libraries(cpp_ml_interface_headers INTERFACE phydll)
    add_subdirectory(dl_clients)
endif()

# TOML++ verlinken
target_link_libraries(${LIBRARY_NAME} PUBLIC tomlplusplus::tomlplusplus)

# 4.2 Executable erstellen
add_executable(${EXECUTABLE_NAME} src/main.cpp)

target_include_directories(${EXECUTABLE_NAME} PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")

# Libraries verlinken
target_link_libraries(${EXECUTABLE_NAME} PRIVATE 
    ${LIBRARY_NAME}
    tomlplusplus::tomlplusplus
)

if (WITH_SCOREP)
    set_target_properties(${EXECUTABLE_NAME} PROPERTIES SCOREP_MPP_SYSTEM "${CPPML_SCOREP_MPP}")
endif()

if(WITH_SMARTSIM)
    target_link_directories(${EXECUTABLE_NAME} PRIVATE ${SMARTREDIS_DIR}/install/lib64)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE smartredis)
endif()

if(WITH_PHYDLL)
    target_link_directories(${EXECUTABLE_NAME} PRIVATE ${PHYDLL_BUILD_DIR}/lib)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE phydll)
endif()

if (MPI_FOUND)
    target_compile_definitions(${LIBRARY_NAME} PUBLIC MPI_FOUND)
    target_compile_definitions(cpp_ml_interface_headers INTERFACE MPI_FOUND)
    target_link_libraries(${LIBRARY_NAME} PUBLIC MPI::MPI_CXX)
    target_link_libraries(${EXECUTABLE_NAME} PRIVATE MPI::MPI_CXX)
    target_link_libraries(cpp_ml_interface_headers INTERFACE MPI::MPI_CXX)
endif()

if(BUILD_TESTING)
    add_executable(ml_coupling_data_tests test/test_ml_coupling_data.cpp)
    target_include_directories(ml_coupling_data_tests PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_test(NAME ml_coupling_data_tests COMMAND ml_coupling_data_tests)

    add_executable(test_flexible_fallback test/test_flexible_fallback.cpp)
    target_include_directories(test_flexible_fallback PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    target_link_libraries(test_flexible_fallback PRIVATE cpp_ml_interface_library)
    if (WITH_SCOREP)
        set_target_properties(test_flexible_fallback PROPERTIES SCOREP_MPP_SYSTEM "${CPPML_SCOREP_MPP}")
    endif()
    add_test(NAME test_flexible_fallback COMMAND test_flexible_fallback)

    add_executable(test_behavior_flow_extrapolator test/test_behavior_flow_extrapolator.cpp)
    target_include_directories(test_behavior_flow_extrapolator PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
    add_test(NAME test_behavior_flow_extrapolator COMMAND test_behavior_flow_extrapolator)

    # --- Provider inference integration test ---
    # Requires a TorchScript model generated by scripts/create_test_model.py.
    # The model is created via a CTest fixture so it is only generated once
    # per test run, even if the test is re-run with --rerun-failed.
    set(TEST_PYTHON_EXECUTABLE "" CACHE FILEPATH
        "Python executable used to generate integration-test models (defaults to python3 in PATH)")
    if(NOT TEST_PYTHON_EXECUTABLE)
        find_program(TEST_PYTHON_EXECUTABLE NAMES python3 python)
    endif()

    if(TEST_PYTHON_EXECUTABLE)
        set(TEST_MODEL_PATH "${CMAKE_CURRENT_BINARY_DIR}/test_addition_model.pt")

        # Fixture: generate the TorchScript model once before the test runs
        add_test(NAME generate_test_model
            COMMAND "${TEST_PYTHON_EXECUTABLE}"
                    "${CMAKE_CURRENT_SOURCE_DIR}/scripts/create_test_model.py"
                    "${TEST_MODEL_PATH}"
        )
        set_tests_properties(generate_test_model PROPERTIES
            FIXTURES_SETUP   InferenceModelFixture
            LABELS           "fixture"
        )

        add_executable(test_provider_inference test/test_provider_inference.cpp)
        target_include_directories(test_provider_inference PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/include")
        target_link_libraries(test_provider_inference PRIVATE cpp_ml_interface_library)
        if (WITH_SCOREP)
            set_target_properties(test_provider_inference PROPERTIES SCOREP_MPP_SYSTEM "${CPPML_SCOREP_MPP}")
        endif()

        add_test(NAME test_provider_inference
            COMMAND test_provider_inference "${TEST_MODEL_PATH}"
        )
        set_tests_properties(test_provider_inference PROPERTIES
            FIXTURES_REQUIRED InferenceModelFixture
            LABELS            "integration"
        )
    else()
        message(WARNING
            "No Python executable found; skipping test_provider_inference. "
            "Set -DTEST_PYTHON_EXECUTABLE=/path/to/python3 to enable it.")
    endif()
endif()
