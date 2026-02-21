cmake_minimum_required(VERSION 3.10)
project(HPC_Registry_Project)

set(CMAKE_CXX_STANDARD 17)


set(EXECUTABLE_NAME "cpp_ml_interface_executable")
set(LIBRARY_NAME "cpp_ml_interface_library")

include(FetchContent)

FetchContent_Declare(
    tomlplusplus
    GIT_REPOSITORY https://github.com/marzer/tomlplusplus.git
    GIT_TAG        v3.4.0
)

FetchContent_MakeAvailable(tomlplusplus)

# 1. Alle Header-Dateien finden, die gescannt werden sollen
file(GLOB_RECURSE ALL_HEADERS "${CMAKE_CURRENT_SOURCE_DIR}/include/**/*.hpp")

# 2. Pfad für die generierte Datei definieren
set(GENERATED_SRC "${CMAKE_CURRENT_SOURCE_DIR}/include/generated_registry.hpp")

# 3. Custom Command für Code-Generierung
add_custom_command(
    OUTPUT "${GENERATED_SRC}"
    COMMAND python3 "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_registry.py" 
            "${GENERATED_SRC}" "MLCouplingProvider,MLCouplingNormalization,MLCouplingBehavior,MLCouplingApplication" ${ALL_HEADERS}
    DEPENDS ${ALL_HEADERS} "${CMAKE_CURRENT_SOURCE_DIR}/scripts/generate_registry.py"
    COMMENT "Generating registry from headers..."
)

# Custom Target damit die generierte Datei gebaut wird
add_custom_target(generate_registry DEPENDS "${GENERATED_SRC}")

# 4.1 SHARED Library mit tatsächlichen Quelldateien
file(GLOB_RECURSE LIB_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")
list(FILTER LIB_SOURCES EXCLUDE REGEX ".*main\\.cpp$")  # main.cpp ausschließen

add_library(${LIBRARY_NAME} SHARED 
    ${LIB_SOURCES}
    ${GENERATED_SRC}  # Generierte Datei mit einbeziehen
)

# Abhängigkeit zur Code-Generierung
add_dependencies(${LIBRARY_NAME} generate_registry)

target_include_directories(${LIBRARY_NAME} PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/include")

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