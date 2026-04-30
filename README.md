

## Requirements

CMake
C++ compiler supporting C++17
Python 3.x
    clang 16.0.6
Clang 16.x

## Installation

The python dependencies can be installed via pip:

```bash
pip install -r requirements.txt
```

`./build.sh` also runs registry parser regression tests automatically.

## Test Usage

The CMakeLists.txt file includes two targets, one to actually compile a `libcpp_ml_interface_library.so` file, and another to create an executable `cpp_ml_interface_executable` that primarily acts as a first line of verification for the library. You can execute it and provide parameters, including a config file to check if it can be parsed and the subsequent `MLCoupling` object can be created successfully. Additionally, you can specify a number of steps for which the behavior class executes essentially a dummy loop to see when coupling and inference calls will be made.

**Example**:
```bash
./build.sh
./build/cpp_ml_interface_executable --config-file example.config.toml --behavior 100
```

`build.sh` actually does two builds, one normal and one with -O3 optimizations, which are placed in `build/` and `build_release/` respectively.


## Including in Your Own Project

WIP

these are untested notes:


```cmake
# Path to the prebuilt CPP-ML-Interface .so
# Probably not really needed unless we use the c api
set(CPPML_SO "/path/to/CPP-ML-Interface/build/libcpp_ml_interface_library.so")

# Path to the CPP-ML-Interface headers
set(CPPML_INCLUDE_DIR "/path/to/CPP-ML-Interface/include")

add_library(cppml_interface SHARED IMPORTED)
set_target_properties(cppml_interface PROPERTIES
    IMPORTED_LOCATION "${CPPML_SO}"
    INTERFACE_INCLUDE_DIRECTORIES "${CPPML_INCLUDE_DIR}"
)

# Link it into your terrain_solver
target_link_libraries(${PROJECT_NAME} PRIVATE cppml_interface)
```