#pragma once

// This file is supposed to provide the definitions for the functions
// that allow us to read a configuration string (potentially from a file, but
// not necessarily) and set up the MLCoupling instance accordingly.

// This entails parsing the configuration, checking in which order the various
// components (normalization, data processor, behavior, application, provider)
// need to be created and initialized, and then creating and initializing them
// with the appropriate parameters obtained from the configuration.

#include <string>
#include <vector>

#include "ml_coupling.hpp"

// Function to create and configure an MLCoupling instance based on a configuration string
template <typename In, typename Out>
MLCoupling<In, Out> create_mlcoupling_from_config(const std::string &config_str);

template <typename In, typename Out>
MLCoupling<In, Out> create_mlcoupling_from_config_file(const std::string &config_file_path);