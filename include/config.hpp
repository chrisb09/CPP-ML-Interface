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
#include <fstream>
#include <stdexcept>

// Include toml++ for parsing TOML configurations
#include <toml++/toml.h>

#include "ml_coupling.hpp"

// Function to create and configure an MLCoupling instance based on a configuration string
template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config(const std::string &config_str) {
    try {
        toml::v3::table config = toml::parse(config_str);

        // Let's iterate through the configs "sections" or "categories" or whatever you want to call them (like [provider], [application], etc. but not hardcoded but all that is in the config)

        config.for_each([](const toml::key& key, const toml::v3::node& node) {
            std::cout << "Section: " << key << std::endl;
            if (node.is_table()) {
                const auto& table = *node.as_table();
                table.for_each([](const toml::key& k, auto&& v) {
                    std::cout << "  " << k << " = ";
                    v.visit([](auto&& val) { std::cout << val; });
                    std::cout << std::endl;
                });
            }
        });


        
        
        // Wert auslesen (z.B. ein Port für SmartSim)
        //int64_t port = config["network"]["port"].value_or(6379);
        
        // In deine void* Liste für die Factory packen
        std::vector<void*> params;
        //params.push_back(&port);
        
        // Hier käme dein generierter Factory-Aufruf
        // auto* provider = create_instance_mlcouplingprovider<double, double>("Smartsim", params);

    } catch (const toml::parse_error& err) {
        std::cerr << "Parsing failed: " << err << std::endl;
    }

    return nullptr; // Placeholder return
}

template <typename In, typename Out>
MLCoupling<In, Out>* create_mlcoupling_from_config_file(const std::string &config_file_path) {
    // Open and read the file content
    std::ifstream file(config_file_path);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open config file: " + config_file_path);
    }
    std::string config_str((std::istreambuf_iterator<char>(file)),
                            std::istreambuf_iterator<char>());
    file.close();
    return create_mlcoupling_from_config<In, Out>(config_str);
}