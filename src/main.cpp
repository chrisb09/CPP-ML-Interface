

// Here we allows for some manual executions to 
// test some things quickly.

// Like:
// - Reading a toml config and creating a MLCoupling from it
// - Testing the behavior of a MLCoupling (like periodic)

// For this, we need to work properly with parameters

#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "ml_coupling.hpp"

#include "config.hpp"

int main(int argc, char** argv) {
    std::cout << "Starting manual test application..." << std::endl;

    // Example: Create a MLCoupling instance directly

    bool something_specified = false;

    bool help_flag = false;
    
    // If flag --config-file <path> is provided, read config from file
    std::string config_file_path;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config-file" && i + 1 < argc) {
            config_file_path = argv[i + 1];
            something_specified = true;
            break;
        }
        if (std::string(argv[i]) == "--help") {
            help_flag = true;
            something_specified = true;
            break;
        }
    }

    if (help_flag) {
        std::cout << "Manual Test Application Help:" << std::endl;
        std::cout << "--config-file <path> : Specify the path to the configuration file to create MLCoupling instance." << std::endl;
        std::cout << "--help               : Show this help message." << std::endl;
        return 0;
    }

    if (!config_file_path.empty()) {
        std::cout << "Reading configuration from file: " << config_file_path << std::endl;
        try {
            MLCoupling<float, float>* ml_coupling = create_mlcoupling_from_config_file<float, float>(config_file_path);
            std::cout << "MLCoupling instance created from config file." << std::endl;
            // Further tests with ml_coupling can be added here
        } catch (const std::exception& e) {
            std::cerr << "Error creating MLCoupling from config file: " << e.what() << std::endl;
        }
    } else {
        std::cout << "No config file provided. Skipping config-based MLCoupling creation." << std::endl;
    }

    // Further manual tests can be added here

    if (!something_specified) {
        std::cout << "No specific tests specified. Use --help for options." << std::endl;
    }

    std::cout << "Manual test application finished." << std::endl;
    return 0;
}