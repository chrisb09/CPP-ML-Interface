

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
#include <iomanip>
#include <cassert>

#include "ml_coupling.hpp"

#include "config.hpp"

int main(int argc, char** argv) {
    std::cout << "Starting manual test application..." << std::endl;

    // Example: Create a MLCoupling instance directly

    bool something_specified = false;

    bool help_flag = false;
    
    // If flag --config-file <path> is provided, read config from file
    std::string config_file_path;
    int max_step = -1;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--config-file" && i + 1 < argc) {
            config_file_path = argv[i + 1];
            something_specified = true;
            i++; // Skip the next argument since it's the config file path
            continue;
        }
        if (std::string(argv[i]) == "--behavior" && i + 1 < argc) {
            max_step = std::stoi(argv[i + 1]);
            assert(max_step >= 0 && "Max step must be non-negative");
            std::cout << "Testing periodic behavior up to step " << max_step << "." << std::endl;
            i++;
            continue;
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
        std::cout << "--behavior <steps>   : Simulate the behavior of the created MLCoupling instance for a given number of steps." << std::endl;
        std::cout << "--help               : Show this help message." << std::endl;
        return 0;
    }

    if (!config_file_path.empty()) {
        std::cout << "Reading configuration from file: " << config_file_path << std::endl;
        try {
            MLCouplingData<float> input_data; // Populate with actual data as needed
            MLCouplingData<float> output_data; // Populate with actual data as needed
            MLCoupling<float, float>* ml_coupling = create_mlcoupling_from_config_file<float, float>(config_file_path, input_data, output_data);
            std::cout << "MLCoupling instance created from config file." << std::endl;
            // Further tests with ml_coupling can be added here

            if (max_step >= 0) {
                if (!ml_coupling) {
                    std::cerr << "Error: ml_coupling is null" << std::endl;
                } else {

                    if (!ml_coupling->behavior) {
                        std::cerr << "Error: ml_coupling->behavior is null" << std::endl;
                    } else {
                        MLCouplingBehavior* behavior = ml_coupling->behavior.get();
                        try {
                            std::cout << "Simulating Behavior << " << get_type_name(*behavior) << " >> for " << max_step << " steps:" << std::endl;
                        } catch (const std::exception &e) {
                            std::cerr << "get_type_name threw: " << e.what() << std::endl;
                        } catch (...) {
                            std::cerr << "get_type_name threw unknown exception" << std::endl;
                        }
                    }
                }

                std::cout << "------------------------------" << std::endl;
                for (int step = 0; step < max_step; ++step) {
                    int width = std::to_string(max_step - 1).length();
                    std::cout << "Step " << std::setw(width) << std::setfill(' ') << step << ": ";  if (ml_coupling->behavior->should_send_data()) {
                        std::cout << "coupling  ";
                    } else {
                        std::cout << "          ";
                    }
                    if (ml_coupling->behavior->should_perform_inference()) {
                        std::cout << "inference";
                    } else {
                        std::cout << "         ";
                    }
                    std::cout << std::endl;
                }
                std::cout << "------------------------------" << std::endl;
            } else {
                std::cout << "No max step specified for behavior testing. Skipping behavior simulation." << std::endl;
            }
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