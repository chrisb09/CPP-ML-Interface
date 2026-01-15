
#include <iostream>

#include "include/behavior/ml_coupling_behavior_periodic.hpp"


// This is a test to run and see how the periodic behavior performs, for the first 100 simulated steps we want to output
// what the behavior assigns

int main() {
    MLCouplingBehaviorPeriodic behavior(10, 5, 1, 24);

    for (int step = 1; step <= 100; ++step) {
        bool perform_inference = behavior.should_perform_inference();
        int time_step_delta = behavior.time_step_delta();
        bool send_data = behavior.should_send_data();

        std::cout << "Step " << step << ": "
                  << (perform_inference ? "Inference" : (send_data ? "Send Data" : "Normal")) << std::endl;
        if (perform_inference) {
            step += time_step_delta;
        }
    }

    return 0;
}
