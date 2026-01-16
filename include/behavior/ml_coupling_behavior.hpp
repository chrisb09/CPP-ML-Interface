#pragma once

// @category: behavior
class MLCouplingBehavior {
    public:

        virtual bool should_perform_inference() = 0;

        virtual int time_step_delta() = 0;
    
        // phydll supports sending data without performing inference
        // so we can send data multiple times and not run the model each time
        // this method indicates whether data should be sent in the current step
        // provided the application supports it
        // managing this is the responsibility of the behavior class itself
        // which has access to the application properties
        virtual bool should_send_data() = 0;

};