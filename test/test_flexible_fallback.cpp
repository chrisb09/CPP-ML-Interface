#include <iostream>
#include <vector>
#include <iomanip>
#include <cassert>

// Include ml_coupling_data to use tensors
#include "data/ml_coupling_data.hpp"
#include "provider/ml_coupling_provider.hpp"
#include "ml_coupling.hpp"

// We need a dummy provider to test the fallback logic
template <typename In, typename Out>
class MLCouplingProviderTest : public MLCouplingProvider<In, Out> {
public:
    MLCouplingData<In> received_input;

    void static_inference(MLCouplingData<In>* input, MLCouplingData<Out>* output) override {
        // Just store what we got so we can inspect it
        received_input = input->deep_copy();
        
        // Populate output with something dummy
        std::vector<Out> flat_out(1, static_cast<Out>(42));
        MLCouplingTensor<Out> out_tensor = MLCouplingTensor<Out>::from_flat_copy(flat_out, {1});
        output->add_tensor(out_tensor);
    }
};

void print_tensor_2d(const MLCouplingTensor<float>& tensor, const std::string& title) {
    std::cout << "--- " << title << " ---\n";
    auto dims = tensor.dimensions();
    
    std::cout << "Shape: [";
    for(size_t i=0; i<dims.size(); ++i) {
        std::cout << dims[i] << (i+1 < dims.size() ? ", " : "");
    }
    std::cout << "]\n";

    if (dims.size() == 1) {
        std::cout << "    [";
        for (size_t i = 0; i < tensor.numel(); ++i) {
            std::cout << std::setw(5) << tensor.at_linear(i) << (i + 1 < tensor.numel() ? ", " : "");
        }
        std::cout << "]\n\n";
        return;
    }

    if (dims.size() > 4) {
        std::cout << "Cannot print tensors with > 4 dimensions nicely.\n\n";
        return;
    }

    int B = dims.size() >= 3 ? dims[0] : 1;
    int M = dims.size() == 4 ? dims[1] : 1;
    int rows = dims[dims.size() - 2];
    int cols = dims[dims.size() - 1];

    if (dims.size() == 2) {
        B = 1; M = 1; rows = dims[0]; cols = dims[1];
    } else if (dims.size() == 3) {
        B = dims[0]; M = 1; rows = dims[1]; cols = dims[2];
    }

    for (int b = 0; b < B; ++b) {
        if (B > 1 || dims.size() >= 3) std::cout << "  Batch " << b << ":\n";
        for (int m = 0; m < M; ++m) {
            if (M > 1) std::cout << "    Step " << m << ":\n";
            for (int r = 0; r < rows; ++r) {
                std::cout << (M > 1 ? "      [" : "    [");
                for (int c = 0; c < cols; ++c) {
                    size_t linear_idx = b * (M * rows * cols) + m * (rows * cols) + r * cols + c;
                    std::cout << std::setw(5) << tensor.at_linear(linear_idx);
                    if (c + 1 < cols) std::cout << ", ";
                }
                std::cout << "]\n";
            }
        }
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    (void)argc; (void)argv;
    std::cout << "Testing Flexible Fallback Merging Logic...\n\n";

    // 1. Create Data
    // For List Strategy: tensors of IDENTICAL batch sizes but varying non-batch dimensions
    std::vector<float> data_list_1 = {1, 2, 3,   4, 5, 6};                            // B=2, D1=3
    std::vector<float> data_list_2 = {7, 8, 9, 10,   11, 12, 13, 14};                 // B=2, D1=2, D2=2 (slice=4)
    std::vector<float> data_list_3 = {15, 16,   17, 18};                              // B=2, D1=2
    
    MLCouplingTensor<float> tensor_list_1 = MLCouplingTensor<float>::from_flat_copy(data_list_1, {2, 3});
    MLCouplingTensor<float> tensor_list_2 = MLCouplingTensor<float>::from_flat_copy(data_list_2, {2, 2, 2});
    MLCouplingTensor<float> tensor_list_3 = MLCouplingTensor<float>::from_flat_copy(data_list_3, {2, 2});
    
    // For Stack Strategy: tensors of exact same shape, with batch size > 1
    std::vector<float> data_stack_1 = {1, 2, 3, 4, 5, 6}; // 2x3
    std::vector<float> data_stack_2 = {11, 12, 13, 14, 15, 16}; // 2x3
    MLCouplingTensor<float> tensor_stack_1 = MLCouplingTensor<float>::from_flat_copy(data_stack_1, {2, 3});
    MLCouplingTensor<float> tensor_stack_2 = MLCouplingTensor<float>::from_flat_copy(data_stack_2, {2, 3});
    
    std::cout << "==========================================\n";
    std::cout << "        TESTING LIST MERGE STRATEGY       \n";
    std::cout << "==========================================\n";
    print_tensor_2d(tensor_list_1, "Original List Tensor 1 [2, 3]");
    print_tensor_2d(tensor_list_2, "Original List Tensor 2 [2, 2, 2]");
    print_tensor_2d(tensor_list_3, "Original List Tensor 3 [2, 2]");
    
    MLCouplingProviderTest<float, float> provider;
    MLCouplingData<float> fallback_out; // dummy
    
    provider.set_merge_strategy(MLCouplingMergeStrategy::List);
    
    MLCouplingData<float> cdata_list_1; cdata_list_1.add_tensor(tensor_list_1);
    MLCouplingData<float> cdata_list_2; cdata_list_2.add_tensor(tensor_list_2);
    MLCouplingData<float> cdata_list_3; cdata_list_3.add_tensor(tensor_list_3);
    
    provider.flex_ordered_set(cdata_list_1);
    provider.flex_ordered_set(cdata_list_2);
    provider.flex_ordered_set(cdata_list_3);
    
    provider.flex_ordered_inference(&fallback_out);
    
    std::cout << "Merged List Result contains " << provider.received_input.size() << " tensors.\n";
    assert(provider.received_input.size() == 1);
    print_tensor_2d(provider.received_input[0], "List Output Tensor [2, 9]");
    
    assert(provider.received_input[0].numel() == 18);
    auto list_dims = provider.received_input[0].dimensions();
    assert(list_dims.size() == 2);
    assert(list_dims[0] == 2);
    assert(list_dims[1] == 9); // 3 + 4 + 2
    
    // Batch 0
    assert(provider.received_input[0].at_linear(0) == 1);
    assert(provider.received_input[0].at_linear(2) == 3);
    assert(provider.received_input[0].at_linear(3) == 7);
    assert(provider.received_input[0].at_linear(6) == 10);
    assert(provider.received_input[0].at_linear(7) == 15);
    // Batch 1
    assert(provider.received_input[0].at_linear(9) == 4);
    assert(provider.received_input[0].at_linear(12) == 11);
    assert(provider.received_input[0].at_linear(17) == 18);
    std::cout << "SUCCESS: List Strategy concatenated the Data objects correctly per batch item.\n\n";
    
    
    std::cout << "==========================================\n";
    std::cout << "        TESTING STACK MERGE STRATEGY      \n";
    std::cout << "==========================================\n";
    print_tensor_2d(tensor_stack_1, "Original Stack Tensor 1 [2, 3]");
    print_tensor_2d(tensor_stack_2, "Original Stack Tensor 2 [2, 3]");
    
    provider.set_merge_strategy(MLCouplingMergeStrategy::Stack);
    
    MLCouplingData<float> cdata_stack_1; cdata_stack_1.add_tensor(tensor_stack_1);
    MLCouplingData<float> cdata_stack_2; cdata_stack_2.add_tensor(tensor_stack_2);
    
    provider.flex_ordered_set(cdata_stack_1);
    provider.flex_ordered_set(cdata_stack_2);
    
    provider.flex_ordered_inference(&fallback_out);
    
    std::cout << "Merged Stack Result contains " << provider.received_input.size() << " tensors.\n";
    assert(provider.received_input.size() == 1); // Should be 1 interleaved tensor!
    
    MLCouplingTensor<float>& stacked_tensor = provider.received_input[0];
    
    print_tensor_2d(stacked_tensor, "Stacked Tensor [B, m, D1]");
    
    auto dims = stacked_tensor.dimensions();
    assert(dims.size() == 3);
    assert(dims[0] == 2); // B = 2
    assert(dims[1] == 2); // m = 2
    assert(dims[2] == 3); // D1 = 3
    
    // Verify values were interleaved correctly
    // Batch 0, Step 0
    assert(stacked_tensor.at({0, 0, 0}) == 1);
    assert(stacked_tensor.at({0, 0, 1}) == 2);
    assert(stacked_tensor.at({0, 0, 2}) == 3);
    // Batch 0, Step 1
    assert(stacked_tensor.at({0, 1, 0}) == 11);
    assert(stacked_tensor.at({0, 1, 1}) == 12);
    assert(stacked_tensor.at({0, 1, 2}) == 13);
    // Batch 1, Step 0
    assert(stacked_tensor.at({1, 0, 0}) == 4);
    assert(stacked_tensor.at({1, 0, 1}) == 5);
    assert(stacked_tensor.at({1, 0, 2}) == 6);
    // Batch 1, Step 1
    assert(stacked_tensor.at({1, 1, 0}) == 14);
    assert(stacked_tensor.at({1, 1, 1}) == 15);
    assert(stacked_tensor.at({1, 1, 2}) == 16);
    
    std::cout << "SUCCESS: Stack Strategy interleaved the tensors perfectly!\n";
    
    return 0;
}