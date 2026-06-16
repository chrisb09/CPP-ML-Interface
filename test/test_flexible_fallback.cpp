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
    
    if (dims.size() < 2) {
        std::cout << "Tensor is 1D: [";
        for (size_t i = 0; i < tensor.numel(); ++i) {
            std::cout << tensor.at_linear(i) << (i + 1 < tensor.numel() ? ", " : "");
        }
        std::cout << "]\n";
        return;
    }

    // We assume the first dimension is rows and the second dimension is columns
    // If it's a 3D tensor (like [B, m, D]), we'll flatten the first two dimensions for display purposes
    int rows = dims[0];
    int cols = dims[1];
    int stride = 1;
    
    if (dims.size() == 3) {
        rows = dims[0] * dims[1];
        cols = dims[2];
    } else if (dims.size() > 3) {
        std::cout << "Cannot print tensors with > 3 dimensions nicely.\n";
        return;
    }

    std::cout << "Shape: [";
    for(size_t i=0; i<dims.size(); ++i) {
        std::cout << dims[i] << (i+1 < dims.size() ? ", " : "");
    }
    std::cout << "]\n";

    for (int r = 0; r < rows; ++r) {
        if (dims.size() == 3 && r % dims[1] == 0) {
            std::cout << "  Batch " << (r / dims[1]) << ":\n";
        }
        std::cout << "    [";
        for (int c = 0; c < cols; ++c) {
            size_t linear_idx = r * cols + c;
            std::cout << std::setw(5) << tensor.at_linear(linear_idx);
            if (c + 1 < cols) std::cout << ", ";
        }
        std::cout << "]\n";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "Testing Flexible Fallback Merging Logic...\n\n";

    // 1. Create Data
    // Let's create two tensors of shape [2, 3] (Batch=2, Features=3)
    std::vector<float> data1_flat = {1, 2, 3, 4, 5, 6};
    std::vector<float> data2_flat = {11, 12, 13, 14, 15, 16};
    
    MLCouplingTensor<float> tensor1 = MLCouplingTensor<float>::from_flat_copy(data1_flat, {2, 3});
    MLCouplingTensor<float> tensor2 = MLCouplingTensor<float>::from_flat_copy(data2_flat, {2, 3});
    
    print_tensor_2d(tensor1, "Original Tensor 1");
    print_tensor_2d(tensor2, "Original Tensor 2");

    // 2. Setup Provider
    MLCouplingProviderTest<float, float> provider;
    
    // We need an application and coupling, but we can just use the base application
    MLCouplingData<float> dummy_in;
    MLCouplingData<float> dummy_out;
    
    // To use proxy, we just manually call provider methods since we are testing provider logic
    // We will test both List and Stack strategy
    
    std::cout << "==========================================\n";
    std::cout << "        TESTING LIST MERGE STRATEGY       \n";
    std::cout << "==========================================\n";
    
    provider.set_merge_strategy(MLCouplingMergeStrategy::List);
    
    MLCouplingData<float> cdata1; cdata1.add_tensor(tensor1);
    MLCouplingData<float> cdata2; cdata2.add_tensor(tensor2);
    
    provider.flex_ordered_set(cdata1);
    provider.flex_ordered_set(cdata2);
    
    MLCouplingData<float> fallback_out; // dummy
    provider.flex_ordered_inference(&fallback_out);
    
    std::cout << "Merged List Result contains " << provider.received_input.size() << " tensors.\n";
    assert(provider.received_input.size() == 2);
    std::cout << "SUCCESS: List Strategy concatenated the Data objects correctly.\n\n";
    
    
    std::cout << "==========================================\n";
    std::cout << "        TESTING STACK MERGE STRATEGY      \n";
    std::cout << "==========================================\n";
    
    provider.set_merge_strategy(MLCouplingMergeStrategy::Stack);
    
    provider.flex_ordered_set(cdata1);
    provider.flex_ordered_set(cdata2);
    
    provider.flex_ordered_inference(&fallback_out);
    
    std::cout << "Merged Stack Result contains " << provider.received_input.size() << " tensors.\n";
    assert(provider.received_input.size() == 1); // Should be 1 interleaved tensor!
    
    MLCouplingTensor<float>& stacked_tensor = provider.received_input[0];
    
    print_tensor_2d(stacked_tensor, "Stacked Tensor [B, m, D]");
    
    auto dims = stacked_tensor.dimensions();
    assert(dims.size() == 3);
    assert(dims[0] == 2); // B = 2
    assert(dims[1] == 2); // m = 2
    assert(dims[2] == 3); // D = 3
    
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
