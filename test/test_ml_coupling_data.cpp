#include <cassert>
#include <iostream>
#include <vector>

#include "data/ml_coupling_data.hpp"

void test_flatten_single_nested_tensor() {
    int row0[] = {10, 20, 30};
    int row1[] = {40, 50, 60};

    void* top[] = {static_cast<void*>(row0), static_cast<void*>(row1)};
    MLCouplingTensor<int> nested = MLCouplingTensor<int>::wrap_nested(
        static_cast<void*>(top),
        std::vector<int>{2, 3},
        MLCouplingMemLayoutNested,
        MLCouplingOwnershipExternal);

    MLCouplingTensor<int> flat = nested.flatten();

    const int expected[] = {10, 20, 30, 40, 50, 60};
    for (int i = 0; i < 6; ++i) {
        assert(flat.at_linear(i) == expected[i]);
    }
}

void test_get_flat_data_for_multiple_tensors() {
    int flat_tensor[] = {1, 2};

    int row0[] = {3, 4, 5};
    int row1[] = {6, 7, 8};
    void* top[] = {static_cast<void*>(row0), static_cast<void*>(row1)};

    MLCouplingData<int> data;
    data.add_tensor(MLCouplingTensor<int>::wrap_flat(flat_tensor, std::vector<int>{2}));
    data.add_tensor(MLCouplingTensor<int>::wrap_nested(
        static_cast<void*>(top),
        std::vector<int>{2, 3},
        MLCouplingMemLayoutNested,
        MLCouplingOwnershipExternal));

    std::vector<int> flat = data.get_flat_data();
    const int expected[] = {1, 2, 3, 4, 5, 6, 7, 8};
    for (int i = 0; i < 8; ++i) {
        assert(flat[i] == expected[i]);
    }
}

void test_deep_copy_copies_owned_tensor() {
    MLCouplingTensor<int> original = MLCouplingTensor<int>::from_flat_copy(
        std::vector<int>{11, 12, 13},
        std::vector<int>{3});

    MLCouplingTensor<int> copied = original.deep_copy();

    original.set_linear(0, 99);

    assert(copied.at_linear(0) == 11);
    assert(copied.at_linear(1) == 12);
    assert(copied.at_linear(2) == 13);
}

void test_element_iterator() {
    MLCouplingTensor<int> tensor = MLCouplingTensor<int>::from_flat_copy(
        std::vector<int>{1, 2, 3, 4},
        std::vector<int>{2, 2});

    int sum = 0;
    for (auto it = tensor.begin_elements(); it != tensor.end_elements(); ++it) {
        sum += *it;
    }
    assert(sum == 10);
}

int main() {
    test_flatten_single_nested_tensor();
    test_get_flat_data_for_multiple_tensors();
    test_deep_copy_copies_owned_tensor();
    test_element_iterator();

    std::cout << "All ml_coupling_data tests passed.\n";
    return 0;
}
