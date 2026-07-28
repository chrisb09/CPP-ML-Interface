#pragma once

#include "../data/ml_coupling_data.hpp"

#include "ml_coupling_application.hpp"

// @registry_name: TurbulenceClosure
// @registry_aliases: turbulence-closure, turbulence_closure, turbulence
template <typename CouplingInput,
          typename CouplingOutput,
          typename LibraryInput = CouplingInput,
          typename LibraryOutput = CouplingOutput>
class MLCouplingApplicationTurbulenceClosure
    : public MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>
{
public:
    MLCouplingApplicationTurbulenceClosure(
        MLCouplingData<CouplingInput> coupling_input,
        MLCouplingData<CouplingOutput> coupling_output,
        MLCouplingNormalization<LibraryInput, CouplingOutput> *normalization)
        : MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(
              std::move(coupling_input), std::move(coupling_output), normalization)
    {
    }

    MLCouplingApplicationTurbulenceClosure(
        MLCouplingData<CouplingInput> coupling_input,
        MLCouplingData<LibraryInput> library_input,
        MLCouplingData<LibraryOutput> library_output,
        MLCouplingData<CouplingOutput> coupling_output,
        MLCouplingNormalization<LibraryInput, CouplingOutput> *normalization)
        : MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>(
              std::move(coupling_input),
              std::move(library_input),
              std::move(library_output),
              std::move(coupling_output),
              normalization)
    {
    }

protected:
    // Pre- and post-processing are already called in MLCoupling's ml_step()

    MLCouplingData<LibraryInput>
    preprocess_coupling_input(MLCouplingData<CouplingInput> coupling_input) override
    {
        // TODO: Implement turbulence closure specific preprocessing here
        this->library_input = MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>::
            preprocess_coupling_input(std::move(coupling_input));
        this->normalization->normalize_input(this->library_input);
        uniform_filtering();
        downsampling();
        return this->library_input;
    }

    MLCouplingData<CouplingOutput>
    postprocess_library_output(MLCouplingData<LibraryOutput> library_output) override
    {
        // TODO: Implement turbulence closure specific postprocessing here
        this->coupling_output = MLCouplingApplication<CouplingInput, CouplingOutput, LibraryInput, LibraryOutput>::
            postprocess_library_output(std::move(library_output));
        this->normalization->denormalize_output(this->coupling_output);
        compute_tau_ij();
        return this->coupling_output;
    }

private:
    void uniform_filtering()
    {
        // Placeholder for uniform filtering implementation
    }

    void downsampling()
    {
        // Placeholder for downsampling implementation
    }

    void compute_tau_ij()
    {
        // Placeholder for computing subgrid-scale stress tensor tau_ij
    }
};
