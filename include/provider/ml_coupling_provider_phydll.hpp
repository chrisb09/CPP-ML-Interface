#pragma once

#include "ml_coupling_provider.hpp"

// @registry_name: Phydll
// @registry_aliases: phydll, PhyDLL
template <typename In, typename Out>
class MLCouplingProviderPhydll : public MLCouplingProvider<In, Out> {

    public:
        void init() override {
            // TODO:
        }

        void inference() override {
            // TODO
        }

        void finalize() override {
            // TODO:
        }

    private:

};