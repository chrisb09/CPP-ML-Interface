# Migration Guide: Upgrading to the Redesigned MLCoupling API

The ML Coupling Interface has been redesigned to merge flexible communication directly into the base provider class, removing boilerplate and introducing type-safe, chained proxy views for advanced workflows. 

This guide details what needs to be changed depending on how you interact with the library.

## 1. Solvers Using the Static API
**Impact: None.**

If your solver previously used the static API to run standard 1-to-1 inference, your code will continue to compile and run without modifications.

*Before:*
```cpp
coupling->ml_step();
```
*After:*
```cpp
coupling->step(); // The preferred, more idiomatic alias, but ml_step() still works.
```

## 2. Implementing a Custom Application Class
**Impact: None.**

The `MLCouplingApplication` contract has not changed. You still implement `prepare_input()` and `finalize_output()` to handle data manipulation and normalization.

## 3. Implementing a Custom Provider
**Impact: High.**

If you have implemented a custom `MLCouplingProvider` (e.g., a mockup, or an extension for a new backend), you must update your base class and method signatures.

### a) Base Class changes
You no longer need to inherit from `MLCouplingProviderFlexible`. All providers now inherit directly from `MLCouplingProvider<In, Out>`.

*Before:*
```cpp
#include "ml_coupling_provider_flexible.hpp"
class MyProvider : public MLCouplingProviderFlexible<In, Out> { ... };
```
*After:*
```cpp
#include "ml_coupling_provider.hpp"
class MyProvider : public MLCouplingProvider<In, Out> { ... };
```

### b) Inference Method Renaming
The core virtual method you must override has been renamed to explicitly denote its static nature.

*Before:*
```cpp
void inference(MLCouplingData<In>* input, MLCouplingData<Out>* output) override { ... }
```
*After:*
```cpp
void static_inference(MLCouplingData<In>* input, MLCouplingData<Out>* output) override { ... }
```

### c) Adding Training Support (Optional)
If your provider supports training, you should now override `static_train`.
```cpp
std::map<std::string, double> static_train(MLCouplingData<In>* input, MLCouplingData<Out>* target) override {
    // ... run training and return metrics
}
```

### d) Implementing Flexible Methods (Optional)
If your provider supports multiple sends/receives or explicit string addressing, you no longer override `send_data` and `receive_data`. Instead, override the `flex_ordered_*` or `flex_keyed_*` methods.
See `include/provider/ml_coupling_provider.hpp` for the signatures. Note: If you do not override these, the base class will automatically buffer the data and fall back to calling your `static_inference`/`static_train` methods!

## 4. Solvers Upgrading to the Flexible API
**Impact: Medium.**

If you previously tried to cast your provider to `MLCouplingProviderFlexible` to use `send_data`, you should now use the proxy views built directly into `MLCoupling`.

*Before (Pseudo-code):*
```cpp
auto flex_prov = dynamic_cast<MLCouplingProviderFlexible*>(coupling->provider.get());
flex_prov->send_data({"key1"}, data);
```

*After:*
```cpp
// Simply use the .keyed() view built into the coupling object
coupling->keyed()
    .set("key1", data)
    .inference({"key1"}, {"out_key"})
    .get("out_key", result);
```

Please refer to the `api_guide.md` for full examples of using the `.ordered()` and `.keyed()` workflows.