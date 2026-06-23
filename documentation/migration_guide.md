# Migration Guide: Upgrading to the Redesigned MLCoupling API

The ML Coupling Interface has been redesigned to merge flexible communication
directly into the base provider class, removing boilerplate and introducing
type-safe, chained proxy views for advanced workflows.

This guide details what needs to change depending on how you interact with
the library.

## 1. Solvers Using the Static API

**Impact: None.**

If your solver previously used the static API to run standard 1-to-1
inference, your code will continue to compile and run without modifications.

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

The `MLCouplingApplication` contract has not changed. You still implement
`prepare_input()` and `finalize_output()` to handle data manipulation and
normalization.

## 3. Implementing a Custom Provider

**Impact: High.**

If you have implemented a custom `MLCouplingProvider` (e.g. a mock, or an
extension for a new backend), you must update the base class and method
signatures.

### a) Base Class

All providers inherit directly from `MLCouplingProvider<In, Out>`. There is
no longer a separate `MLCouplingProviderFlexible` intermediate class — the
flexible methods live in the base class with default fallback implementations.

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

The core virtual method you must override has been renamed to explicitly
denote its static nature.

*Before:*
```cpp
void inference(MLCouplingData<In>* input, MLCouplingData<Out>* output) override { ... }
```
*After:*
```cpp
void static_inference(MLCouplingData<In>* input, MLCouplingData<Out>* output) override { ... }
```

### c) Adding Training Support (Optional)

If your provider supports training, override `static_train`.

```cpp
std::map<std::string, double> static_train(MLCouplingData<In>* input,
                                            MLCouplingData<Out>* target) override {
    // ... run training and return metrics
}
```

### d) Implementing Flexible Methods (Optional)

If your provider supports multiple sends/receives or explicit string
addressing, override the `flex_ordered_*` and `flex_keyed_*` virtuals. They
are part of the base class. If you do **not** override them, the base class
will buffer the staged data and fall back to calling `static_inference` /
`static_train` automatically. This is the **default, recommended** path
for any provider that is conceptually static.

The full set of virtuals exposed by the base class:

| Method | Purpose |
| :--- | :--- |
| `flex_ordered_set(MLCouplingData<In>)` | Stage one input, ordered |
| `flex_ordered_set_target(MLCouplingData<Out>)` | Stage one target, ordered |
| `flex_ordered_inference(MLCouplingData<Out>*)` | Run an ordered inference (populates the output) |
| `flex_ordered_train(long long)` | Run an ordered training step |
| `flex_keyed_set(const std::string&, MLCouplingData<In>)` | Stage one input, keyed |
| `flex_keyed_set_target(const std::string&, MLCouplingData<Out>)` | Stage one target, keyed |
| `flex_keyed_inference(keys_in, keys_out, fallback_out)` | Run a keyed inference |
| `flex_keyed_train(step_id, keys_in, keys_target)` | Run a keyed training step |

If a provider implements only the static methods, the base class transparently
collects the multiple staged inputs into a single `MLCouplingData` collection
(merge-by-concatenation) and forwards it to `static_inference`. No code change
is required for that to work.

## 4. Solvers Upgrading to the Flexible API

**Impact: Medium.**

If you previously tried to cast your provider to a flexible subclass to use
`send_data` / `receive_data`, use the proxy views built directly into
`MLCoupling` instead.

*Before (Pseudo-code):*
```cpp
auto flex_prov = dynamic_cast<MLCouplingProviderFlexible*>(coupling->provider.get());
flex_prov->send_data({"key1"}, data);
```

*After:*
```cpp
// Use the .keyed() view built into the coupling object
coupling->keyed()
    .set("key1", data)
    .inference({"key1"}, {"out_key"})
    .get("out_key", result);
```

See `api_guide.md` for full examples of the `.ordered()` and `.keyed()`
workflows.

## 5. C ABI Consumers (Optional)

**Impact: Lower compile time, smaller binary.**

The C API (`c_api.cpp`) was previously parameterized on the full
`MLCouplingDataType` enum (8 types), forcing the compiler to instantiate
input × output combinations for all 64 type pairs. It now exposes a
narrower `MLCouplingCAPIDataType` enum restricted to {float, double}, and
the C API uses a dedicated `MLCouplingCAPISupportedTypes` variant that the
`create_provider()` entry point visits — reducing the type matrix to 4
combinations.

Downstream C consumers that previously called `create_provider` with one of
the removed type tags must switch to the closest {float, double} value.
The C++ internal `MLCouplingDataType` enum is unchanged, so header-only
consumers are unaffected.
