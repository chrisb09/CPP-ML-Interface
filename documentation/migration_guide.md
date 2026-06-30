# Migration Guide: Upgrading to the Redesigned MLCoupling API

The ML Coupling Interface has been redesigned to merge flexible communication
directly into the base provider class, removing boilerplate and introducing
type-safe, chained proxy views for advanced workflows.

This guide details what needs to change depending on how you interact with
the library.

## 1. Solvers Using the Static API

**Impact: Medium.**

### a) `step()` now returns `int`

`MLCoupling::step()` now returns an `int` step delta: `0` when no inference
occurred (solver should perform a normal time step), or `N` when inference
was performed and the solver should advance time by `N` steps without
running its usual RHS computation.

*Before:*
```cpp
coupling->step();
// always ran rhs()/rungeKutta() after
```
*After:*
```cpp
int delta = coupling->step();
if (delta == 0) {
    rhs(); rhsBnd(); rungeKuttaStep(); setTimeStep(); lhsBnd();
} else {
    m_physicalTime += delta * m_timeStep * m_timeRef;
    m_time += delta * m_timeStep;
    globalTimeStep += delta;
}
```

### b) `ml_step()` alias removed

The backward-compat alias `void ml_step() { step(); }` has been removed.
Replace all calls to `coupling->ml_step()` with `coupling->step()` and
handle the new `int` return value.

## 2. Implementing a Custom Application Class

**Impact: Medium.**

### a) New `ml_step(provider&, behavior&)` virtual

The orchestration has moved from per-override virtuals to a single
`ml_step(MLCouplingProvider<In,Out>&, MLCouplingBehavior&)` virtual.
Override this to drive the provider/behavior interaction:

```cpp
int ml_step(MLCouplingProvider<In,Out>& provider, MLCouplingBehavior& behavior) override
{
    if (behavior.should_send_data())
    {
        prepare_input();
    }
    if (behavior.should_perform_inference())
    {
        provider.static_inference(&this->input_data_after_preprocessing,
                                   &this->output_data_before_postprocessing);
        finalize_output();
        return behavior.time_step_delta();
    }
    return 0;
}
```

### b) Old `ml_step(MLCouplingData<In>)` removed

The pure virtual `MLCouplingData<Out> ml_step(MLCouplingData<In>)` has been
removed. If your application subclass overrode it, replace with the new
`ml_step(provider&, behavior&)` signature above.

### c) `coupling_step(MLCouplingData<In>)` now non-pure

`coupling_step` is no longer a pure virtual — it has an empty default body.
If your subclass overrode it with a no-op, you may remove the override.

### d) `step(bool, bool)` removed

The old `MLCouplingApplication::step(bool perform_coupling, bool perform_inference)`
method has been removed. Data flow is now driven by the behavior's
`should_send_data()` / `should_perform_inference()` queries, not by boolean
flags.

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

## 6. Application-Led Orchestration (New)

**Impact: New feature, no migration needed.**

A new `ml_step(MLCouplingProvider<In,Out>&, MLCouplingBehavior&)` virtual
on `MLCouplingApplication` drives provider/behavior orchestration. The
`MLCoupling::step()` method delegates to it.

### a) New Behavior: `MLCouplingBehaviorFlowExtrapolator`

A new behavior class `MLCouplingBehaviorFlowExtrapolator` (registry name:
`FlowExtrapolatorBehavior`, aliases: `flow-extrapolator-behavior`,
`maia-flow-extrapolator-behavior`) is available in
`include/behavior/ml_coupling_behavior_flow_extrapolator.hpp`. It subclasses
`MLCouplingBehavior` directly and manages MAIA-style coupling/inference
timing with HDF-output avoidance.

Constructor parameters (all TOML-configurable):

| Parameter | TOML key | Description |
| :--- | :--- | :--- |
| `inference_interval` | `behavior.inference_interval` | Steps between inference cycles |
| `coupled_steps_before_inference` | `behavior.coupled_steps_before_inference` | Number of coupled steps before each inference |
| `step_increment_after_inference` | `behavior.step_increment_after_inference` | Time-step increment after inference |
| `hdf_output_interval` | `behavior.hdf_output_interval` | HDF output interval for avoidance shifting |
| `total_timesteps` | `behavior.total_timesteps` | Total simulation steps |
| `scaling_factor` | `behavior.scaling_factor` (default 1.0) | Step-index scaling factor |
| `forecast_window` | `behavior.forecast_window` (default 1) | Forecast window size |
| `input_step_distance` | `behavior.input_step_distance` (default 1) | Stride between coupled steps |
| `inference_start_step` | `behavior.inference_start_step` (default 0) | First inference logical step |
| `global_step_offset` | `behavior.global_step_offset` (default 0) | Offset applied to logical→global step mapping |

### b) Known-Affected Consumers

The following consumers are affected by the removal of the old
`ml_step(MLCouplingData<In>)` virtual and `step()` return-type change.
They are listed here for awareness; fixing them is a follow-up task outside
this transition:

- `mini_app/solver_cpp`
- `module_test/solver.cpp`
