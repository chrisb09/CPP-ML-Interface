# ML Coupling Interface: API Guide

This guide details how to interact with the redesigned C++ `MLCoupling` interface. The interface supports simple static operations as well as flexible (ordered and keyed) multi-step workflows.

## 1. Initialization

Creating a coupling instance is identical across all interaction types. It usually involves passing configuration and buffers (if not using flexible coupling exclusively).

```cpp
auto coupling = MLCoupling<float, double>::create_from_config(
    "config.toml", 
    input_buffer, 
    output_buffer
);
```

## 2. Static Execution (The Shortcuts)

If your workflow consists of exactly one input and one output tensor (or dataset) per step, you should use the high-level static API.

### Static Inference
Calls the provider's inference method, automatically applying `prepare_input()` and `finalize_output()` from your Application class.
```cpp
coupling->step();
// Note: coupling->ml_step() is also supported for backward compatibility.
```

### Static Training
Runs the provider's training step on the current data and records tracked metadata.
```cpp
long long current_iteration = 100;
coupling->train_step(current_iteration);
```

## 3. Flexible Execution (Proxy Views)

When your provider supports it (like SmartSim or PhyDLL), you can send and receive multiple pieces of data. You do this using "Proxy Views", which allow you to chain operations.

> **Why `set_target` instead of just `set` for labels?**
> In C++, the input (`In`) and output (`Out`) types of the ML coupling can differ (e.g., float inputs, double outputs). Since training targets (labels) represent the model's output, they use the `Out` type. Having separate `set()` and `set_target()` methods ensures type safety and prevents compiler ambiguities.

### Ordered View (`.ordered()`)
Use this when you want to stage data sequentially without explicitly naming the buffers. The order of staging must match what the provider/model expects.

**Inference:**
```cpp
coupling->ordered()
    .set(feature_part_1)
    .set(feature_part_2)
    .inference()
    .get(output_part_1)
    .get(output_part_2);
```

**Training:**
```cpp
auto metrics = coupling->ordered()
    .set(features)
    .set_target(labels)
    .train(current_iteration);
```

### Keyed View (`.keyed()`)
Use this when you want to explicitly name your inputs and outputs, allowing the provider to map them correctly regardless of the order they are staged.

**Inference:**
```cpp
coupling->keyed()
    .set("encoder_in", encoder_features)
    .set("decoder_in", decoder_features)
    .inference({"encoder_in", "decoder_in"}, {"final_out"})
    .get("final_out", predictions);
```

**Training:**
```cpp
auto metrics = coupling->keyed()
    .set("features", my_features)
    .set_target("labels", ground_truth)
    .train(current_iteration, {"features"}, {"labels"});
```

## 4. Flexible API Fallback (Merge Logic)

The `MLCoupling` interface provides robust backward compatibility. You can use the flexible `.ordered()` and `.keyed()` proxy views **even if the backend provider is strictly static** (i.e., it only implements `static_inference`).

If a static provider encounters multiple `.set()` calls (e.g., staging multiple features sequentially or under different keys), the `MLCouplingProvider` base class automatically executes a **Merge-by-Concatenation fallback**:

1. **Staging:** All staged data objects are held in memory buffers.
2. **Merging:** When `.inference()` or `.train()` is invoked, the fallback concatenates the internal lists of tensors from all staged `MLCouplingData` objects into a single, unified `MLCouplingData` object. It maintains the canonical order defined by the sequential `.set()` calls or the list of keys provided to `.inference({"key1", "key2"}, ...)`.
3. **Execution:** This single, merged `MLCouplingData` object is then passed to the provider's `static_inference` or `static_train` method.

This allows you to write clean, multi-step staging logic in your solvers without having to worry if the underlying provider supports native multi-step data transfer.

## 5. Training Tracking

The `MLCoupling` interface includes a built-in `TrainingTracker` to monitor metadata returned during training steps (like loss or accuracy).

### Enabling Tracking
Before your training loop, register the fields you want to monitor:
```cpp
coupling->track("loss");
coupling->track("learning_rate");
```

### Retrieving Metrics
You can query the history or the current state of the tracked metrics at any time:

```cpp
// Get the latest value for a specific field
double current_loss = coupling->get_current("loss");

// Get the latest values for all tracked fields
std::map<std::string, double> all_current = coupling->get_current();

// Get the full history across all iterations for a specific field
std::vector<double> loss_history = coupling->get_history("loss");

// Get the full history for all tracked fields
std::map<std::string, std::vector<double>> all_history = coupling->get_history();
```
