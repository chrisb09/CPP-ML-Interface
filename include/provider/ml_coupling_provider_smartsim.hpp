#pragma once

#include "ml_coupling_provider.hpp"
#include "../tool.h"
#include "../data/ml_coupling_data_type.hpp"
#include "../data/ml_coupling_memory_layout.hpp"
#include "../logging.hpp"
#include "../scorep_profiling_state.hpp"
#include "../smartsim_key_balancing.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <cstdlib> // for setenv
#include <climits>

#if defined(WITH_SMARTSIM)
#include "client.h"
#endif

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

// @registry_name: Smartsim
// @registry_aliases: smartsim, SmartSim
template <typename In, typename Out>
class MLCouplingProviderSmartsim : public MLCouplingProvider<In, Out>
{

// local variables
#if defined(WITH_SMARTSIM)
    SmartRedis::Client *client;
#endif

    // parameters

    const std::string device;
    const std::string model_backend;
    const std::string model_name;
    const int num_gpus;
    const int first_gpu;
    const int batch_size;
    const int min_batch_size;
    const int min_batch_timeout;
    const int command_timeout;
    const int socket_timeout;
    const int model_timeout;
    const std::vector<std::string> tf_input_labels;
    const std::vector<std::string> tf_output_labels;
    const std::vector<std::string> tf_input_keys; // Added new member
    mlcoupling::smartsim_key_balancing::RedisKeyBalancer key_balancer;

    static int env_int(const char *name, int fallback)
    {
        const char *raw = std::getenv(name);
        if (raw == nullptr || *raw == '\0')
        {
            return fallback;
        }
        char *end = nullptr;
        errno = 0;
        const long parsed = std::strtol(raw, &end, 10);
        if (end == raw || errno == ERANGE || parsed < INT_MIN || parsed > INT_MAX)
        {
            return fallback;
        }
        return static_cast<int>(parsed);
    }

    static int resolve_nodes(int configured_nodes)
    {
        if (configured_nodes >= 0)
        {
            return configured_nodes;
        }
        return env_int("MLCOUPLING_SMARTSIM_NODES", -1);
    }

    static int resolve_num_gpus(const std::string &device, int configured_num_gpus)
    {
        if (configured_num_gpus >= 0)
        {
            return configured_num_gpus;
        }
        if (device != "GPU")
        {
            return 0;
        }
        return env_int("MLCOUPLING_SMARTSIM_NUM_GPUS", -1);
    }

    static void set_timeout_env_if_not_set(const char *name, int value)
    {
        if (value >= 0 && std::getenv(name) == nullptr)
        {
            setenv(name, std::to_string(value).c_str(), 1);
        }
    }

private:
    MLCouplingProviderSmartsim(std::string device = std::string(),
                               std::string model_backend = std::string(),
                               std::string model_path = std::string(),
                               std::string_view model = std::string_view(),
                               std::string model_name = "model",
                               std::string host = "",
                               int port = -1,
                               std::vector<std::string> hosts = {},
                               std::vector<int> ports = {},
                               int nodes = -1,
                               int num_gpus = -1,
                               int first_gpu = 0,
                               int batch_size = 0,
                               int min_batch_size = 0,
                               int min_batch_timeout = 0,
                               int command_timeout = -1,
                               int socket_timeout = -1,
                               int model_timeout = -1,
                               const std::vector<std::string> &tf_input_labels = {},
                               const std::vector<std::string> &tf_output_labels = {},
                               const std::vector<std::string> &tf_input_keys = {}, // New parameter
                               MLCouplingData<In>* input_after_preprocessing = nullptr,
                               MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : device(std::move(device)),
          model_backend(std::move(model_backend)),
          model_name(std::move(model_name)),
          num_gpus(resolve_num_gpus(device, num_gpus)),
          first_gpu(first_gpu),
          batch_size(batch_size),
          min_batch_size(min_batch_size),
          min_batch_timeout(min_batch_timeout),
          command_timeout(command_timeout),
          socket_timeout(socket_timeout),
          model_timeout(model_timeout),
          tf_input_labels(std::move(tf_input_labels)),
          tf_output_labels(std::move(tf_output_labels)),
          tf_input_keys(std::move(tf_input_keys)), // Initialize new parameter
          input_after_preprocessing(input_after_preprocessing),
          output_before_postprocessing(output_before_postprocessing)
    {

#if !defined(WITH_SMARTSIM)
        guarantee(false, "SmartSim provider is not enabled. Please make sure WITH_SMARTSIM is defined and the necessary dependencies are installed.");
#endif

        const int resolved_nodes = resolve_nodes(nodes);

        if (resolved_nodes < 0)
        {
            logging::warning("nodes not configured; set nodes explicitly or MLCOUPLING_SMARTSIM_NODES.");
        }
        if (this->device == "GPU" && this->num_gpus < 0)
        {
            logging::warning("num_gpus not configured; set num_gpus explicitly or MLCOUPLING_SMARTSIM_NUM_GPUS.");
        }

        validate_parameter(this->device,
                           this->model_backend,
                           model_path,
                           std::string_view(), // model (not used in this provider)
                           this->model_name,
                           host,
                           port,
                           hosts,
                           ports,
                           resolved_nodes,
                           this->num_gpus,
                           this->first_gpu,
                           this->batch_size,
                           this->min_batch_size,
                           this->min_batch_timeout,
                           this->command_timeout,
                           this->socket_timeout,
                           this->model_timeout,
                           tf_input_labels,
                           tf_output_labels,
                           tf_input_keys);

        const int key_balance_streams = this->num_gpus > 0 ? this->num_gpus : 0;
        key_balancer = mlcoupling::smartsim_key_balancing::RedisKeyBalancer::from_environment(
            this->rank, key_balance_streams, resolved_nodes);
        if (key_balancer.enabled()) {
            if (this->rank == 0) {
                for (std::size_t shard = 0; shard < key_balancer.tags().size(); ++shard) {
                    const std::string& tag = key_balancer.tags()[shard];
                    logging::info("SMARTSIM_KEY_BALANCE shard=" + std::to_string(shard) +
                                  " tag=" + tag +
                                  " slot=" + std::to_string(
                                      mlcoupling::smartsim_key_balancing::redis_hash_slot(tag)) +
                                  " expected_slot_range=[" + std::to_string(
                                      mlcoupling::smartsim_key_balancing::RedisKeyBalancer::slot_first(
                                          static_cast<int>(shard), resolved_nodes)) +
                                  "," + std::to_string(
                                      mlcoupling::smartsim_key_balancing::RedisKeyBalancer::slot_last(
                                          static_cast<int>(shard), resolved_nodes)) + "]");
                }
            }
            logging::info("SMARTSIM_KEY_BALANCE_ASSIGNMENT rank=" + std::to_string(this->rank) +
                          " target_shard=" + std::to_string(key_balancer.target_shard()) +
                          " gpu=" + std::to_string(
                              this->rank % (key_balance_streams > 0 ? key_balance_streams : 1)));
        }

        // Before creating a smartsim client (SmartRedis to be exact), we need to set the appropriate env vars

        // Most importantly, SSDB has to be set if it isn't already, either via host and port, or hosts and ports
        std::string ssdb;
        if (host != "" && port != -1)
        {
            ssdb = host + ":" + std::to_string(port);
            setenv("SSDB", ssdb.c_str(), 1);
        }
        else if (!hosts.empty() && !ports.empty() && hosts.size() == ports.size())
        {
            ssdb = "";
            for (int i = 0; i < (int)hosts.size(); i++)
            {
                ssdb += hosts[i] + ":" + std::to_string(ports[i]);
                if (i < (int)hosts.size() - 1)
                {
                    ssdb += ",";
                }
            }
            setenv("SSDB", ssdb.c_str(), 1);
        }

        // Setting the database type based on the number of nodes
        setenv("SR_DB_TYPE", resolved_nodes > 1 ? "Clustered" : "Standalone", 1);

        // Setting the SmartRedis timeouts if provided and not already set in the environment
        set_timeout_env_if_not_set("SR_CMD_TIMEOUT", this->command_timeout);
        set_timeout_env_if_not_set("SR_SOCKET_TIMEOUT", this->socket_timeout);
        set_timeout_env_if_not_set("SR_MODEL_TIMEOUT", this->model_timeout);

#if defined(WITH_SMARTSIM)

        int world_rank = this->rank;

        client = new SmartRedis::Client("solver_" + std::to_string(world_rank));

        bool is_multi = (std::getenv("MLCOUPLING_MULTI_MODEL") != nullptr);
        if (this->rank == 0 || is_multi)
        {
            if (this->rank == 0)
            {
                logging::debug("SmartSim Coupling Provider initialized with the following parameters:");
                logging::debug("Device: " + this->device);
                logging::debug("Model Backend: " + this->model_backend);
                logging::debug("SmartSim DB Nodes: " + std::to_string(resolved_nodes));
                logging::debug("Number of GPUs: " + std::to_string(this->num_gpus));
                logging::debug("First GPU: " + std::to_string(this->first_gpu));
                logging::debug("Batch Size: " + std::to_string(this->batch_size));
                logging::debug("Min Batch Size: " + std::to_string(this->min_batch_size));
                logging::debug("Min Batch Timeout: " + std::to_string(this->min_batch_timeout) + " ms");
                if (this->command_timeout >= 0) logging::debug("Command Timeout: " + std::to_string(this->command_timeout) + " s");
                if (this->socket_timeout >= 0) logging::debug("Socket Timeout: " + std::to_string(this->socket_timeout) + " s");
                if (this->model_timeout >= 0) logging::debug("Model Timeout: " + std::to_string(this->model_timeout) + " ms");
                if (!ssdb.empty()) {
                    logging::debug("SSDB: " + ssdb);
                }
            }

            // Load the model into the database

            if (!model_path.empty())
            {
                // Load model from file path
                if (this->device == "GPU")
                {
                    client->set_model_from_file_multigpu(this->model_name, model_path, this->model_backend, this->first_gpu, this->num_gpus, this->batch_size, this->min_batch_size, this->min_batch_timeout, "", tf_input_labels, tf_output_labels);
                }
                else
                {
                    client->set_model_from_file(this->model_name, model_path, this->model_backend, "CPU", this->batch_size, this->min_batch_size, this->min_batch_timeout, "", tf_input_labels, tf_output_labels);
                }
            }
            else if (!model.empty())
            {
                // Load model from string
                if (this->device == "GPU")
                {
                    client->set_model_multigpu(this->model_name, model, this->model_backend, this->first_gpu, this->num_gpus, this->batch_size, this->min_batch_size, this->min_batch_timeout, "", tf_input_labels, tf_output_labels);
                }
                else
                {
                    client->set_model(this->model_name, model, this->model_backend, "CPU", this->batch_size, this->min_batch_size, this->min_batch_timeout, "", tf_input_labels, tf_output_labels);
                }
            }
        }

#ifdef MLCOUPLING_PROVIDER_HAS_MPI
        int mpi_initialized = 0;
        MPI_Initialized(&mpi_initialized);
        if (mpi_initialized)
        {
            MPI_Barrier(MPI_COMM_WORLD);
        }
#endif

#endif
    }

public:
    MLCouplingProviderSmartsim(std::string device,
                               std::string model_backend,
                               std::string model_path,
                               std::string model_name = "model",
                               std::string host = "",
                               int port = -1,
                               int nodes = -1,
                               int num_gpus = -1,
                               int first_gpu = 0,
                               int batch_size = 0,
                               int min_batch_size = 0,
                               int min_batch_timeout = 0,
                               int command_timeout = -1,
                               int socket_timeout = -1,
                               int model_timeout = -1,
                               const std::vector<std::string> &tf_input_labels = {},
                               const std::vector<std::string> &tf_output_labels = {},
                               const std::vector<std::string>& tf_input_keys = {},
                               MLCouplingData<In>* input_after_preprocessing = nullptr,
                               MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : MLCouplingProviderSmartsim(std::move(device), std::move(model_backend), std::move(model_path), std::string_view(), std::move(model_name), std::move(host), port, std::vector<std::string>(), std::vector<int>(), nodes, num_gpus, first_gpu, batch_size, min_batch_size, min_batch_timeout, command_timeout, socket_timeout, model_timeout, tf_input_labels, tf_output_labels, tf_input_keys, input_after_preprocessing, output_before_postprocessing) {};

    MLCouplingProviderSmartsim(std::string device,
                               std::string model_backend,
                               std::string_view model,
                               std::string model_name = "model",
                               std::string host = "",
                               int port = -1,
                               int nodes = -1,
                               int num_gpus = -1,
                               int first_gpu = 0,
                               int batch_size = 0,
                               int min_batch_size = 0,
                               int min_batch_timeout = 0,
                               int command_timeout = -1,
                               int socket_timeout = -1,
                               int model_timeout = -1,
                               const std::vector<std::string> &tf_input_labels = {},
                               const std::vector<std::string> &tf_output_labels = {},
                               const std::vector<std::string>& tf_input_keys = {},
                               MLCouplingData<In>* input_after_preprocessing = nullptr,
                               MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : MLCouplingProviderSmartsim(std::move(device), std::move(model_backend), std::string(), std::move(model), std::move(model_name), std::move(host), port, std::vector<std::string>(), std::vector<int>(), nodes, num_gpus, first_gpu, batch_size, min_batch_size, min_batch_timeout, command_timeout, socket_timeout, model_timeout, tf_input_labels, tf_output_labels, tf_input_keys, input_after_preprocessing, output_before_postprocessing) {};

    void validate_parameter(const std::string &device,
                            const std::string &model_backend,
                            const std::string &model_path,
                            const std::string_view &model,
                            const std::string &model_name,
                            const std::string &host,
                            int port,
                            const std::vector<std::string> &hosts,
                            const std::vector<int> &ports,
                            int nodes,
                            int num_gpus,
                            int first_gpu,
                            int batch_size,
                            int min_batch_size,
                            int min_batch_timeout,
                            int command_timeout,
                            int socket_timeout,
                            int model_timeout,
                            const std::vector<std::string> &tf_input_labels,
                            const std::vector<std::string>& tf_output_labels,
                            const std::vector<std::string>& tf_input_keys)
    {
        guarantee(!model_name.empty(), "model_name must be specified");
        guarantee(!(model_path.empty() && model.empty()), "Either model_path or model must be specified");
        guarantee(model_path.empty() || model.empty(), "Cannot specify both model_path and model");

        guarantee(device == "CPU" || device == "GPU", "Device must be either 'CPU' or 'GPU'");
        guarantee(num_gpus != -1, "num_gpus could not be resolved; set num_gpus explicitly or check Slurm GPU env vars");
        guarantee(!(device == "GPU" && num_gpus == 0), "If device is GPU, num_gpus cannot be 0");
        guarantee(model_backend == "TF" || model_backend == "ONNX" || model_backend == "TFLITE" || model_backend == "TORCH", "Model backend must be either 'TF', 'ONNX', 'TFLITE', or 'TORCH'");
        guarantee(num_gpus >= 0, "num_gpus cannot be negative");
        guarantee(first_gpu >= 0, "first_gpu cannot be negative");
        guarantee(num_gpus == 0 || first_gpu < num_gpus, "first_gpu must be less than num_gpus");

        guarantee(nodes > 0, "nodes must be greater than 0");
        guarantee(nodes != 2, "nodes cannot be 2 for smartsim provider, as smartsim (actually redis to be exact) does not support 2-node clusters (2 nodes could have issues with deciding which is down in case of failure, while 3 or more nodes can have a majority vote)");

        bool is_ssdb_set = getenv("SSDB") != nullptr;

        guarantee(is_ssdb_set || (!host.empty() && (port > 0 && port < 65535) || (!hosts.empty() && hosts.size() == ports.size())), "If SSDB environment variable is not set, then either host and port must be specified, or hosts and ports must be specified with matching sizes");

        guarantee(batch_size >= 0, "batch_size cannot be negative");
        guarantee(min_batch_size >= 0, "min_batch_size cannot be negative");
        guarantee(min_batch_timeout >= 0, "min_batch_timeout cannot be negative");
        guarantee(command_timeout >= -1, "command_timeout cannot be less than -1");
        guarantee(socket_timeout >= -1, "socket_timeout cannot be less than -1");
        guarantee(model_timeout >= -1, "model_timeout cannot be less than -1");

        guarantee((model_backend == "TF" || model_backend == "ONNX" || model_backend == "TFLITE" || model_backend == "TORCH") && (tf_input_labels.empty() && tf_output_labels.empty()), "tf_input_labels and tf_output_labels can only be specified for TF and TFLITE backends");
        guarantee(tf_input_labels.empty() == tf_output_labels.empty(), "tf_input_labels and tf_output_labels must be specified together for TF and TFLITE backends");
    }

    void static_inference(MLCouplingData<In> *input_after_preprocessing,
                          MLCouplingData<Out> *output_before_postprocessing) override
    {

        guarantee(input_after_preprocessing != nullptr, "Smartsim inference requires input_after_preprocessing.");
        guarantee(output_before_postprocessing != nullptr, "Smartsim inference requires output_before_postprocessing.");

        auto &input_data_after_preprocessing = *input_after_preprocessing;
        auto &output_data_before_postprocessing = *output_before_postprocessing;

#ifdef WITH_SMARTSIM

        auto to_size_t_dims = [](const std::vector<int> &dims)
        {
            std::vector<size_t> converted_dims;
            converted_dims.reserve(dims.size());
            for (int dim : dims)
            {
                converted_dims.push_back(static_cast<size_t>(dim));
            }
            return converted_dims;
        };

#ifdef USE_SCOREP
        const bool profile_details = ml_coupling_scorep::detailed_regions_are_enabled();
        SCOREP_USER_METRIC_LOCAL(smartsim_input_bytes);
        SCOREP_USER_METRIC_LOCAL(smartsim_output_bytes);
        SCOREP_USER_REGION_DEFINE(handle_smartsim_chunk_plan)
        SCOREP_USER_REGION_DEFINE(handle_smartsim_put_tensor)
        SCOREP_USER_REGION_DEFINE(handle_smartsim_run_model)
        SCOREP_USER_REGION_DEFINE(handle_smartsim_unpack_tensor)
        if (profile_details) {
        static bool scorep_smartsim_metrics_initialized = false;
        if (!scorep_smartsim_metrics_initialized) {
            SCOREP_USER_METRIC_INIT(smartsim_input_bytes, "smartsim_input_bytes", "bytes", SCOREP_USER_METRIC_TYPE_UINT64, SCOREP_USER_METRIC_CONTEXT_CALLPATH);
            SCOREP_USER_METRIC_INIT(smartsim_output_bytes, "smartsim_output_bytes", "bytes", SCOREP_USER_METRIC_TYPE_UINT64, SCOREP_USER_METRIC_CONTEXT_CALLPATH);
            scorep_smartsim_metrics_initialized = true;
        }
        }
#endif

        size_t max_bytes = 500ULL * 1024 * 1024; // 500 MiB limit
        size_t num_chunks = 1;

#ifdef USE_SCOREP
        if (profile_details) {
        SCOREP_USER_REGION_BEGIN(handle_smartsim_chunk_plan, "smartsim_chunk_plan", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif
        for (size_t i = 0; i < input_data_after_preprocessing.size(); ++i) {
            auto &tensor = input_data_after_preprocessing[i];
            size_t bytes = tensor.numel() * tensor.element_size();
            if (bytes > max_bytes) {
                size_t chunks = (bytes + max_bytes - 1) / max_bytes;
                if (chunks > num_chunks) num_chunks = chunks;
            }
        }
        for (size_t i = 0; i < output_data_before_postprocessing.size(); ++i) {
            auto &tensor = output_data_before_postprocessing[i];
            size_t bytes = tensor.numel() * tensor.element_size();
            if (bytes > max_bytes) {
                size_t chunks = (bytes + max_bytes - 1) / max_bytes;
                if (chunks > num_chunks) num_chunks = chunks;
            }
        }

        if (num_chunks > 1) {
            for (size_t i = 0; i < input_data_after_preprocessing.size(); ++i) {
                auto &tensor = input_data_after_preprocessing[i];
                std::vector<size_t> dims = to_size_t_dims(tensor.dimensions());
                size_t dim0 = dims.empty() ? 1 : dims[0];
                if (num_chunks > dim0) {
                    throw std::runtime_error("Cannot chunk input tensor " + std::to_string(i) + ": single batch element size exceeds SmartSim 500 MiB limit.");
                }
            }
            for (size_t i = 0; i < output_data_before_postprocessing.size(); ++i) {
                auto &tensor = output_data_before_postprocessing[i];
                std::vector<size_t> dims = to_size_t_dims(tensor.dimensions());
                size_t dim0 = dims.empty() ? 1 : dims[0];
                if (num_chunks > dim0) {
                    throw std::runtime_error("Cannot chunk output tensor " + std::to_string(i) + ": single batch element size exceeds SmartSim 500 MiB limit.");
                }
            }
            logging::debug("Tensor size exceeds 500 MiB limit. Splitting inference into " + std::to_string(num_chunks) + " chunks.");
        }
#ifdef USE_SCOREP
        if (profile_details) {
        SCOREP_USER_REGION_END(handle_smartsim_chunk_plan)
        }
#endif

        if (std::getenv("DEBUG_PROVIDER_INPUT")) {
            for (size_t ti = 0; ti < input_data_after_preprocessing.size(); ++ti) {
                auto &t = input_data_after_preprocessing[ti];
                int n = static_cast<int>(t.numel());
                double sum = 0, ssq = 0;
                float first = 0, last = 0;
                if constexpr (std::is_same_v<In, float>) {
                    const float* raw = static_cast<const float*>(t.root());
                    first = raw[0]; last = raw[n-1];
                    for (int i = 0; i < n; ++i) { sum += raw[i]; ssq += raw[i]*raw[i]; }
                } else if constexpr (std::is_same_v<In, double>) {
                    const double* raw = static_cast<const double*>(t.root());
                    first = raw[0]; last = raw[n-1];
                    for (int i = 0; i < n; ++i) { sum += raw[i]; ssq += raw[i]*raw[i]; }
                }
                std::cerr << "DEBUG SMARTSIM INPUT rank=" << this->rank << " tensor=" << ti << " shape=";
                for (int d : t.dimensions()) std::cerr << d << " ";
                std::cerr << " numel=" << n << " sum=" << sum << " sumSq=" << ssq << " first=" << first << " last=" << last << std::endl;
            }
        }

        for (size_t chunk_idx = 0; chunk_idx < num_chunks; ++chunk_idx) {
            if (num_chunks > 1) logging::debug("Write these tensors to SmartSim (chunk " + std::to_string(chunk_idx+1) + "/" + std::to_string(num_chunks) + "):");
            else logging::debug("Write these tensors to SmartSim:");

            std::vector<std::string> input_tensor_names;
            for (size_t tensor_index = 0; tensor_index < input_data_after_preprocessing.size(); ++tensor_index)
            {
                std::string input_name;
                if (this->tf_input_keys.empty()) {
                    input_name = "input_" + std::to_string(this->rank) + "_" + std::to_string(tensor_index);
                } else {
                    guarantee(tensor_index < this->tf_input_keys.size(), "Not enough tf_input_keys provided for multi-input model.");
                    input_name = this->tf_input_keys[tensor_index];
                }
                input_name = key_balancer.prefix_key(input_name);
                
                auto &tensor = input_data_after_preprocessing[tensor_index];
                
                if (num_chunks > 1 && !tensor.is_contiguous()) {
                    throw std::runtime_error("Chunking of tensors > 500MiB is currently only supported for contiguous memory layouts.");
                }

                std::vector<size_t> dims = to_size_t_dims(tensor.dimensions());
                size_t dim0 = dims.empty() ? 1 : dims[0];
                size_t start_dim0 = (dim0 * chunk_idx) / num_chunks;
                size_t end_dim0 = (dim0 * (chunk_idx + 1)) / num_chunks;
                size_t cur_dim0 = end_dim0 - start_dim0;
                if (!dims.empty()) dims[0] = cur_dim0;
                
                size_t offset_elements = start_dim0 * (tensor.numel() / dim0);
                void *data = (void*)((char*)tensor.root() + offset_elements * tensor.element_size());

                MLCouplingDataType ml_type = to_ml_coupling_data_type<In>();
                SRTensorType sr_type = to_srtensor_type(ml_type);
                logging::debug("  " + tensor.to_string("Tensor " + std::to_string(tensor_index)));

#ifdef USE_SCOREP
                if (profile_details) {
                SCOREP_USER_METRIC_UINT64(smartsim_input_bytes, tensor.numel() * tensor.element_size());
                SCOREP_USER_REGION_BEGIN(handle_smartsim_put_tensor, "smartsim_put_tensor", SCOREP_USER_REGION_TYPE_COMMON)
                }
#endif
                client->put_tensor(input_name, data, dims, sr_type, to_sr_memory_layout(tensor.layout()));
#ifdef USE_SCOREP
                if (profile_details) {
                SCOREP_USER_REGION_END(handle_smartsim_put_tensor)
                }
#endif
                input_tensor_names.push_back(input_name);

                std::cout << "DEBUG: put_tensor " << input_name << " dims=[";
                for (size_t i = 0; i < dims.size(); ++i) std::cout << dims[i] << (i+1==dims.size()?"":",");
                std::cout << "]" << std::endl;
            }

            logging::debug("Input tensor names sent to SmartSim:");
            for (const auto &name : input_tensor_names) { logging::debug("  " + name); }

            std::vector<std::string> output_tensor_names;
            for (size_t tensor_index = 0; tensor_index < output_data_before_postprocessing.size(); ++tensor_index)
            {
                std::string output_name = "output_" + std::to_string(this->rank) + "_" + std::to_string(tensor_index);
                output_tensor_names.push_back(key_balancer.prefix_key(output_name));
            }

            logging::debug("Output tensor names expected from SmartSim:");
            for (const auto &name : output_tensor_names) { logging::debug("  " + name); }

            try {
#ifdef USE_SCOREP
                if (profile_details) {
                SCOREP_USER_REGION_BEGIN(handle_smartsim_run_model, "smartsim_run_model", SCOREP_USER_REGION_TYPE_COMMON)
                }
#endif
                if (this->device == "GPU") {
                    const int offset = this->rank >= 0 ? this->rank : 0;
                    client->run_model_multigpu(this->model_name, input_tensor_names, output_tensor_names, offset, this->first_gpu, this->num_gpus);
                } else {
                    client->run_model(this->model_name, input_tensor_names, output_tensor_names);
                }
#ifdef USE_SCOREP
                if (profile_details) {
                SCOREP_USER_REGION_END(handle_smartsim_run_model)
                }
#endif
            } catch (const std::exception& ex) {
                std::cerr << "run_model failed: " << ex.what() << std::endl;
                throw;
            }

            logging::debug("Retrieve these tensors from SmartSim:");
            for (size_t tensor_index = 0; tensor_index < output_data_before_postprocessing.size(); ++tensor_index)
            {
                const std::string& output_name = output_tensor_names[tensor_index];
                auto &tensor = output_data_before_postprocessing[tensor_index];
                
                if (num_chunks > 1 && !tensor.is_contiguous()) {
                    throw std::runtime_error("Chunking of tensors > 500MiB is currently only supported for contiguous memory layouts.");
                }

                std::vector<size_t> full_dims = to_size_t_dims(tensor.dimensions());
                size_t dim0 = full_dims.empty() ? 1 : full_dims[0];
                size_t start_dim0 = (dim0 * chunk_idx) / num_chunks;
                size_t end_dim0 = (dim0 * (chunk_idx + 1)) / num_chunks;
                size_t cur_dim0 = end_dim0 - start_dim0;
                
                std::vector<size_t> dims = full_dims;
                if (!dims.empty()) dims[0] = cur_dim0;

                size_t offset_elements = start_dim0 * (tensor.numel() / dim0);
                void *data = (void*)((char*)tensor.root() + offset_elements * tensor.element_size());

                MLCouplingDataType ml_type = to_ml_coupling_data_type<Out>();
                SRTensorType sr_type = to_srtensor_type(ml_type);
                SRMemoryLayout sr_layout = to_sr_memory_layout(tensor.layout());
                
                if (sr_layout == SRMemLayoutContiguous) {
                    size_t cur_numel = cur_dim0 * (tensor.numel() / dim0);
                    dims = { cur_numel };
                }

                try {
#ifdef USE_SCOREP
                    if (profile_details) {
                    SCOREP_USER_METRIC_UINT64(smartsim_output_bytes, tensor.numel() * tensor.element_size());
                    SCOREP_USER_REGION_BEGIN(handle_smartsim_unpack_tensor, "smartsim_unpack_tensor", SCOREP_USER_REGION_TYPE_COMMON)
                    }
#endif
                    client->unpack_tensor(output_name, data, dims, sr_type, sr_layout);
#ifdef USE_SCOREP
                    if (profile_details) {
                    SCOREP_USER_REGION_END(handle_smartsim_unpack_tensor)
                    }
#endif
                } catch (const std::exception& ex) {
                    std::cerr << "unpack_tensor failed for " << output_name << ": " << ex.what() << std::endl;
                    throw;
                }
                logging::debug("  " + tensor.to_string("Tensor " + std::to_string(tensor_index)));
            }
        }

#endif
    }

private:
    MLCouplingData<In> *input_after_preprocessing = nullptr;
    MLCouplingData<Out> *output_before_postprocessing = nullptr;
};
