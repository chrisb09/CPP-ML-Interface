#pragma once

#include "ml_coupling_provider.hpp"
#include "../tool.h"
#include "../data/ml_coupling_data_type.hpp"
#include "../data/ml_coupling_memory_layout.hpp"
#include "../logging.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <cstdlib> // for setenv

#if defined(WITH_SMARTSIM)
#include "client.h"
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
                               MLCouplingData<In> *input_after_preprocessing = nullptr,
                               MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : device(device),
          model_backend(model_backend),
          model_name(model_name),
          num_gpus(resolve_num_gpus(device, num_gpus)),
          first_gpu(first_gpu),
          batch_size(batch_size),
          min_batch_size(min_batch_size),
          min_batch_timeout(min_batch_timeout),
          command_timeout(command_timeout),
          socket_timeout(socket_timeout),
          model_timeout(model_timeout),
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
                           tf_output_labels);

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
                               MLCouplingData<In> *input_after_preprocessing = nullptr,
                               MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : MLCouplingProviderSmartsim(std::move(device), std::move(model_backend), std::move(model_path), std::string_view(), std::move(model_name), std::move(host), port, std::vector<std::string>(), std::vector<int>(), nodes, num_gpus, first_gpu, batch_size, min_batch_size, min_batch_timeout, command_timeout, socket_timeout, model_timeout, tf_input_labels, tf_output_labels, input_after_preprocessing, output_before_postprocessing) {};

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
                               MLCouplingData<In> *input_after_preprocessing = nullptr,
                               MLCouplingData<Out> *output_before_postprocessing = nullptr)
        : MLCouplingProviderSmartsim(std::move(device), std::move(model_backend), std::string(), std::move(model), std::move(model_name), std::move(host), port, std::vector<std::string>(), std::vector<int>(), nodes, num_gpus, first_gpu, batch_size, min_batch_size, min_batch_timeout, command_timeout, socket_timeout, model_timeout, tf_input_labels, tf_output_labels, input_after_preprocessing, output_before_postprocessing) {};

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
                            const std::vector<std::string> &tf_output_labels)
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

        logging::debug("Write these tensors to SmartSim:");

        std::vector<std::string> input_tensor_names;
        // loop over tensors
        for (size_t tensor_index = 0; tensor_index < input_data_after_preprocessing.size(); ++tensor_index)
        {
            std::string input_name = "input_" + std::to_string(this->rank) + "_" + std::to_string(tensor_index);
            // client->put_tensor(name, data, dims, type, mem_layout);
            auto &tensor = input_data_after_preprocessing[tensor_index];
            void *data = tensor.root();
            MLCouplingDataType ml_type = to_ml_coupling_data_type<In>();
            SRTensorType sr_type = to_srtensor_type(ml_type);
            std::vector<size_t> dims = to_size_t_dims(tensor.dimensions());
            logging::debug("  " + tensor.to_string("Tensor " + std::to_string(tensor_index)));

            client->put_tensor(input_name,
                               data,
                               dims,
                               sr_type,
                               to_sr_memory_layout(tensor.layout()));
            input_tensor_names.push_back(input_name);
        }

        logging::debug("Input tensor names sent to SmartSim:");
        for (const auto &name : input_tensor_names)
        {
            logging::debug("  " + name);
        }

        std::vector<std::string> output_tensor_names;
        for (size_t tensor_index = 0; tensor_index < output_data_before_postprocessing.size(); ++tensor_index)
        {
            std::string output_name = "output_" + std::to_string(this->rank) + "_" + std::to_string(tensor_index);
            output_tensor_names.push_back(output_name);
        }

        logging::debug("Output tensor names expected from SmartSim:");
        for (const auto &name : output_tensor_names)
        {
            logging::debug("  " + name);
        }

        try {
            if (this->device == "GPU")
            {
                const int offset = this->rank >= 0 ? this->rank : 0;
                client->run_model_multigpu(this->model_name, input_tensor_names, output_tensor_names, offset, this->first_gpu, this->num_gpus);
            }
            else
            {
                client->run_model(this->model_name, input_tensor_names, output_tensor_names);
            }
        } catch (const std::exception& ex) {
            std::cerr << "run_model failed: " << ex.what() << std::endl;
            throw;
        }

        /*
        void unpack_tensor(const std::string& name,
                           void* data,
                           const std::vector<size_t>& dims,
                           const SRTensorType type,
                           const SRMemoryLayout mem_layout);
        */
        logging::debug("Retrieve these tensors from SmartSim:");
        for (size_t tensor_index = 0; tensor_index < output_data_before_postprocessing.size(); ++tensor_index)
        {
            std::string output_name = "output_" + std::to_string(this->rank) + "_" + std::to_string(tensor_index);
            auto &tensor = output_data_before_postprocessing[tensor_index];
            void *data = tensor.root();
            MLCouplingDataType ml_type = to_ml_coupling_data_type<Out>();
            SRTensorType sr_type = to_srtensor_type(ml_type);
            SRMemoryLayout sr_layout = to_sr_memory_layout(tensor.layout());
            std::vector<size_t> dims;
            if (sr_layout == SRMemLayoutContiguous)
            {
                dims = { tensor.numel() };
            }
            else
            {
                dims = to_size_t_dims(tensor.dimensions());
            }
            try {
                client->unpack_tensor(output_name,
                                      data,
                                      dims,
                                      sr_type,
                                      sr_layout);
            } catch (const std::exception& ex) {
                std::cerr << "unpack_tensor failed for " << output_name << ": " << ex.what() << std::endl;
                throw;
            }
            logging::debug("  " + tensor.to_string("Tensor " + std::to_string(tensor_index)));
        }

#endif
    }

private:
    MLCouplingData<In> *input_after_preprocessing = nullptr;
    MLCouplingData<Out> *output_before_postprocessing = nullptr;
};
