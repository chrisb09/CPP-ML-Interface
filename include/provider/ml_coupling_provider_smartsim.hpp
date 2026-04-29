#pragma once

#include "ml_coupling_provider_flexible.hpp"
#include "../tool.h"
#include "../data/ml_coupling_data_type.hpp"
#include "../data/ml_coupling_memory_layout.hpp"

#include <string>
#include <string_view>
#include <vector>
#include <cstdlib> // for setenv

# if defined(WITH_SMARTSIM)
#include "client.h"
# endif

// @registry_name: Smartsim
// @registry_aliases: smartsim, SmartSim
template <typename In, typename Out>
class MLCouplingProviderSmartsim : public MLCouplingProviderFlexible<In, Out>
{

    // local variables
    # if defined(WITH_SMARTSIM)
    SmartRedis::Client* client;
    # endif

    // parameters

    const std::string device;
    const std::string model_backend;
    const std::string model_name;
    const int num_gpus;
    const int first_gpu;
    const int batch_size;
    const int min_batch_size;
    const int min_batch_timeout;

    // TODO: we should have an intermediate abstract/virtual class that adds the requirement for send_data, which smartsim and phydll can do seperately, but which aixelerate cannot.

    /*
    SmartRedis runtime env vars:
    https://www.craylabs.org/docs/sr_runtime.html

    export SSDB="10.128.0.153:6379,10.128.0.154:6379,10.128.0.155:6379"
    // Redis database address list. Use a single host:port for standalone mode,
    // or a comma-separated host:port list for clustered mode. Prefixes like
    // "tcp://" are also supported.

    export SR_DB_TYPE="Clustered"
    // Database type: "Clustered" or "Standalone".

    export SR_CONN_INTERVAL=10
    // Milliseconds to wait between connection attempts.

    export SR_CONN_TIMEOUT=30
    // Total time in seconds to keep retrying connection attempts before failing.

    export SR_SOCKET_TIMEOUT=30
    // Per-attempt socket reply timeout in seconds. This can add to the overall
    // delay imposed by the connection and command retry timeouts.

    export SR_CMD_INTERVAL=10
    // Milliseconds to wait between command retry attempts.

    export SR_CMD_TIMEOUT=30
    // Total time in seconds to keep retrying a command before timing out.

    export SR_MODEL_TIMEOUT=300000
    // Milliseconds to wait for model execution before timing out.

    export SR_THREAD_COUNT=4
    // SmartRedis worker-pool thread count. Default is 4; 0 uses hardware concurrency.

    export SR_LOG_FILE="logs/smartredis.log"
    // Path to the SmartRedis log file.

    export SR_LOG_LEVEL="INFO"
    // Logging verbosity, e.g. QUIET, INFO, DEBUG, or DEVELOPER.
    */

    /* Other env vars:
    SR_MODEL_LOAD_BACKOFF_MS
    SR_MODEL_LOAD_RETRIES
    */

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
                               int nodes = 1,
                               int num_gpus = 0,
                               int first_gpu = 0,
                               int batch_size = 0,
                               int min_batch_size = 0,
                               int min_batch_timeout = 0,
                               const std::vector<std::string>& tf_input_labels = {},
                               const std::vector<std::string>& tf_output_labels = {})
                               :
          device(device),
          model_backend(model_backend),
          model_name(model_name),
          num_gpus(num_gpus),
          first_gpu(first_gpu),
          batch_size(batch_size),
          min_batch_size(min_batch_size),
          min_batch_timeout(min_batch_timeout)
    {

        # if !defined(WITH_SMARTSIM)
        guarantee(false, "SmartSim provider is not enabled. Please make sure WITH_SMARTSIM is defined and the necessary dependencies are installed.");
        # endif

        validate_parameter(this->device,
                           this->model_backend,
                           model_path,
                           std::string_view(), // model (not used in this provider)
                           this->model_name,
                           host,
                           port,
                           hosts,
                           ports,
                           nodes,
                           this->num_gpus,
                           this->first_gpu,
                           this->batch_size,
                           this->min_batch_size,
                           this->min_batch_timeout,
                           tf_input_labels,
                           tf_output_labels);

        // Before creating a smartsim client (SmartRedis to be exact), we need to set the appropriate env vars

        // Most importantly, SSDB has to be set if it isn't already, either via host and port, or hosts and ports
        std::string ssdb;
        if (host != "" && port != -1) {
            ssdb = host + ":" + std::to_string(port);
            setenv("SSDB", ssdb.c_str(), 1);
        } else if (!hosts.empty() && !ports.empty() && hosts.size() == ports.size()) {
            ssdb = "";
            for (int i = 0; i < hosts.size(); i++) {
                ssdb += hosts[i] + ":" + std::to_string(ports[i]);
                if (i < hosts.size() - 1) {
                    ssdb += ",";
                }
            }
        }

        // Setting the database type based on the number of nodes
        setenv("SR_DB_TYPE", nodes > 1 ? "Clustered" : "Standalone", 1);

        # if defined(WITH_SMARTSIM)

        int world_rank = this->rank;

        client = new SmartRedis::Client("solver_" + std::to_string(world_rank));

        if (this->rank <= 0) {
            std::cout << "SmartSim Coupling Provider initialized with the following parameters:" << std::endl;
            std::cout << "Device: " << this->device << std::endl;
            std::cout << "Model Backend: " << this->model_backend << std::endl;
            std::cout << "Number of GPUs: " << this->num_gpus << std::endl;
            std::cout << "First GPU: " << this->first_gpu << std::endl;
            std::cout << "Batch Size: " << this->batch_size << std::endl;
            std::cout << "Min Batch Size: " << this->min_batch_size << std::endl;
            std::cout << "Min Batch Timeout: " << this->min_batch_timeout << " ms" << std::endl;
            if (!ssdb.empty()) {
                std::cout << "SSDB: " << ssdb << std::endl;
            }

            // Load the model into the database

            if (!model_path.empty()) {
                // Load model from file path
                if (this->device == "GPU") {
                    client->set_model_from_file_multigpu(this->model_name, model_path, this->model_backend, first_gpu, num_gpus, batch_size, min_batch_size, min_batch_timeout, "", tf_input_labels, tf_output_labels);
                } else {
                    client->set_model_from_file(this->model_name, model_path, this->model_backend, "CPU", batch_size, min_batch_size, min_batch_timeout, "", tf_input_labels, tf_output_labels);
                }
            } else if (!model.empty()) {
                // Load model from string
                if (this->device == "GPU") {
                    client->set_model_multigpu(this->model_name, model, this->model_backend, first_gpu, num_gpus, batch_size, min_batch_size, min_batch_timeout, "", tf_input_labels, tf_output_labels);
                } else {
                    client->set_model(this->model_name, model, this->model_backend, "CPU", batch_size, min_batch_size, min_batch_timeout, "", tf_input_labels, tf_output_labels);
                }
            }
        }


        # endif
        
    }

public:
    MLCouplingProviderSmartsim(std::string device,
                               std::string model_backend,
                               std::string model_path,
                               std::string model_name = "model",
                               std::string host = "localhost",
                               int port = 6379,
                               int nodes = 1,
                               int num_gpus = 0,
                               int first_gpu = 0,
                               int batch_size = 0,
                               int min_batch_size = 0,
                               int min_batch_timeout = 0,
                               const std::vector<std::string>& tf_input_labels = {},
                               const std::vector<std::string>& tf_output_labels = {}
                            )
        : MLCouplingProviderSmartsim(std::move(device), std::move(model_backend), std::move(model_path), std::string_view(), std::move(model_name), std::move(host), port, std::vector<std::string>(), std::vector<int>(), nodes, num_gpus, first_gpu, batch_size, min_batch_size, min_batch_timeout, tf_input_labels, tf_output_labels)
        {};


    MLCouplingProviderSmartsim(std::string device,
                               std::string model_backend,
                               std::string_view model,
                               std::string model_name = "model",
                               std::string host = "localhost",
                               int port = 6379,
                               int nodes = 1,
                               int num_gpus = 0,
                               int first_gpu = 0,
                               int batch_size = 0,
                               int min_batch_size = 0,
                               int min_batch_timeout = 0,
                               const std::vector<std::string>& tf_input_labels = {},
                               const std::vector<std::string>& tf_output_labels = {}
                            )
        : MLCouplingProviderSmartsim(std::move(device), std::move(model_backend), std::string(), std::move(model), std::move(model_name), std::move(host), port, std::vector<std::string>(), std::vector<int>(), nodes, num_gpus, first_gpu, batch_size, min_batch_size, min_batch_timeout, tf_input_labels, tf_output_labels)
        {};

    void validate_parameter(const std::string& device,
                               const std::string& model_backend,
                               const std::string& model_path,
                               const std::string_view& model,
                               const std::string& model_name,
                               const std::string& host,
                               int port,
                               const std::vector<std::string>& hosts,
                               const std::vector<int>& ports,
                               int nodes,
                               int num_gpus,
                               int first_gpu,
                               int batch_size,
                               int min_batch_size,
                               int min_batch_timeout,
                               const std::vector<std::string> &tf_input_labels,
                               const std::vector<std::string> &tf_output_labels)
    {
        guarantee(!model_name.empty(), "model_name must be specified");
        guarantee(!(model_path.empty() && model.empty()), "Either model_path or model must be specified");
        guarantee(model_path.empty() || model.empty(), "Cannot specify both model_path and model");

        guarantee(device == "CPU" || device == "GPU", "Device must be either 'CPU' or 'GPU'");
        guarantee(!(device == "GPU" && num_gpus == 0), "If device is GPU, num_gpus cannot be 0");
        guarantee(model_backend == "TF" || model_backend == "ONNX" || model_backend == "TFLITE" || model_backend == "TORCH", "Model backend must be either 'TF', 'ONNX', 'TFLITE', or 'TORCH'");
        guarantee(num_gpus >= 0, "num_gpus cannot be negative");
        guarantee(first_gpu >= 0, "first_gpu cannot be negative");
        guarantee(num_gpus == 0 || first_gpu < num_gpus, "first_gpu must be less than num_gpus");

        guarantee(nodes > 0, "nodes must be greater than 0");
        guarantee(nodes != 2, "nodes cannot be 2 for smartsim provider, as smartsim (actually redis to be exact) does not support 2-node clusters (2 nodes could have issues with deciding which is down in case of failure, while 3 or more nodes can have a majority vote)");

        bool is_ssdb_set = getenv("SSDB") != nullptr;

        guarantee(is_ssdb_set || (!host.empty() && (port > 0 && port < 65535) || (!hosts.empty() && hosts.size() == ports.size())) , "If SSDB environment variable is not set, then either host and port must be specified, or hosts and ports must be specified with matching sizes");

        guarantee(batch_size >= 0, "batch_size cannot be negative");
        guarantee(min_batch_size >= 0, "min_batch_size cannot be negative");
        guarantee(min_batch_timeout >= 0, "min_batch_timeout cannot be negative");

        guarantee((model_backend == "TF" || model_backend == "TFLITE") || (tf_input_labels.empty() && tf_output_labels.empty()), "tf_input_labels and tf_output_labels can only be specified for TF and TFLITE backends");
        guarantee(tf_input_labels.empty() == tf_output_labels.empty(), "tf_input_labels and tf_output_labels must be specified together for TF and TFLITE backends");
    }

    void send_data(MLCouplingData<In> input_data_after_preprocessing) override
    {
        // TODO
    }

    void inference(MLCouplingData<In> input_data_after_preprocessing, MLCouplingData<Out>& output_data_before_postprocessing) override
    {

        # ifdef WITH_SMARTSIM

        auto to_size_t_dims = [](const std::vector<int>& dims) {
            std::vector<size_t> converted_dims;
            converted_dims.reserve(dims.size());
            for (int dim : dims) {
                converted_dims.push_back(static_cast<size_t>(dim));
            }
            return converted_dims;
        };

        std::cout << "Write these tensors to SmartSim: " << std::endl;

        std::vector<std::string> input_tensor_names;
        // loop over tensors
        for (size_t tensor_index = 0; tensor_index < input_data_after_preprocessing.size(); ++tensor_index) {
            std::string input_name = "input_" + std::to_string(this->rank) + "_" + std::to_string(tensor_index);
            //client->put_tensor(name, data, dims, type, mem_layout);
            auto& tensor = input_data_after_preprocessing[tensor_index];
            void* data = tensor.root();
            MLCouplingDataType ml_type = to_ml_coupling_data_type<In>();
            SRTensorType sr_type = to_srtensor_type(ml_type);
            std::vector<size_t> dims = to_size_t_dims(tensor.dimensions());
            std::cout << "  " << tensor.to_string("Tensor " + std::to_string(tensor_index)) << std::endl;

            client->put_tensor(input_name,
                        data,
                        dims,
                        sr_type,
                        to_sr_memory_layout(tensor.layout()));
            input_tensor_names.push_back(input_name);


            // Sanity check, get the tensor back from the database and print it out to make sure it looks correct (this is especially important for GPU tensors to make sure the data is actually being sent to the database correctly, since GPU support in SmartSim is relatively new and we want to be sure this part works correctly before moving on to inference)
            
                void* retrieved_data = nullptr;
                std::vector<size_t> retrieved_dims = dims;
                SRTensorType retrieved_type = sr_type;
                client->get_tensor(input_name, retrieved_data, retrieved_dims, retrieved_type, to_sr_memory_layout(tensor.layout()));
                std::string message = "Sanity check for tensor sent to SmartSim with name: " + input_name;
                message += " and expected shape: " + tensor.shape_string() + " and expected dtype: " + to_string(tensor.data_type());
                std::cout << message << std::endl;
                std::vector<int> retrieved_dims_int;
                retrieved_dims_int.reserve(retrieved_dims.size());
                for (size_t dim : retrieved_dims) {
                    retrieved_dims_int.push_back(static_cast<int>(dim));
                }
                MLCouplingTensor<In> retrieved_tensor = MLCouplingTensor<In>(retrieved_data, std::move(retrieved_dims_int), tensor.layout(), MLCouplingOwnershipExternal);
                std::cout << "  " << retrieved_tensor.to_string("Retrieved Tensor") << std::endl;

        }

        
        std::cout << "Input tensor names sent to SmartSim: " << std::endl;
        for (const auto& name : input_tensor_names) {
            std::cout << "  " << name << std::endl;
        }

        
        std::vector<std::string> output_tensor_names;
        for (size_t tensor_index = 0; tensor_index < output_data_before_postprocessing.size(); ++tensor_index) {
            std::string output_name = "output_" + std::to_string(this->rank) + "_" + std::to_string(tensor_index);
            output_tensor_names.push_back(output_name);
        }

        std::cout << "Output tensor names expected from SmartSim: " << std::endl;
        for (const auto& name : output_tensor_names) {
            std::cout << "  " << name << std::endl;
        }

        if (this->device == "GPU") {
            const int offset = this->rank >= 0 ? this->rank : 0;
            client->run_model_multigpu(this->model_name, input_tensor_names, output_tensor_names, offset, this->first_gpu, this->num_gpus);
        } else {
            client->run_model(this->model_name, input_tensor_names, output_tensor_names);
        }

        /*
        void unpack_tensor(const std::string& name,
                           void* data,
                           const std::vector<size_t>& dims,
                           const SRTensorType type,
                           const SRMemoryLayout mem_layout);
        */
        std::cout << "Retrieve these tensors from SmartSim: " << std::endl;
        for (size_t tensor_index = 0; tensor_index < output_data_before_postprocessing.size(); ++tensor_index) {
            std::string output_name = "output_" + std::to_string(this->rank) + "_" + std::to_string(tensor_index);
            auto& tensor = output_data_before_postprocessing[tensor_index];
            void* data = tensor.root();
            MLCouplingDataType ml_type = to_ml_coupling_data_type<Out>();
            SRTensorType sr_type = to_srtensor_type(ml_type);
            std::vector<size_t> dims = to_size_t_dims(tensor.dimensions());
            client->unpack_tensor(output_name,
                        data,
                        dims,
                        sr_type,
                        to_sr_memory_layout(tensor.layout()));
            std::cout << "  " << tensor.to_string("Tensor " + std::to_string(tensor_index)) << std::endl;
        }

        # endif
        // TODO
    }

};
