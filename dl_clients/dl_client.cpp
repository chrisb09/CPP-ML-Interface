#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <vector>

#include <mpi.h>

extern "C" {
#include "phydll.h"
}

#ifdef PHYDLL_DL_USE_TORCH
#include <torch/script.h>
#include <torch/torch.h>
#endif

namespace {
int get_env_int(const char *name, int fallback) {
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return std::atoi(value);
}

std::string get_env_str(const char *name, const std::string &fallback) {
    const char *value = std::getenv(name);
    if (!value || value[0] == '\0') {
        return fallback;
    }
    return std::string(value);
}
} // namespace

int main(int argc, char **argv) {
    MPI_Init(&argc, &argv);

    const std::string model_path = (argc > 1) ? argv[1] : get_env_str("PHYDLL_MODEL", "");
    const int dl_count = get_env_int("PHYDLL_DL_COUNT", 2);
    const std::string out_label_0 = get_env_str("PHYDLL_OUT_LABEL_0", "DL-OUT-0");
    const std::string out_label_1 = get_env_str("PHYDLL_OUT_LABEL_1", "DL-OUT-1");

#ifdef PHYDLL_DL_USE_TORCH
    torch::jit::script::Module model;
    if (!model_path.empty()) {
        try {
            model = torch::jit::load(model_path);
            model.eval();
        } catch (const c10::Error &e) {
            std::cerr << "Failed to load TorchScript model: " << e.what() << std::endl;
            MPI_Abort(MPI_COMM_WORLD, 1);
        }
    } else {
        std::cerr << "No model path provided (arg or PHYDLL_MODEL); continuing without model."
                  << std::endl;
    }
#else
    if (!model_path.empty()) {
        std::cerr << "Model path provided, but libtorch is not enabled." << std::endl;
    }
#endif

    phydll_init(const_cast<char *>("dl"));
    phydll_define_dl(dl_count);

    int field_size = 0;
    phydll_get_field_size(&field_size);
    if (field_size <= 0) {
        std::cerr << "Invalid field size received: " << field_size << std::endl;
        phydll_finalize();
        MPI_Finalize();
        return 1;
    }

    std::vector<double> phy_field_0(field_size);
    std::vector<double> phy_field_1(field_size);
    std::vector<double> phy_field_2(field_size);
    std::vector<double> dl_field_0(field_size, 0.0);
    std::vector<double> dl_field_1(field_size, 0.0);

    while (phydll_is_phy_signal()) {
        phydll_irecv();
        phydll_wait_irecv();

        char label_0[64] = {0};
        char label_1[64] = {0};
        char label_2[64] = {0};
        double *phy_ptr_0 = phy_field_0.data();
        double *phy_ptr_1 = phy_field_1.data();
        double *phy_ptr_2 = phy_field_2.data();

        phydll_get_field(&phy_ptr_0, label_0);
        phydll_get_field(&phy_ptr_1, label_1);
        phydll_get_field(&phy_ptr_2, label_2);

        bool used_model = false;

#ifdef PHYDLL_DL_USE_TORCH
        if (!model_path.empty()) {
            std::vector<float> input(field_size * 3);
            for (int i = 0; i < field_size; ++i) {
                input[i] = static_cast<float>(phy_field_0[i]);
                input[i + field_size] = static_cast<float>(phy_field_1[i]);
                input[i + 2 * field_size] = static_cast<float>(phy_field_2[i]);
            }

            auto options = torch::TensorOptions().dtype(torch::kFloat32);
            auto input_tensor = torch::from_blob(input.data(), {1, 3, field_size}, options).clone();

            auto output = model.forward({input_tensor}).toTensor();
            output = output.contiguous().view({-1});

            const int out_size = static_cast<int>(output.numel());
            const int copy_size = std::min(out_size, field_size * 2);
            auto output_ptr = output.data_ptr<float>();

            for (int i = 0; i < copy_size; ++i) {
                if (i < field_size) {
                    dl_field_0[i] = static_cast<double>(output_ptr[i]);
                } else {
                    dl_field_1[i - field_size] = static_cast<double>(output_ptr[i]);
                }
            }

            if (copy_size < field_size) {
                for (int i = copy_size; i < field_size; ++i) {
                    dl_field_0[i] = 0.0;
                }
            }
            if (copy_size < field_size * 2) {
                for (int i = std::max(0, copy_size - field_size); i < field_size; ++i) {
                    dl_field_1[i] = 0.0;
                }
            }

            used_model = true;
        }
#endif

        if (!used_model) {
            for (int i = 0; i < field_size; ++i) {
                dl_field_0[i] = -(50.0 + phy_field_0[i] + phy_field_2[i]);
                dl_field_1[i] = -(80.0 + phy_field_1[i] + phy_field_2[i]);
            }
        }

        double *dl_ptr_0 = dl_field_0.data();
        double *dl_ptr_1 = dl_field_1.data();

        char out_label_buf[64] = {0};
        std::snprintf(out_label_buf, sizeof(out_label_buf), "%s", out_label_0.c_str());
        phydll_set_field(&dl_ptr_0, out_label_buf);

        std::snprintf(out_label_buf, sizeof(out_label_buf), "%s", out_label_1.c_str());
        phydll_set_field(&dl_ptr_1, out_label_buf);

        phydll_send();
    }

    phydll_finalize();
    MPI_Finalize();
    return 0;
}
