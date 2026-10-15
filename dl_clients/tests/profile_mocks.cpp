#include <algorithm>
#include <cassert>
#include <cstdlib>
#include <cstring>
#include <map>
#include <string>
#include <vector>
extern "C" {
#include "phydll.h"
}

namespace {
int sent = 0;
int received = 0;
int signals = 0;
int dest = 0;
double data = 3.0;
double* output = nullptr;
std::vector<std::string> stack;
std::map<std::string, int> visits;
int frames() { return std::atoi(std::getenv("TEST_FRAME_COUNT")); }
int warmups() {
    const char* env = std::getenv("CMI_SCOREP_WARMUP_CALLS");
    return env ? std::atoi(env) : 1;
}
}

void profile_begin(const char* name) {
    assert(sent >= warmups());
    if (std::strcmp(name, "dl_inference_steady") == 0) {
        assert(std::getenv("CMI_SCOREP_WARMUP_CALLS"));
        assert(stack.empty());
        assert(signals == sent + 1); // Successful control signal precedes the parent.
        assert(received == sent); // Parent starts before receive, not after it.
    } else if (std::getenv("CMI_SCOREP_WARMUP_CALLS")) {
        assert(!stack.empty() && stack.front() == "dl_inference_steady");
    }
    ++visits[name];
    stack.emplace_back(name);
}

void profile_end(const char* name) {
    assert(!stack.empty() && stack.back() == name);
    stack.pop_back();
}

extern "C" {
int MPI_Init(int*, char***) { return 0; }
int MPI_Finalize() { return 0; }
int MPI_Comm_split(MPI_Comm, int, int, MPI_Comm* comm) { *comm = 0; return 0; }
int MPI_Comm_dup(MPI_Comm, MPI_Comm* comm) { *comm = 2; return 0; }
int MPI_Comm_free(MPI_Comm* comm) { *comm = 0; return 0; }
int MPI_Recv(void* buffer, int size, int, int, int, MPI_Comm, MPI_Status*) {
    // Version-4 wire header: tiny packed frame, no model or shape payload.
    struct Header {
        int32_t magic, version, model_len, backend_len, device_len, batch_size, num_inputs, num_outputs;
        int64_t total_input, total_output;
        int32_t dtype, layout, num_input_dims, num_output_dims, layout_kind, phy_count, dl_count, readiness;
        int64_t field_size;
    };
    const Header header{0x4D4C434D, 4, 0, 0, 0, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1, 1};
    static_assert(sizeof(Header) == 88);
    assert(size == sizeof(header));
    std::memcpy(buffer, &header, sizeof(header));
    return 0;
}
int MPI_Isend(const void*, int, int, int, int, MPI_Comm, MPI_Request* request) { *request = 1; return 0; }
int MPI_Waitall(int, MPI_Request*, MPI_Status*) { return 0; }
int MPI_Abort(MPI_Comm, int code) { std::exit(code); }
int MPI_Barrier(MPI_Comm) { assert(stack.empty()); return 0; }
void phydll_init(char*) {}
void phydll_define_dl(int count) { assert(count == 1); }
void phydll_get_field_size(int* size) { *size = 1; }
void phydll_get_field_counts(int* phy, int* dl) { *phy = *dl = 1; }
int phydll_get_ndest() { return 1; }
int* phydll_get_dest() { return &dest; }
bool phydll_is_phy_signal() {
    // The real implementation blocks in a broadcast here. Its final negative
    // signal must stay outside all user regions and must not trigger a receive.
    assert(stack.empty());
    ++signals;
    return sent < frames();
}
void phydll_irecv() { assert(sent < frames()); ++received; }
void phydll_wait_irecv() {}
void phydll_get_field_view(const double** field, char* label) {
    *field = &data;
    std::strcpy(label, "PHY-DATA");
}
void phydll_set_field(double** field, char*) { output = *field; }
void phydll_send() { assert(output && *output == -data); ++sent; }
void phydll_finalize() {
    assert(stack.empty());
    assert(sent == frames() && received == frames() && signals == frames() + 1);
#ifdef USE_SCOREP
    const int expected = std::max(0, frames() - warmups());
    for (const char* name : {"dl_recv", "dl_frame_copy", "dl_output_allocate", "dl_send_output", "dl_send"})
        assert(visits[name] == expected);
    assert(visits["dl_inference_steady"] == (std::getenv("CMI_SCOREP_WARMUP_CALLS") ? expected : 0));
#else
    assert(visits.empty());
#endif
    std::free(output);
}
}
