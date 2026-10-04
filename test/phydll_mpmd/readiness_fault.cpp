// Test executable interposition only: production clients never link this file.
#include <mpi.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>

extern "C" int MPI_Isend(const void* buffer, int count, MPI_Datatype datatype,
                         int dest, int tag, MPI_Comm comm, MPI_Request* request) {
    const char* fault = std::getenv("PHYDLL_TEST_TOKEN_FAULT");
    if (fault && datatype == MPI_UINT64_T && count == 1) {
        if (std::strcmp(fault, "sequence") == 0) {
            // Stable until abort; the faulty first frame must never complete.
            static const uint64_t wrong_sequence = UINT64_MAX;
            return PMPI_Isend(&wrong_sequence, count, datatype, dest, tag, comm, request);
        }
        if (std::strcmp(fault, "count") == 0)
            return PMPI_Isend(buffer, 0, datatype, dest, tag, comm, request);
    }
    return PMPI_Isend(buffer, count, datatype, dest, tag, comm, request);
}
