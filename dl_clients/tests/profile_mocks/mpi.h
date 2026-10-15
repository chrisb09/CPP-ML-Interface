#pragma once
#include <stdint.h>
typedef int MPI_Comm;
typedef int MPI_Fint;
typedef int MPI_Request;
typedef int MPI_Status;
#define MPI_COMM_NULL 0
#define MPI_COMM_WORLD 1
#define MPI_UNDEFINED -1
#define MPI_REQUEST_NULL 0
#define MPI_BYTE 1
#define MPI_UINT64_T 2
#define MPI_STATUSES_IGNORE nullptr
#ifdef __cplusplus
extern "C" {
#endif
int MPI_Init(int*, char***);
int MPI_Finalize(void);
int MPI_Comm_split(MPI_Comm, int, int, MPI_Comm*);
int MPI_Comm_dup(MPI_Comm, MPI_Comm*);
int MPI_Comm_free(MPI_Comm*);
int MPI_Recv(void*, int, int, int, int, MPI_Comm, MPI_Status*);
int MPI_Isend(const void*, int, int, int, int, MPI_Comm, MPI_Request*);
int MPI_Waitall(int, MPI_Request*, MPI_Status*);
int MPI_Abort(MPI_Comm, int);
int MPI_Barrier(MPI_Comm);
#ifdef __cplusplus
}
#endif
