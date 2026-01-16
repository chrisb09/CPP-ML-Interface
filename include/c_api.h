#ifdef __cplusplus
extern "C" {
#endif

    // First, let's define the types In and Out can be.
    // 0: float    // real(c_float): 32-bit floating point
    // 1: double   // real(c_double): 64-bit floating point
    // 2: int32_t  // integer(c_int32_t): 32-bit integer

    // Reines C! Keine "class", keine "templates".
    void* create_provider(const char* name, int in_selection, int out_selection, void** params, int param_count);
    void destroy_provider(void* handle);

#ifdef __cplusplus
}
#endif