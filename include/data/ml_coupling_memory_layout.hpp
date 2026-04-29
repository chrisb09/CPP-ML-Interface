#pragma once

#include <stdexcept>

#ifdef WITH_SMARTSIM
#include "sr_enums.h"
#endif

typedef enum
{
    MLCouplingMemLayoutInvalid = 0,           // Invalid or uninitialized memory layout
    MLCouplingMemLayoutNested = 1,            // Row-major pointer-tree layout (contiguous at innermost layer)
    MLCouplingMemLayoutContiguous = 2,        // Row-major contiguous layout
    MLCouplingMemLayoutFortranNested = 3,     // Column-major pointer-tree layout (contiguous at innermost layer)
    MLCouplingMemLayoutFortranContiguous = 4  // Column-major contiguous layout
} MLCouplingMemoryLayout;

inline constexpr const char* to_string(MLCouplingMemoryLayout layout)
{
    switch (layout)
    {
    case MLCouplingMemLayoutInvalid:
        return "Invalid";
    case MLCouplingMemLayoutNested:
        return "Nested";
    case MLCouplingMemLayoutContiguous:
        return "Contiguous";
    case MLCouplingMemLayoutFortranNested:
        return "FortranNested";
    case MLCouplingMemLayoutFortranContiguous:
        return "FortranContiguous";
    default:
        return "Unknown";
    }
}

inline constexpr bool is_nested_layout(MLCouplingMemoryLayout layout)
{
    return layout == MLCouplingMemLayoutNested || layout == MLCouplingMemLayoutFortranNested;
}

inline constexpr bool is_contiguous_layout(MLCouplingMemoryLayout layout)
{
    return layout == MLCouplingMemLayoutContiguous || layout == MLCouplingMemLayoutFortranContiguous;
}

inline constexpr bool is_fortran_layout(MLCouplingMemoryLayout layout)
{
    return layout == MLCouplingMemLayoutFortranNested || layout == MLCouplingMemLayoutFortranContiguous;
}

inline constexpr MLCouplingMemoryLayout to_contiguous_layout(MLCouplingMemoryLayout layout)
{
    if (layout == MLCouplingMemLayoutFortranNested || layout == MLCouplingMemLayoutFortranContiguous)
    {
        return MLCouplingMemLayoutFortranContiguous;
    }
    return MLCouplingMemLayoutContiguous;
}

inline constexpr MLCouplingMemoryLayout to_nested_layout(MLCouplingMemoryLayout layout)
{
    if (layout == MLCouplingMemLayoutFortranNested || layout == MLCouplingMemLayoutFortranContiguous)
    {
        return MLCouplingMemLayoutFortranNested;
    }
    return MLCouplingMemLayoutNested;
}

#ifdef WITH_SMARTSIM
inline SRMemoryLayout to_sr_memory_layout(MLCouplingMemoryLayout layout)
{
    switch (layout)
    {
    case MLCouplingMemLayoutNested:
        return SRMemLayoutNested;
    case MLCouplingMemLayoutContiguous:
        return SRMemLayoutContiguous;
    case MLCouplingMemLayoutFortranNested:
        return SRMemLayoutFortranNested;
    case MLCouplingMemLayoutFortranContiguous:
        return SRMemLayoutFortranContiguous;
    default:
        throw std::invalid_argument("Invalid MLCouplingMemoryLayout");
    }
}

inline MLCouplingMemoryLayout to_ml_coupling_memory_layout(SRMemoryLayout layout)
{
    switch (layout)
    {
    case SRMemLayoutNested:
        return MLCouplingMemLayoutNested;
    case SRMemLayoutContiguous:
        return MLCouplingMemLayoutContiguous;
    case SRMemLayoutFortranNested:
        return MLCouplingMemLayoutFortranNested;
    case SRMemLayoutFortranContiguous:
        return MLCouplingMemLayoutFortranContiguous;
    default:
        throw std::invalid_argument("Invalid SRMemoryLayout");
    }
}
#endif
