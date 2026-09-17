#pragma once

#include <stdexcept>

#ifdef WITH_SMARTSIM
#include "sr_enums.h"
#endif

/**
 * @file ml_coupling_memory_layout.hpp
 * @brief Memory layout enumerations and transformation predicates for multi-dimensional arrays.
 */

/**
 * @brief Memory layout specifying ordering and pointer nesting of multi-dimensional tensors.
 * @ingroup cpp_core
 */
typedef enum
{
    MLCouplingMemLayoutInvalid = 0,          /**< Invalid or uninitialized memory layout */
    MLCouplingMemLayoutNested = 1,           /**< Row-major pointer-tree layout (C/C++ array of pointers) */
    MLCouplingMemLayoutContiguous = 2,       /**< Row-major contiguous buffer (C/C++ native single block) */
    MLCouplingMemLayoutFortranNested = 3,    /**< Column-major pointer-tree layout */
    MLCouplingMemLayoutFortranContiguous = 4 /**< Column-major contiguous buffer (Fortran native single block) */
} MLCouplingMemoryLayout;

/**
 * @brief Converts a memory layout enum to a string constant.
 * @param layout Layout enum.
 * @return String description of the layout.
 */
inline constexpr const char *to_string(MLCouplingMemoryLayout layout)
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

/**
 * @brief Tests whether the layout is a nested pointer-tree.
 * @param layout Layout to test.
 * @return True if nested (C or Fortran).
 */
inline constexpr bool is_nested_layout(MLCouplingMemoryLayout layout)
{
    return layout == MLCouplingMemLayoutNested || layout == MLCouplingMemLayoutFortranNested;
}

/**
 * @brief Tests whether the layout is a single contiguous flat buffer.
 * @param layout Layout to test.
 * @return True if contiguous.
 */
inline constexpr bool is_contiguous_layout(MLCouplingMemoryLayout layout)
{
    return layout == MLCouplingMemLayoutContiguous || layout == MLCouplingMemLayoutFortranContiguous;
}

/**
 * @brief Tests whether the layout uses Fortran column-major ordering.
 * @param layout Layout to test.
 * @return True if column-major.
 */
inline constexpr bool is_fortran_layout(MLCouplingMemoryLayout layout)
{
    return layout == MLCouplingMemLayoutFortranNested || layout == MLCouplingMemLayoutFortranContiguous;
}

/**
 * @brief Converts a layout to its corresponding contiguous equivalent (preserving row/column major ordering).
 * @param layout Input layout.
 * @return Contiguous memory layout.
 */
inline constexpr MLCouplingMemoryLayout to_contiguous_layout(MLCouplingMemoryLayout layout)
{
    if (layout == MLCouplingMemLayoutFortranNested || layout == MLCouplingMemLayoutFortranContiguous)
    {
        return MLCouplingMemLayoutFortranContiguous;
    }
    return MLCouplingMemLayoutContiguous;
}

/**
 * @brief Converts a layout to its corresponding nested pointer-tree equivalent.
 * @param layout Input layout.
 * @return Nested memory layout.
 */
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
