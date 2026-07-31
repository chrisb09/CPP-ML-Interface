#pragma once

// The solver enables detailed regions only after the first ML inference.
// Keep this header-only because providers are compiled into separate binaries.
namespace ml_coupling_scorep {
inline bool detailed_regions_enabled = false;

inline void set_detailed_regions_enabled(bool enabled)
{
    detailed_regions_enabled = enabled;
}

inline bool detailed_regions_are_enabled()
{
    return detailed_regions_enabled;
}
} // namespace ml_coupling_scorep
