#pragma once

#ifdef USE_SCOREP
#include <scorep/SCOREP_User.h>
#endif

// Unconfigured callers retain the explicit manual gate.
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

// Override the shared provider gate only for an opted-in application step.
class InferenceScope {
public:
    InferenceScope(bool configured, bool selected)
        : configured_(configured), selected_(configured && selected),
          previous_(detailed_regions_enabled)
    {
        if (configured_) set_detailed_regions_enabled(selected_);
#ifdef USE_SCOREP
        if (selected_) {
            SCOREP_USER_REGION_BY_NAME_BEGIN("cmi_inference_steady", SCOREP_USER_REGION_TYPE_COMMON)
        }
#endif
    }

    ~InferenceScope()
    {
#ifdef USE_SCOREP
        if (selected_) {
            SCOREP_USER_REGION_BY_NAME_END("cmi_inference_steady")
        }
#endif
        if (configured_) set_detailed_regions_enabled(previous_);
    }

    InferenceScope(const InferenceScope&) = delete;
    InferenceScope& operator=(const InferenceScope&) = delete;

private:
    bool configured_;
    bool selected_;
    bool previous_;
};
} // namespace ml_coupling_scorep
