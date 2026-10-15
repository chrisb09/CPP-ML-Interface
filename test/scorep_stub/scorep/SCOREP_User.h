#pragma once

#include <algorithm>
#include <cassert>
#include <string>
#include <vector>

namespace scorep_test {
struct Event {
    std::string name;
    bool steady;
};
inline std::vector<std::string> stack;
inline std::vector<Event> events;
inline void begin(const char* name)
{
    stack.emplace_back(name);
    events.push_back({name, std::find(stack.begin(), stack.end(), "cmi_inference_steady") != stack.end()});
}
inline void end(const char* name)
{
    assert(!stack.empty() && stack.back() == name);
    stack.pop_back();
}
inline int count(const char* name, bool selected_only = false)
{
    return std::count_if(events.begin(), events.end(), [&](const Event& event) {
        return event.name == name && (!selected_only || event.steady);
    });
}
inline void reset()
{
    assert(stack.empty());
    events.clear();
}
}

#define SCOREP_USER_REGION_TYPE_COMMON 0
#define SCOREP_USER_REGION_DEFINE(handle) static const char* handle = nullptr;
#define SCOREP_USER_REGION_BEGIN(handle, name, type) handle = name; scorep_test::begin(handle);
#define SCOREP_USER_REGION_END(handle) scorep_test::end(handle);
#define SCOREP_USER_REGION_BY_NAME_BEGIN(name, type) scorep_test::begin(name);
#define SCOREP_USER_REGION_BY_NAME_END(name) scorep_test::end(name);
