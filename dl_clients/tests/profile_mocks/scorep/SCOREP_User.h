#pragma once
void profile_begin(const char*);
void profile_end(const char*);
#define SCOREP_USER_REGION_DEFINE(handle) static const char* handle
#define SCOREP_USER_REGION_TYPE_COMMON 0
#define SCOREP_USER_REGION_BEGIN(handle, name, type) do { handle = name; profile_begin(name); } while (0)
#define SCOREP_USER_REGION_END(handle) profile_end(handle)
