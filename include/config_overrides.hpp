#pragma once

#include <cstdint>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

using ConfigOverrideValue = std::variant<int64_t, double, std::string, bool, std::vector<std::string>>;
using ConfigSectionOverrides = std::unordered_map<std::string, std::unordered_map<std::string, ConfigOverrideValue>>;
using ConfigDottedOverrides = std::unordered_map<std::string, ConfigOverrideValue>;

struct ConfigOverrides {
    ConfigSectionOverrides sections;
    ConfigDottedOverrides dotted;

    ConfigOverrides() = default;

    explicit ConfigOverrides(ConfigSectionOverrides section_overrides)
        : sections(std::move(section_overrides)) {}

    explicit ConfigOverrides(ConfigDottedOverrides dotted_overrides)
        : dotted(std::move(dotted_overrides)) {}

    ConfigOverrides(std::initializer_list<std::pair<const std::string, ConfigOverrideValue>> dotted_overrides)
        : dotted(dotted_overrides) {}

    bool empty() const {
        return sections.empty() && dotted.empty();
    }
};
