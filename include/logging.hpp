#pragma once

#include <iostream>
#include <stdexcept>
#include <string>
#include <algorithm>

namespace logging {

enum class Level { NONE, DEBUG, INFO, WARNING, ERROR };

inline Level GLOBAL_LEVEL = Level::INFO;
inline bool ERROR_SEPERATE = true; // If true, ERROR messages will be printed to std::cerr instead of std::cout

inline bool is_at_least(Level level, Level threshold) {
    return static_cast<int>(level) >= static_cast<int>(threshold);
}

inline std::string to_string(Level level) {
    switch (level) {
        case Level::NONE: return "NONE";
        case Level::DEBUG: return "DEBUG";
        case Level::INFO: return "INFO";
        case Level::WARNING: return "WARNING";
        case Level::ERROR: return "ERROR";
        default: return "UNKNOWN";
    }
}

inline void set_level(Level level) {
    GLOBAL_LEVEL = level;
}

inline void set_error_seperate(bool separate) {
    ERROR_SEPERATE = separate;
}

inline Level get_level(std::string level_str) {
    std::transform(level_str.begin(), level_str.end(), level_str.begin(), ::tolower);
    if (level_str == "none") return Level::NONE;
    if (level_str == "debug") return Level::DEBUG;
    if (level_str == "info") return Level::INFO;
    if (level_str == "warning") return Level::WARNING;
    if (level_str == "error") return Level::ERROR;
    throw std::invalid_argument("Invalid log level string: " + level_str);
}

inline void write(Level level, const std::string& message) {
    if (is_at_least(level, GLOBAL_LEVEL)) {
        if (level == Level::ERROR && ERROR_SEPERATE) {
            std::cerr << "[" << to_string(level) << "] " << message << std::endl;
        } else {
            std::cout << "[" << to_string(level) << "] " << message << std::endl;
        }
    }
}

inline void debug(const std::string& msg)   { write(Level::DEBUG, msg); }
inline void info(const std::string& msg)    { write(Level::INFO, msg); }
inline void warning(const std::string& msg) { write(Level::WARNING, msg); }
inline void error(const std::string& msg)   { write(Level::ERROR, msg); }

} // namespace log

