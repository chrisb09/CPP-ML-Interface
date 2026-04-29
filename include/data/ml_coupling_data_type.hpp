#pragma once

#include <cstdint>
#include <stdexcept>
#include <type_traits>
#include <stdexcept>
#include <variant>

#ifdef WITH_SMARTSIM
#include "sr_enums.h"
#endif

typedef enum
{
    MLCouplingDataTypeInvalid = 0, // Invalid or uninitialized tensor type
    MLCouplingDataTypeDouble = 1,  // Double-precision floating point tensor type
    MLCouplingDataTypeFloat = 2,   // Floating point tensor type
    MLCouplingDataTypeInt8 = 3,    // 8-bit signed integer tensor type
    MLCouplingDataTypeInt16 = 4,   // 16-bit signed integer tensor type
    MLCouplingDataTypeInt32 = 5,   // 32-bit signed integer tensor type
    MLCouplingDataTypeInt64 = 6,   // 64-bit signed integer tensor type
    MLCouplingDataTypeUint8 = 7,   // 8-bit unsigned integer tensor type
    MLCouplingDataTypeUint16 = 8   // 16-bit unsigned integer tensor type
} MLCouplingDataType;

inline constexpr const char* to_string(MLCouplingDataType type)
{
    switch (type)
    {
    case MLCouplingDataTypeInvalid:
        return "Invalid";
    case MLCouplingDataTypeDouble:
        return "Double";
    case MLCouplingDataTypeFloat:
        return "Float";
    case MLCouplingDataTypeInt8:
        return "Int8";
    case MLCouplingDataTypeInt16:
        return "Int16";
    case MLCouplingDataTypeInt32:
        return "Int32";
    case MLCouplingDataTypeInt64:
        return "Int64";
    case MLCouplingDataTypeUint8:
        return "Uint8";
    case MLCouplingDataTypeUint16:
        return "Uint16";
    default:
        return "Unknown";
    }
}

// A lightweight tag to carry type information
template <typename T> 
struct TypeTag { 
    using type = T; 
};

// A variant that can hold any of our supported type tags
using MLCouplingSupportedTypes = std::variant<
    TypeTag<double>, 
    TypeTag<float>,
    TypeTag<int8_t>,
    TypeTag<int16_t>,
    TypeTag<int32_t>,
    TypeTag<int64_t>,
    TypeTag<uint8_t>,
    TypeTag<uint16_t>
>;

template <typename T>
inline constexpr MLCouplingDataType to_ml_coupling_data_type();

inline MLCouplingSupportedTypes ml_coupling_data_type_to_supported_type(MLCouplingDataType type);

inline MLCouplingSupportedTypes get_type_tag(int selection) {
    return ml_coupling_data_type_to_supported_type(static_cast<MLCouplingDataType>(selection));
}

inline MLCouplingDataType to_ml_coupling_data_type(const MLCouplingSupportedTypes& type_tag)
{
    return std::visit([](auto&& tag) -> MLCouplingDataType {
        using T = typename std::decay_t<decltype(tag)>::type;
        return to_ml_coupling_data_type<T>();
    }, type_tag);
}

inline MLCouplingSupportedTypes ml_coupling_data_type_to_supported_type(MLCouplingDataType type)
{
    switch (type)
    {
    case MLCouplingDataTypeDouble:
        return TypeTag<double>{};
    case MLCouplingDataTypeFloat:
        return TypeTag<float>{};
    case MLCouplingDataTypeInt8:
        return TypeTag<int8_t>{};
    case MLCouplingDataTypeInt16:
        return TypeTag<int16_t>{};
    case MLCouplingDataTypeInt32:
        return TypeTag<int32_t>{};
    case MLCouplingDataTypeInt64:
        return TypeTag<int64_t>{};
    case MLCouplingDataTypeUint8:
        return TypeTag<uint8_t>{};
    case MLCouplingDataTypeUint16:
        return TypeTag<uint16_t>{};
    default:
        throw std::invalid_argument("Invalid MLCouplingDataType");
    }
}


#ifdef WITH_SMARTSIM
// Define a mapping from MLCouplingDataType to SmartSim's SRTensorType
inline SRTensorType to_srtensor_type(MLCouplingDataType type)
{
    switch (type)
    {
    case MLCouplingDataTypeDouble:
        return SRTensorTypeDouble;
    case MLCouplingDataTypeFloat:
        return SRTensorTypeFloat;
    case MLCouplingDataTypeInt8:
        return SRTensorTypeInt8;
    case MLCouplingDataTypeInt16:
        return SRTensorTypeInt16;
    case MLCouplingDataTypeInt32:
        return SRTensorTypeInt32;
    case MLCouplingDataTypeInt64:
        return SRTensorTypeInt64;
    case MLCouplingDataTypeUint8:
        return SRTensorTypeUint8;
    case MLCouplingDataTypeUint16:
        return SRTensorTypeUint16;
    default:
        throw std::invalid_argument("Invalid MLCouplingDataType");
    }
}

// Define a mapping from SRTensorType to MLCouplingDataType
inline MLCouplingDataType to_ml_coupling_data_type(SRTensorType type)
{
    switch (type)
    {
    case SRTensorTypeDouble:
        return MLCouplingDataTypeDouble;
    case SRTensorTypeFloat:
        return MLCouplingDataTypeFloat;
    case SRTensorTypeInt8:
        return MLCouplingDataTypeInt8;
    case SRTensorTypeInt16: 
        return MLCouplingDataTypeInt16;
    case SRTensorTypeInt32:
        return MLCouplingDataTypeInt32;
    case SRTensorTypeInt64:
        return MLCouplingDataTypeInt64;
    case SRTensorTypeUint8:
        return MLCouplingDataTypeUint8;
    case SRTensorTypeUint16:
        return MLCouplingDataTypeUint16;
    default:
        throw std::invalid_argument("Invalid SRTensorType");
    }
}
#endif

// Define a mapping from C++ types to MLCouplingDataType

template <typename T>
inline constexpr MLCouplingDataType to_ml_coupling_data_type()
{
    using DecayedType = std::remove_cv_t<std::remove_reference_t<T>>;

    if constexpr (std::is_same_v<DecayedType, double>)
    {
        return MLCouplingDataTypeDouble;
    }
    else if constexpr (std::is_same_v<DecayedType, float>)
    {
        return MLCouplingDataTypeFloat;
    }
    else
    {
        constexpr bool is_supported_integer_alias =
            std::is_same_v<DecayedType, signed char> ||
            std::is_same_v<DecayedType, unsigned char> ||
            std::is_same_v<DecayedType, char> ||
            std::is_same_v<DecayedType, short> ||
            std::is_same_v<DecayedType, unsigned short> ||
            std::is_same_v<DecayedType, int> ||
            std::is_same_v<DecayedType, unsigned int> ||
            std::is_same_v<DecayedType, long> ||
            std::is_same_v<DecayedType, unsigned long> ||
            std::is_same_v<DecayedType, long long> ||
            std::is_same_v<DecayedType, unsigned long long> ||
            std::is_same_v<DecayedType, int8_t> ||
            std::is_same_v<DecayedType, int16_t> ||
            std::is_same_v<DecayedType, int32_t> ||
            std::is_same_v<DecayedType, int64_t> ||
            std::is_same_v<DecayedType, uint8_t> ||
            std::is_same_v<DecayedType, uint16_t>;

        if constexpr (is_supported_integer_alias)
        {
            if constexpr (std::is_signed_v<DecayedType>)
            {
                if constexpr (sizeof(DecayedType) == sizeof(int8_t))
                {
                    return MLCouplingDataTypeInt8;
                }
                else if constexpr (sizeof(DecayedType) == sizeof(int16_t))
                {
                    return MLCouplingDataTypeInt16;
                }
                else if constexpr (sizeof(DecayedType) == sizeof(int32_t))
                {
                    return MLCouplingDataTypeInt32;
                }
                else if constexpr (sizeof(DecayedType) == sizeof(int64_t))
                {
                    return MLCouplingDataTypeInt64;
                }
                else
                {
                    return MLCouplingDataTypeInvalid;
                }
            }
            else
            {
                if constexpr (sizeof(DecayedType) == sizeof(uint8_t))
                {
                    return MLCouplingDataTypeUint8;
                }
                else if constexpr (sizeof(DecayedType) == sizeof(uint16_t))
                {
                    return MLCouplingDataTypeUint16;
                }
                else
                {
                    return MLCouplingDataTypeInvalid;
                }
            }
        }
        else
        {
            return MLCouplingDataTypeInvalid;
        }
    }
}

inline constexpr MLCouplingDataType kMLCouplingTypeInt = to_ml_coupling_data_type<int>();
inline constexpr MLCouplingDataType kMLCouplingTypeUnsignedInt = to_ml_coupling_data_type<unsigned int>();
inline constexpr MLCouplingDataType kMLCouplingTypeLong = to_ml_coupling_data_type<long>();
inline constexpr MLCouplingDataType kMLCouplingTypeUnsignedLong = to_ml_coupling_data_type<unsigned long>();
inline constexpr MLCouplingDataType kMLCouplingTypeLongLong = to_ml_coupling_data_type<long long>();
inline constexpr MLCouplingDataType kMLCouplingTypeUnsignedLongLong =
    to_ml_coupling_data_type<unsigned long long>();



//
