#include "ida_native/status.hpp"

#include <chrono>
#include <cmath>
#include <iomanip>
#include <sstream>

namespace ida_native {

std::string json_number(float value) {
    return std::isfinite(value) ? std::to_string(value) : "null";
}

std::string json_number(double value) {
    return std::isfinite(value) ? std::to_string(value) : "null";
}

std::string json_escape(const std::string& value) {
    std::string escaped;
    escaped.reserve(value.size());
    for (const char ch : value) {
        switch (ch) {
            case '\\': escaped += "\\\\"; break;
            case '"': escaped += "\\\""; break;
            case '\n': escaped += "\\n"; break;
            case '\r': escaped += "\\r"; break;
            case '\t': escaped += "\\t"; break;
            default: escaped += ch; break;
        }
    }
    return escaped;
}

std::string now_utc_iso8601() {
    const auto now = std::chrono::system_clock::now();
    const auto now_c = std::chrono::system_clock::to_time_t(now);
    std::tm utc{};
#if defined(_WIN32)
    gmtime_s(&utc, &now_c);
#else
    gmtime_r(&now_c, &utc);
#endif
    std::ostringstream output;
    output << std::put_time(&utc, "%Y-%m-%dT%H:%M:%SZ");
    return output.str();
}

}  // namespace ida_native
