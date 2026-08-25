#pragma once

#include <filesystem>
#include <string>
#include <vector>

#include "ida_native/request.hpp"

namespace ida_native {

std::string json_escape(const std::string& value);
// Renders a finite float/double via std::to_string; a non-finite value
// (NaN/Inf) renders as the JSON literal `null` instead of std::to_string's
// own output for those values -- a bare, unquoted, lowercase "nan"/"inf"/
// "-inf", which is NOT valid JSON (strict parsers, including Python's
// stdlib json module, reject it; only the capitalized NaN/Infinity/
// -Infinity tokens are accepted as an extension). Every status/manifest
// writer emitting a training-derived float (loss, grad_norm, lr, etc. --
// any of which can legitimately go non-finite during a real divergence)
// must use this instead of a raw std::to_string call, so a diverging
// burn's own status file stays parseable by a standard-compliant reader
// instead of requiring a string-replace workaround.
std::string json_number(float value);
std::string json_number(double value);
std::string now_utc_iso8601();
void write_status(
    const std::filesystem::path& status_file,
    const NativeRequest& request,
    const std::string& phase,
    const std::vector<std::string>& extra_entries = {}
);

}  // namespace ida_native
