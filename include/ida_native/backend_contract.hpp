#pragma once

#include <string_view>

namespace ida_native {

// Public admission names for deployment-owned precision backends. The public
// binary can describe and reject these contracts, but it does not contain
// their implementation or instruction-level math.
inline constexpr std::string_view kBlackwellMxf4Fp4Backend =
    "blackwell_mxf4_fp4";
inline constexpr std::string_view kBlackwellMxf4Fp4RequestArch = "sm_120";
inline constexpr char kBlackwellMxf4Fp4Env[] = "IDA_NATIVE_MXF4";
inline constexpr std::string_view kBlackwellNvfp4PrecisionProfile =
    "blackwell_nvfp4";
inline constexpr std::string_view kBlackwellNvfp4RequestArch = "sm_120";
inline constexpr char kBlackwellNvfp4Env[] = "IDA_NATIVE_NVFP4";

// These are intentionally false in the public build. A private deployment
// may provide a different native binary through an opaque runtime package;
// an environment variable must never turn the public binary into an
// implementation selector.
inline constexpr bool mxf4_enabled() noexcept { return false; }
inline constexpr bool nvfp4_enabled() noexcept { return false; }

}  // namespace ida_native
