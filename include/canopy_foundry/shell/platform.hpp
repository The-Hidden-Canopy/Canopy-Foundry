#pragma once

#include "canopy_foundry/shell/model.hpp"

namespace canopy_foundry::shell {

// Runs the platform-native window loop. The loop owns window lifecycle and
// local presentation only; it does not launch training or contact the Hub.
[[nodiscard]] int run_native_shell(ShellState& state);

}  // namespace canopy_foundry::shell
