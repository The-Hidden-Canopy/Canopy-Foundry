#include "canopy_foundry/shell/platform.hpp"

#include <iostream>

namespace canopy_foundry::shell {

int run_native_shell(ShellState& state) {
    std::cerr
        << "foundry_shell: no native window backend is available on this build.\n"
        << "foundry_shell: use --headless for contract inspection, or install "
           "the platform window development package.\n"
        << "foundry_shell: current view=" << view_name(state.view) << '\n';
    return 2;
}

}  // namespace canopy_foundry::shell
