#pragma once

#include <ostream>

namespace ida_native {

inline void print_cli_welcome(std::ostream& output) {
    output
        << "Welcome to Canopy Foundry - native C++ training for Hidden Canopy.\n"
        << "Instructions: read STARTER_GUIDE.md and "
           "docs/neural-foundry-guide.md.\n"
        << "Optional HF/Git cache: dot-source scripts/enable_hf_git_cache.ps1; "
           "read docs/hf-git-cache.md. Tokens and cache files stay local.\n"
        << "Power path: local config -> local data/checkpoint -> native trainer.\n"
        << "Control path: the Hub handles identity and bounded status; local paths stay here.\n"
        << "Kernel telemetry: disabled by default; use --kernel-telemetry DIR "
           "to write a bounded V2-style local bundle (32 MiB / 250,000 records) "
           "for a separate explicit uploader. Ambient telemetry settings are ignored.\n"
        << "The bundle contains wrapped-kernel launch geometry, host-enqueue timing, and "
           "step diagnostics only. Paths, datasets, checkpoints, credentials, "
           "stderr, IDs, device-duration claims, and promotion claims stay local.\n"
        << "Donations and support: Hidden Canopy Hub - "
           "https://thehiddencanopy.com/flight-deck.html#support\n"
        << "Thank you for helping make native AI training more transparent "
           "and responsible.\n\n";
}

}  // namespace ida_native
