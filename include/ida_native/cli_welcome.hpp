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
        << "Leaderboard: opt in only; normal training never submits leaderboard data.\n"
        << "Publication requires a separate confirmation after a successful run. "
           "If you opt in, only a public model ID/revision, sanitized GPU vendor/model, "
           "stable pseudonymous operator ID, approved metrics, and a deployment fingerprint "
           "are published. Raw UUID/serial, paths, datasets, checkpoints, credentials, "
           "stderr, and private telemetry stay local.\n"
        << "Donations and support: Hidden Canopy Hub - "
           "https://thehiddencanopy.com/flight-deck.html#support\n"
        << "Thank you for helping make native AI training more transparent "
           "and responsible.\n\n";
}

}  // namespace ida_native
