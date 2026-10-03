#include "canopy_foundry/checkpoint/generation.hpp"

#include <iostream>

namespace {

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

std::string hash(char value) {
    return std::string(64, value);
}

canopy_foundry::CheckpointGeneration exact_generation() {
    return {
        41,
        12,
        hash('a'),
        hash('b'),
        hash('c'),
        hash('d'),
    };
}

}  // namespace

int main() {
    using namespace canopy_foundry;
    if (!require(checkpoint_purpose_name(CheckpointPurpose::ExactResume) == "exact_resume",
                 "checkpoint purpose name changed")) return 1;

    const auto candidate = exact_generation();
    if (!require(candidate.complete_for(CheckpointPurpose::ModelExport),
                 "complete model export generation was rejected")) return 1;
    if (!require(candidate.complete_for(CheckpointPurpose::ExactResume),
                 "complete exact-resume generation was rejected")) return 1;
    const auto valid = validate_checkpoint_generation(
        candidate, CheckpointPurpose::ExactResume, &candidate);
    if (!require(valid.valid, "matching checkpoint generation was rejected")) return 1;

    auto torn = candidate;
    torn.optimizer_sha256.clear();
    if (!require(!torn.complete_for(CheckpointPurpose::ExactResume),
                 "torn optimizer member was accepted for exact resume")) return 1;
    if (!require(torn.complete_for(CheckpointPurpose::ModelExport),
                 "model export was coupled to optimizer state")) return 1;

    auto malformed_optional = candidate;
    malformed_optional.optimizer_sha256 = "not-a-hash";
    if (!require(!malformed_optional.complete_for(CheckpointPurpose::ModelExport),
                 "malformed optional checkpoint hash was ignored")) return 1;

    auto mismatched = candidate;
    mismatched.generation++;
    const auto mismatch = validate_checkpoint_generation(
        mismatched, CheckpointPurpose::ExactResume, &candidate);
    if (!require(!mismatch.valid && mismatch.reason.find("agree") != std::string::npos,
                 "torn generation was not rejected with bounded evidence")) return 1;

    return 0;
}
