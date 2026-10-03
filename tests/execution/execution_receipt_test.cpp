#include "canopy_foundry/execution/receipt.hpp"

#include <iostream>

namespace {

bool require(bool condition, const char* message) {
    if (!condition) std::cerr << message << '\n';
    return condition;
}

}  // namespace

int main() {
    using namespace canopy_foundry;
    ExecutionReceipt receipt{};
    receipt.run_id = "local-test";
    receipt.request_sha256 = std::string(64, 'a');
    receipt.execution_plan_sha256 = std::string(64, 'b');
    receipt.evidence_tier = EvidenceTier::UnitVerified;

    receipt.observed = true;
    if (!require(!receipt_state_is_monotonic(receipt),
                 "observed receipt without accepted state was admitted")) return 1;

    receipt.requested = true;
    receipt.resolved = true;
    receipt.accepted = true;
    receipt.observed = true;
    receipt.persisted = true;
    receipt.model_export_complete = true;
    receipt.observation.moe_scheduler_reuses = 3;
    receipt.observation.transport = "host_staged";
    if (!require(receipt_state_is_monotonic(receipt),
                 "valid receipt state sequence was rejected")) return 1;

    const auto text = receipt_json(receipt);
    if (!require(text.find("\"requested\":true") != std::string::npos &&
                     text.find("\"evidence_tier\":\"UNIT_VERIFIED\"") != std::string::npos &&
                     text.find("host_staged") != std::string::npos,
                 "receipt did not serialize bounded evidence")) return 1;

    receipt.exact_resume_supported = true;
    receipt.persisted = false;
    if (!require(!receipt_state_is_monotonic(receipt),
                 "exact resume was presented before persistence")) return 1;
    return 0;
}
