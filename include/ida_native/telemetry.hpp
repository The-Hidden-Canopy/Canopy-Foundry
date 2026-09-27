#pragma once

#include "ida_native/request.hpp"
#include "ida_native/trainer.hpp"

namespace ida_native {

void write_training_contract_files(
    const NativeRequest& request,
    const SmokeStepResult& result
);

// Write the bounded V2-style kernel telemetry bundle. This is an explicit
// local export; the native binary never uploads telemetry.
void write_shareable_kernel_telemetry(
    const std::filesystem::path& path,
    const NativeRequest& request,
    const SmokeStepResult& result
);

}  // namespace ida_native
