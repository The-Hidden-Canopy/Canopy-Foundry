#pragma once

#include "ida_native/request.hpp"
#include "ida_native/trainer.hpp"

namespace ida_native {

void write_training_contract_files(
    const NativeRequest& request,
    const SmokeStepResult& result
);

}  // namespace ida_native
