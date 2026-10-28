#include "canopy_foundry/execution/service.hpp"
#include "canopy_foundry/shell/model.hpp"
#ifdef NDEBUG
#undef NDEBUG
#endif
#include <cassert>
#include <filesystem>
#include <string>

int main(int argc, char** argv) {
    assert(argc == 2);
    const auto root = std::filesystem::path(argv[1]);
    const auto request_path = root / "configs/examples/planning_only_request.json";
    const auto capabilities = canopy_foundry::build_capabilities();
    const auto via_service = canopy_foundry::plan_request(request_path, capabilities);
    const auto direct = canopy_foundry::resolve_execution_plan(
        ida_native::load_request(request_path), capabilities);
    assert(canopy_foundry::canonical_plan_json(via_service.plan) ==
           canopy_foundry::canonical_plan_json(direct.plan));
    canopy_foundry::shell::ShellState shell;
    shell.document.loaded = true;
    shell.document.path = request_path;
    canopy_foundry::shell::resolve_selected_request(shell);
    assert(shell.plan.resolved && shell.plan.accepted == via_service.plan.accepted);
    assert(shell.plan.master == via_service.plan.precision.master_storage);
    const auto run_check = canopy_foundry::check_local_run(request_path);
    assert(run_check.exit_code == 3 || run_check.exit_code == 4);
    canopy_foundry::shell::check_selected_run(shell);
    assert(shell.status == run_check.message);
    canopy_foundry::shell::cancel_selected_run(shell);
    assert(shell.status == canopy_foundry::cancel_local_run("").message);
    shell.document.path = root / "nonexistent-request.json";
    canopy_foundry::shell::resolve_selected_request(shell);
    assert(!shell.plan.resolved && shell.plan.master == "-");
    assert(shell.status.find("failed") != std::string::npos);
}
