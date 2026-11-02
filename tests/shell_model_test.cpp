#include "canopy_foundry/shell/model.hpp"

#include <nlohmann/json.hpp>

#include <cassert>
#include <filesystem>
#include <fstream>
#include <string>

int main() {
    using canopy_foundry::shell::ShellState;

    const auto root =
        std::filesystem::temp_directory_path() / "canopy-foundry-shell-model-test";
    std::error_code cleanup_error;
    std::filesystem::remove_all(root, cleanup_error);
    std::filesystem::create_directories(root);
    std::filesystem::create_directories(root / "frontend");
    for (const char* name : {"index.html", "styles.css", "app.js"}) {
        std::ofstream(root / "frontend" / name) << "fixture";
    }

    const nlohmann::json document = {
        {"schema", "canopy-foundry-execution-plan.v1"},
        {"device", {
            {"backend", "cuda"},
            {"required_arch", "sm_61"},
            {"parallel_mode", "single_device"},
            {"transport", "none"},
        }},
        {"model", {{"architecture_contract", "ida_lattice_native_v1"}}},
        {"optimizer", {{"type", "lion"}}},
        {"planner_revision", "planner-foundation-v2-port-v1"},
        {"precision", {{"master_storage", "bf16"}}},
    };
    const auto document_path = root / "plan.json";
    std::ofstream(document_path) << document.dump();

    ShellState state;
    state.frontend_root = root / "frontend";
    assert(canopy_foundry::shell::frontend_available(state.frontend_root));
    assert(std::string(canopy_foundry::shell::view_name(
               canopy_foundry::shell::View::Simulation)) == "Simulation");

    std::string error;
    assert(canopy_foundry::shell::load_document(state, document_path, error));
    assert(state.document.loaded);
    assert(state.document.type == "execution plan");
    assert(state.document.backend == "cuda");
    assert(state.document.required_arch == "sm_61");
    assert(state.document.model_contract == "ida_lattice_native_v1");
    assert(state.document.optimizer == "lion");
    assert(state.document.posture == "REQUEST PREVIEW");
    assert(error.empty());

    std::ofstream(root / "bad.json") << "[1, 2, 3]";
    assert(!canopy_foundry::shell::load_document(state, root / "bad.json", error));
    assert(error == "local JSON document must be an object");

    std::ofstream(root / "invalid.json") << "{";
    assert(!canopy_foundry::shell::load_document(state, root / "invalid.json", error));
    assert(error.rfind("invalid JSON:", 0) == 0);

    assert(!canopy_foundry::shell::load_document(state, root / "missing.json", error));
    assert(error == "could not open local JSON document");

    const std::string upper_left = "\xE2\x96\x98";
    const std::string lower_right = "\xE2\x97\x97";
    const nlohmann::json sprite = {
        {"WALK", {
            {"mode", "quadrant"},
            {"frames", nlohmann::json::array({
                nlohmann::json::array({
                    "\x1b[38;2;1;2;3m" + upper_left + " \x1b[0m",
                }),
                nlohmann::json::array({
                    "\x1b[38;2;4;5;6m" + lower_right + " \x1b[0m",
                }),
            })},
        }},
    };
    const auto sprite_path = root / "ida-cli-sprite.json";
    std::ofstream(sprite_path) << sprite.dump();

    canopy_foundry::shell::SpriteSheet sprite_sheet;
    assert(canopy_foundry::shell::load_ida_sprite_sheet(
        sprite_path, sprite_sheet, error));
    assert(sprite_sheet.loaded);
    assert(sprite_sheet.set_name == "WALK");
    assert(sprite_sheet.frames.size() == 2);
    assert(sprite_sheet.frames[0].width == 2);
    assert(sprite_sheet.frames[0].height == 1);
    assert(sprite_sheet.frames[0].cells[0].mask == 0x1);
    assert(sprite_sheet.frames[0].cells[0].foreground.red == 1);
    assert(sprite_sheet.frames[0].cells[0].foreground.green == 2);
    assert(sprite_sheet.frames[0].cells[0].foreground.blue == 3);
    assert(canopy_foundry::shell::sprite_frame_at(sprite_sheet, 0) ==
           &sprite_sheet.frames[0]);
    assert(canopy_foundry::shell::sprite_frame_at(sprite_sheet, 5) ==
           &sprite_sheet.frames[0]);
    assert(canopy_foundry::shell::sprite_frame_at(sprite_sheet, 6) ==
           &sprite_sheet.frames[1]);
    assert(canopy_foundry::shell::sprite_frame_at(sprite_sheet, 12) ==
           &sprite_sheet.frames[0]);

    // Idle pixels do not need a repaint every timer tick; breathing and a
    // legacy frame transition do. Missing assets never animate.
    assert(!canopy_foundry::shell::sprite_visual_changes(sprite_sheet, 0, 1));
    assert(canopy_foundry::shell::sprite_visual_changes(sprite_sheet, 2, 3));
    assert(canopy_foundry::shell::sprite_visual_changes(sprite_sheet, 5, 6));
    canopy_foundry::shell::SpriteSheet empty_sheet;
    assert(!canopy_foundry::shell::sprite_visual_changes(empty_sheet, 2, 3));

    auto sequenced = sprite;
    sequenced["WALK"]["presentation_frames"] = {1, 0, 1};
    std::ofstream(sprite_path) << sequenced.dump();
    assert(canopy_foundry::shell::load_ida_sprite_sheet(sprite_path, sprite_sheet, error));
    assert(sprite_sheet.frames.size() == 2);
    assert(canopy_foundry::shell::sprite_frame_at(sprite_sheet, 0) == &sprite_sheet.frames[1]);
    assert(canopy_foundry::shell::sprite_frame_at(sprite_sheet, 6) == &sprite_sheet.frames[0]);
    assert(canopy_foundry::shell::sprite_frame_at(sprite_sheet, 12) == &sprite_sheet.frames[1]);
    // The public sheet can become stale after loading; selection must refuse
    // a presentation index that no longer names a frame.
    auto stale_sequence = sprite_sheet;
    stale_sequence.frames.resize(1);
    assert(canopy_foundry::shell::sprite_frame_at(stale_sequence, 0) == nullptr);

    sequenced["WALK"]["presentation_frames"] = {2};
    std::ofstream(sprite_path) << sequenced.dump();
    assert(!canopy_foundry::shell::load_ida_sprite_sheet(sprite_path, sprite_sheet, error));
    assert(error == "invalid presentation frame index");

    canopy_foundry::shell::SpriteFrame dimensions;
    dimensions.width = 24;
    dimensions.height = 18;
    const auto fit = canopy_foundry::shell::fit_sprite(dimensions, 200, 138);
    assert(fit.width == 92 && fit.height == 138);
    const auto narrow = canopy_foundry::shell::fit_sprite(dimensions, 20, 138);
    assert(narrow.width == 20 && narrow.height == 30);
    assert(canopy_foundry::shell::fit_sprite(dimensions, 0, 138).width == 0);

    auto with_native = sprite;
    with_native["NATIVE_RGBA"] = {
        {"width", 2}, {"height", 1},
        {"frames", nlohmann::json::array({nlohmann::json::array({"12345680abcdef00"})})},
    };
    std::ofstream(sprite_path) << with_native.dump();
    assert(canopy_foundry::shell::load_ida_sprite_sheet(sprite_path, sprite_sheet, error));
    assert(sprite_sheet.native_frames.size() == 1);
    const auto* native = canopy_foundry::shell::native_sprite_frame_at(sprite_sheet, 0);
    assert(native && native->pixels.size() == 2);
    assert(native->pixels[0].red == 0x12 && native->pixels[0].green == 0x34 &&
           native->pixels[0].blue == 0x56 && native->pixels[0].alpha == 0x80);
    assert(native->pixels[1].alpha == 0);
    // A single native pose remains stable even when legacy frames advance.
    assert(canopy_foundry::shell::native_sprite_frame_at(sprite_sheet, 6) == native);
    const auto native_size = canopy_foundry::shell::fit_native_sprite(*native, 80, 80);
    assert(native_size.width == 80 && native_size.height == 40);
    with_native["NATIVE_RGBA"]["frames"][0][0] = "bad";
    std::ofstream(sprite_path) << with_native.dump();
    assert(canopy_foundry::shell::load_ida_sprite_sheet(sprite_path, sprite_sheet, error));
    assert(sprite_sheet.native_frames.empty()); // malformed optional art keeps legacy usable
    const auto small_card = canopy_foundry::shell::companion_card_size(672);
    const auto large_card = canopy_foundry::shell::companion_card_size(1326);
    assert(small_card.width == 224 && small_card.height == 208);
    assert(large_card.width == 320 && large_card.height == 304);

    std::ofstream(root / "bad-sprite.json") << "{\"WALK\": {\"frames\": []}}";
    canopy_foundry::shell::SpriteSheet bad_sprite;
    assert(!canopy_foundry::shell::load_ida_sprite_sheet(
        root / "bad-sprite.json", bad_sprite, error));
    assert(error.find("frames") != std::string::npos);

    canopy_foundry::shell::SpriteSheet missing_sprite;
    assert(!canopy_foundry::shell::load_ida_sprite_sheet(
        root / "missing-sprite.json", missing_sprite, error));
    assert(error == "could not open IDA CLI sprite sheet");
    assert(!missing_sprite.loaded);

    std::filesystem::remove_all(root, cleanup_error);
    return 0;
}
