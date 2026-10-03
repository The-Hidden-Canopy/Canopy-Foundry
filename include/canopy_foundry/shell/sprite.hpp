#pragma once

#include <cstdint>
#include <filesystem>
#include <string>
#include <vector>

namespace canopy_foundry::shell {

struct SpriteColor {
    std::uint8_t red{0};
    std::uint8_t green{0};
    std::uint8_t blue{0};
    bool opaque{false};
};

// A quadrant-mode terminal cell. The four low bits describe the visible
// quadrants in reading order: upper-left, upper-right, lower-left,
// lower-right. This keeps the CLI's palette and geometry without requiring a
// terminal, image library, or platform-specific renderer in Foundry.
struct SpriteCell {
    std::uint8_t mask{0};
    SpriteColor foreground;
    SpriteColor background;
};

struct SpriteFrame {
    int width{0};
    int height{0};
    std::vector<SpriteCell> cells;
};

struct SpriteSheet {
    bool loaded{false};
    std::filesystem::path path;
    std::string set_name{"WALK"};
    std::string error;
    int frame_hold_ticks{6};
    std::vector<SpriteFrame> frames;
};

// Reads the generated, terminal-ready IDA CLI sheet. The WALK set is preferred
// because it contains multiple poses; the one-frame IDA set remains a safe
// fallback when a smaller asset is supplied by a deployment.
[[nodiscard]] bool load_ida_sprite_sheet(
    const std::filesystem::path& path,
    SpriteSheet& sheet,
    std::string& error
);

[[nodiscard]] const SpriteFrame* sprite_frame_at(
    const SpriteSheet& sheet,
    std::uint64_t animation_tick
) noexcept;

// A small shared breathing motion keeps the sprite alive without implying
// runtime activity. It is presentation-only and does not read execution state.
[[nodiscard]] int sprite_breathe_offset(std::uint64_t animation_tick) noexcept;

}  // namespace canopy_foundry::shell
