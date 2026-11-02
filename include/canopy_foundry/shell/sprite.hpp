#pragma once

#include <cstddef>
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

// Native full-colour source pixels, before the terminal atlas's two-colour
// quadrant packing. Transparent pixels leave the card background untouched.
struct NativeSpritePixel {
    std::uint8_t red{0}, green{0}, blue{0}, alpha{0};
};

struct NativeSpriteFrame {
    int width{0};
    int height{0};
    int view_left{0}, view_top{0}, view_width{0}, view_height{0};
    std::vector<NativeSpritePixel> pixels;
};

struct SpriteSheet {
    bool loaded{false};
    std::filesystem::path path;
    std::string set_name{"WALK"};
    std::string error;
    int frame_hold_ticks{6};
    std::vector<SpriteFrame> frames;
    std::vector<NativeSpriteFrame> native_frames;
    // Optional presentation order; frame data itself is retained intact.
    std::vector<std::size_t> presentation_frames;
};

struct SpriteSize { int width{0}; int height{0}; };

// Quadrants came from a terminal with cells twice as tall as they are wide.
// Fit that aspect in the available space instead of drawing square quadrants.
[[nodiscard]] SpriteSize fit_sprite(const SpriteFrame& frame, int width, int height) noexcept;
[[nodiscard]] SpriteSize fit_native_sprite(const NativeSpriteFrame& frame, int width, int height) noexcept;
[[nodiscard]] SpriteSize companion_card_size(int content_width) noexcept;

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

[[nodiscard]] const NativeSpriteFrame* native_sprite_frame_at(
    const SpriteSheet& sheet, std::uint64_t animation_tick
) noexcept;

// A small shared breathing motion keeps the sprite alive without implying
// runtime activity. It is presentation-only and does not read execution state.
[[nodiscard]] int sprite_breathe_offset(std::uint64_t animation_tick) noexcept;
// True when the pixels selected for the card can differ between two ticks.
[[nodiscard]] bool sprite_visual_changes(const SpriteSheet& sheet,
    std::uint64_t before, std::uint64_t after) noexcept;

}  // namespace canopy_foundry::shell
