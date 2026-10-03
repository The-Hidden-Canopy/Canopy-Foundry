#include "canopy_foundry/shell/sprite.hpp"

#include <nlohmann/json.hpp>

#include <algorithm>
#include <cctype>
#include <fstream>
#include <limits>
#include <sstream>
#include <string_view>
#include <utility>

namespace canopy_foundry::shell {
namespace {

using json = nlohmann::json;

constexpr SpriteColor kDefaultForeground{210, 215, 225, true};

std::vector<int> sgr_parameters(std::string_view sequence) {
    std::vector<int> values;
    std::size_t start = 0;
    while (start <= sequence.size()) {
        const auto separator = sequence.find(';', start);
        const auto end = separator == std::string_view::npos
            ? sequence.size()
            : separator;
        if (end == start) {
            values.push_back(0);
        } else {
            int value = 0;
            bool valid = true;
            for (std::size_t index = start; index < end; ++index) {
                const auto character = sequence[index];
                if (character < '0' || character > '9') {
                    valid = false;
                    break;
                }
                if (value > 1000000) {
                    valid = false;
                    break;
                }
                value = value * 10 + (character - '0');
            }
            if (valid) values.push_back(value);
        }
        if (separator == std::string_view::npos) break;
        start = separator + 1;
    }
    return values;
}

void apply_sgr(std::string_view sequence, SpriteColor& foreground, SpriteColor& background) {
    const auto values = sgr_parameters(sequence);
    for (std::size_t index = 0; index < values.size(); ++index) {
        const int code = values[index];
        if (code == 0) {
            foreground = kDefaultForeground;
            background = {};
        } else if (code == 39) {
            foreground = kDefaultForeground;
        } else if (code == 49) {
            background = {};
        } else if (code == 38 || code == 48) {
            const bool is_background = code == 48;
            if (index + 4 < values.size() && values[index + 1] == 2) {
                const auto clamp = [](int value) -> std::uint8_t {
                    return static_cast<std::uint8_t>(std::clamp(value, 0, 255));
                };
                SpriteColor parsed{
                    clamp(values[index + 2]),
                    clamp(values[index + 3]),
                    clamp(values[index + 4]),
                    true,
                };
                if (is_background) background = parsed;
                else foreground = parsed;
                index += 4;
            }
        }
    }
}

std::pair<std::uint32_t, std::size_t> decode_utf8(std::string_view value, std::size_t offset) {
    const auto byte = [&value](std::size_t index) {
        return static_cast<unsigned char>(value[index]);
    };
    const auto first = byte(offset);
    if (first < 0x80) return {first, 1};
    if ((first & 0xe0) == 0xc0 && offset + 1 < value.size()) {
        return {static_cast<std::uint32_t>(((first & 0x1f) << 6) | (byte(offset + 1) & 0x3f)), 2};
    }
    if ((first & 0xf0) == 0xe0 && offset + 2 < value.size()) {
        return {static_cast<std::uint32_t>(
                    ((first & 0x0f) << 12) |
                    ((byte(offset + 1) & 0x3f) << 6) |
                    (byte(offset + 2) & 0x3f)),
                3};
    }
    if ((first & 0xf8) == 0xf0 && offset + 3 < value.size()) {
        return {static_cast<std::uint32_t>(
                    ((first & 0x07) << 18) |
                    ((byte(offset + 1) & 0x3f) << 12) |
                    ((byte(offset + 2) & 0x3f) << 6) |
                    (byte(offset + 3) & 0x3f)),
                4};
    }
    return {0xfffd, 1};
}

std::uint8_t quadrant_mask(std::uint32_t codepoint) {
    switch (codepoint) {
        case 0x2598: return 0x1;  // upper-left
        case 0x259d: return 0x2;  // upper-right
        case 0x2596: return 0x4;  // lower-left
        case 0x2597: return 0x8;  // lower-right
        case 0x259a: return 0x9;  // diagonal upper-left/lower-right
        case 0x259e: return 0x6;  // diagonal upper-right/lower-left
        case 0x259b: return 0x7;
        case 0x259c: return 0xb;
        case 0x2599: return 0xd;
        case 0x259f: return 0xe;
        case 0x2580: return 0x3;  // upper half
        case 0x2584: return 0xc;  // lower half
        case 0x258c: return 0x5;  // left half
        case 0x2590: return 0xa;  // right half
        case 0x2588: return 0xf;
        default: return 0;
    }
}

std::vector<SpriteCell> parse_row(std::string_view row) {
    std::vector<SpriteCell> cells;
    SpriteColor foreground = kDefaultForeground;
    SpriteColor background;
    std::size_t offset = 0;
    while (offset < row.size()) {
        if (row[offset] == '\x1b' && offset + 1 < row.size() && row[offset + 1] == '[') {
            const auto terminator = row.find('m', offset + 2);
            if (terminator == std::string_view::npos) break;
            apply_sgr(row.substr(offset + 2, terminator - offset - 2), foreground, background);
            offset = terminator + 1;
            continue;
        }
        const auto [codepoint, length] = decode_utf8(row, offset);
        offset += std::max<std::size_t>(1, length);
        SpriteCell cell;
        cell.mask = quadrant_mask(codepoint);
        cell.foreground = foreground;
        cell.background = background;
        cells.push_back(cell);
    }
    return cells;
}

using SpriteFrameList = std::vector<SpriteFrame>;

bool parse_entry(const json& entry, SpriteFrameList& output, std::string& error) {
    if (!entry.is_object() || !entry.contains("frames") || !entry["frames"].is_array()) {
        error = "sprite set has no frame array";
        return false;
    }

    for (const auto& raw_frame : entry["frames"]) {
        if (!raw_frame.is_array() || raw_frame.empty()) {
            error = "sprite frame has no rows";
            return false;
        }
        SpriteFrame frame;
        std::vector<std::vector<SpriteCell>> rows;
        for (const auto& raw_row : raw_frame) {
            if (!raw_row.is_string()) {
                error = "sprite frame row is not text";
                return false;
            }
            rows.push_back(parse_row(raw_row.get<std::string>()));
            frame.width = std::max(frame.width, static_cast<int>(rows.back().size()));
        }
        frame.height = static_cast<int>(rows.size());
        if (frame.width == 0) {
            error = "sprite frame contains no cells";
            return false;
        }
        frame.cells.assign(
            static_cast<std::size_t>(frame.width * frame.height), SpriteCell{});
        for (int y = 0; y < frame.height; ++y) {
            for (int x = 0; x < static_cast<int>(rows[static_cast<std::size_t>(y)].size()); ++x) {
                frame.cells[static_cast<std::size_t>(y * frame.width + x)] =
                    rows[static_cast<std::size_t>(y)][static_cast<std::size_t>(x)];
            }
        }
        output.push_back(std::move(frame));
    }
    return !output.empty();
}

}  // namespace

bool load_ida_sprite_sheet(
    const std::filesystem::path& path,
    SpriteSheet& sheet,
    std::string& error
) {
    std::ifstream input(path, std::ios::binary);
    if (!input) {
        error = "could not open IDA CLI sprite sheet";
        sheet = {};
        sheet.path = path;
        sheet.error = error;
        return false;
    }
    std::ostringstream contents;
    contents << input.rdbuf();

    json root;
    try {
        root = json::parse(contents.str());
    } catch (const json::parse_error& parse_error) {
        error = std::string("invalid IDA CLI sprite sheet: ") + parse_error.what();
        sheet = {};
        sheet.path = path;
        sheet.error = error;
        return false;
    }
    if (!root.is_object()) {
        error = "IDA CLI sprite sheet must be an object";
        sheet = {};
        sheet.path = path;
        sheet.error = error;
        return false;
    }

    SpriteSheet candidate;
    candidate.path = path;
    std::string parse_error;
    const char* selected = nullptr;
    for (const char* name : {"WALK", "IDA"}) {
        if (!root.contains(name)) continue;
        SpriteFrameList frames;
        if (parse_entry(root.at(name), frames, parse_error) && !frames.empty()) {
            candidate.set_name = name;
            candidate.frames = std::move(frames);
            selected = name;
            break;
        }
    }
    if (selected == nullptr) {
        error = parse_error.empty()
            ? "IDA CLI sprite sheet contains neither WALK nor IDA frames"
            : parse_error;
        candidate.error = error;
        sheet = std::move(candidate);
        return false;
    }

    candidate.loaded = true;
    candidate.error.clear();
    sheet = std::move(candidate);
    error.clear();
    return true;
}

const SpriteFrame* sprite_frame_at(
    const SpriteSheet& sheet,
    std::uint64_t animation_tick
) noexcept {
    if (!sheet.loaded || sheet.frames.empty()) return nullptr;
    const auto hold = static_cast<std::uint64_t>(std::max(1, sheet.frame_hold_ticks));
    const auto index = (animation_tick / hold) % sheet.frames.size();
    return &sheet.frames[static_cast<std::size_t>(index)];
}

int sprite_breathe_offset(std::uint64_t animation_tick) noexcept {
    const int phase = static_cast<int>(animation_tick % 12);
    if (phase <= 6) return phase / 3;
    return (12 - phase) / 3;
}

}  // namespace canopy_foundry::shell
