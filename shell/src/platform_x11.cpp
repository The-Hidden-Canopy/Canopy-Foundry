#if defined(__linux__)

#include "canopy_foundry/shell/platform.hpp"
#include "canopy_foundry/shell/guide.hpp"

#include <X11/Xlib.h>
#include <X11/Xutil.h>
#include <X11/keysym.h>

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <sys/select.h>
#include <sys/wait.h>
#include <spawn.h>
#include <thread>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

extern char** environ;

namespace canopy_foundry::shell {
namespace {

constexpr int kTopHeight = 64;
constexpr int kLeftWidth = 204;
constexpr int kRightWidth = 316;

// XAllocColor is a server round trip and allocates a colormap reference. Reuse
// the colors for the lifetime of this display instead of allocating per pixel.
std::unordered_map<std::string, unsigned long> named_colors;
std::unordered_map<std::uint32_t, unsigned long> sprite_colors;
XFontStruct* stage_label_font = nullptr;

void open_guide_document(ShellState& state, GuideDocument document) {
    const auto path = guide_document_path(state.frontend_root, document);
    if (path.empty()) { state.status = "Local guide not found. Check the Foundry installation."; return; }
    auto filename = path.string();
    char program[] = "xdg-open";
    char* args[] = {program, filename.data(), nullptr};
    pid_t child = 0;
    const int error = posix_spawnp(&child, program, nullptr, nullptr, args, environ);
    if (error != 0) { state.status = "Could not start the local guide viewer."; return; }
    std::thread([child] { int status = 0; (void)waitpid(child, &status, 0); }).detach();
    state.status = "Opening local guide in the default application.";
}

unsigned long color(Display* display, const char* name) {
    const auto found = named_colors.find(name);
    if (found != named_colors.end()) return found->second;
    XColor exact{};
    XColor screen{};
    const Colormap colormap = DefaultColormap(display, DefaultScreen(display));
    const auto result = XAllocNamedColor(display, colormap, name, &screen, &exact) == 0
        ? BlackPixel(display, DefaultScreen(display)) : screen.pixel;
    named_colors.emplace(name, result);
    return result;
}

void fill(
    Display* display,
    Window window,
    GC gc,
    int left,
    int top,
    int width,
    int height,
    unsigned long value
) {
    XSetForeground(display, gc, value);
    XFillRectangle(
        display, window, gc, left, top,
        static_cast<unsigned int>(std::max(0, width)),
        static_cast<unsigned int>(std::max(0, height)));
}

void line(
    Display* display,
    Window window,
    GC gc,
    int x1,
    int y1,
    int x2,
    int y2,
    unsigned long value
) {
    XSetForeground(display, gc, value);
    XDrawLine(display, window, gc, x1, y1, x2, y2);
}

void text(
    Display* display,
    Window window,
    GC gc,
    const std::string& value,
    int x,
    int y,
    unsigned long value_color
) {
    XSetForeground(display, gc, value_color);
    XDrawString(
        display, window, gc, x, y,
        value.c_str(), static_cast<int>(value.size()));
}

// Restrict text to a column, then restore the GC's unrestricted clip.
void text_clipped(
    Display* display, Window window, GC gc, const std::string& value,
    int draw_x, int y, int clip_x, int width, unsigned long value_color
) {
    if (width <= 0 || clip_x > std::numeric_limits<short>::max()) return;
    const long long requested_right = static_cast<long long>(clip_x) + width;
    const int clip_left = std::max(0, clip_x);
    if (clip_left > std::numeric_limits<short>::max()) return;
    const long long clipped_width = std::min(requested_right - clip_left,
        static_cast<long long>(std::numeric_limits<unsigned short>::max()));
    if (clipped_width <= 0) return;
    XRectangle clip{static_cast<short>(clip_left), 0,
                    static_cast<unsigned short>(clipped_width),
                    std::numeric_limits<unsigned short>::max()};
    XSetClipRectangles(display, gc, 0, 0, &clip, 1, Unsorted);
    text(display, window, gc, value, draw_x, y, value_color);
    XSetClipMask(display, gc, None);
}

void text(
    Display* display, Window window, GC gc, const std::string& value,
    int x, int y, int width, unsigned long value_color
) {
    text_clipped(display, window, gc, value, x, y, x, width, value_color);
}

void panel(
    Display* display,
    Window window,
    GC gc,
    int left,
    int top,
    int right,
    int bottom,
    unsigned long fill_color,
    unsigned long border_color
) {
    fill(display, window, gc, left, top, right - left, bottom - top, fill_color);
    XSetForeground(display, gc, border_color);
    XDrawRectangle(
        display, window, gc, left, top,
        static_cast<unsigned int>(std::max(0, right - left)),
        static_cast<unsigned int>(std::max(0, bottom - top)));
}

unsigned long sprite_color(Display* display, const SpriteColor& value) {
    const std::uint32_t key = (static_cast<std::uint32_t>(value.red) << 16) |
                              (static_cast<std::uint32_t>(value.green) << 8) | value.blue;
    const auto found = sprite_colors.find(key);
    if (found != sprite_colors.end()) return found->second;
    XColor exact{};
    exact.red = static_cast<unsigned short>(value.red) * 257;
    exact.green = static_cast<unsigned short>(value.green) * 257;
    exact.blue = static_cast<unsigned short>(value.blue) * 257;
    exact.flags = DoRed | DoGreen | DoBlue;
    const Colormap colormap = DefaultColormap(display, DefaultScreen(display));
    const auto result = XAllocColor(display, colormap, &exact) == 0
        ? BlackPixel(display, DefaultScreen(display)) : exact.pixel;
    sprite_colors.emplace(key, result);
    return result;
}

void draw_sprite(
    Display* display,
    Window window,
    GC gc,
    const SpriteFrame& frame,
    int left,
    int top,
    SpriteSize size,
    unsigned long fallback
) {
    if (frame.width <= 0 || frame.height <= 0 || size.width <= 0 || size.height <= 0) return;
    const auto map_x = [&](int x) { return left + static_cast<int>(static_cast<std::int64_t>(x) * size.width / (frame.width * 2LL)); };
    const auto map_y = [&](int y) { return top + static_cast<int>(static_cast<std::int64_t>(y) * size.height / (frame.height * 2LL)); };
    for (int y = 0; y < frame.height; ++y) {
        for (int x = 0; x < frame.width; ++x) {
            const auto& cell = frame.cells[static_cast<std::size_t>(y * frame.width + x)];
            for (int part_y = 0; part_y < 2; ++part_y) {
                for (int part_x = 0; part_x < 2; ++part_x) {
                    const int bit = 1 << (part_y * 2 + part_x);
                    const bool foreground = (cell.mask & bit) != 0;
                    if (!foreground && !cell.background.opaque) continue;
                    const unsigned long value = foreground
                        ? sprite_color(display, cell.foreground)
                        : sprite_color(display, cell.background);
                    XSetForeground(display, gc, value == BlackPixel(display, DefaultScreen(display)) &&
                            !foreground ? fallback : value);
                    const int x0 = map_x(x * 2 + part_x);
                    const int y0 = map_y(y * 2 + part_y);
                    const int x1 = map_x(x * 2 + part_x + 1);
                    const int y1 = map_y(y * 2 + part_y + 1);
                    if (x1 <= x0 || y1 <= y0) continue;
                    XFillRectangle(
                        display,
                        window,
                        gc,
                        x0, y0,
                        static_cast<unsigned int>(x1 - x0),
                        static_cast<unsigned int>(y1 - y0));
                }
            }
        }
    }
}

bool draw_native_sprite(Display* display, Window window, GC gc, const NativeSpriteFrame& frame,
                        int left, int top, SpriteSize size) {
    if (frame.width <= 0 || frame.height <= 0 || size.width <= 0 || size.height <= 0 ||
        frame.view_left < 0 || frame.view_top < 0 || frame.view_width <= 0 || frame.view_height <= 0 ||
        frame.view_left + frame.view_width > frame.width || frame.view_top + frame.view_height > frame.height ||
        frame.pixels.size() != static_cast<std::size_t>(frame.width) * frame.height) return false;
    const int screen = DefaultScreen(display);
    Visual* visual = DefaultVisual(display, screen);
    XImage* image = XCreateImage(display, visual, static_cast<unsigned int>(DefaultDepth(display, screen)),
                                 ZPixmap, 0, nullptr, static_cast<unsigned int>(size.width),
                                 static_cast<unsigned int>(size.height), 32, 0);
    if (!image) return false;
    image->data = static_cast<char*>(std::calloc(static_cast<std::size_t>(image->bytes_per_line),
                                                  static_cast<std::size_t>(size.height)));
    if (!image->data) { XDestroyImage(image); return false; }
    const auto component = [](int value, unsigned long mask) -> unsigned long {
        if (!mask) return 0;
        unsigned int shift = 0;
        while (((mask >> shift) & 1UL) == 0) ++shift;
        const auto max_value = mask >> shift;
        return ((static_cast<unsigned long>(value) * max_value + 127) / 255 << shift) & mask;
    };
    for (int y = 0; y < size.height; ++y) {
        const int sy = frame.view_top + static_cast<int>(static_cast<std::int64_t>(y) * frame.view_height / size.height);
        for (int x = 0; x < size.width; ++x) {
            const int sx = frame.view_left + static_cast<int>(static_cast<std::int64_t>(x) * frame.view_width / size.width);
            const auto& pixel = frame.pixels[static_cast<std::size_t>(sy * frame.width + sx)];
            const auto blend = [&](int value, int bg) { return (value * pixel.alpha + bg * (255 - pixel.alpha) + 127) / 255; };
            const int r = blend(pixel.red, 16), g = blend(pixel.green, 30), b = blend(pixel.blue, 42);
            const auto value = visual->c_class == TrueColor
                ? component(r, visual->red_mask) | component(g, visual->green_mask) | component(b, visual->blue_mask)
                : sprite_color(display, SpriteColor{static_cast<std::uint8_t>(r), static_cast<std::uint8_t>(g), static_cast<std::uint8_t>(b), true});
            XPutPixel(image, x, y, value);
        }
    }
    XPutImage(display, window, gc, image, 0, 0, left, top,
              static_cast<unsigned int>(size.width), static_cast<unsigned int>(size.height));
    XDestroyImage(image);
    return true;
}

void draw_ida_card(
    Display* display,
    Window window,
    GC gc,
    ShellState& state,
    int left,
    int top,
    int right,
    int bottom
) {
    if (!state.companion_visible) return;
    const auto teal = color(display, "#55d5c5");
    panel(display, window, gc, left, top, right, bottom,
          color(display, "#101e2a"), color(display, "#286967"));
    text(display, window, gc, "IDA CLI / LOCAL PRESENCE", left + 16, top + 27, teal);
    const auto* frame = sprite_frame_at(state.ida_sprite, state.animation_tick);
    if (frame == nullptr) {
        text(display, window, gc, "Presentation atlas unavailable.", left + 16, top + 80,
             color(display, "#b9c5d1"));
        text(display, window, gc, "No execution claim is implied.", left + 16, top + 108,
             color(display, "#748294"));
        return;
    }
    const int available_width = std::max(1, right - left - 24);
    const int available_height = std::max(1, bottom - top - 70);
    const auto* native = native_sprite_frame_at(state.ida_sprite, state.animation_tick);
    const auto size = native ? fit_native_sprite(*native, available_width, available_height)
                             : fit_sprite(*frame, available_width, available_height);
    const int sprite_left = left + (right - left - size.width) / 2;
    const int sprite_top = top + 40 + sprite_breathe_offset(state.animation_tick);
    if (!native || !draw_native_sprite(display, window, gc, *native, sprite_left, sprite_top, size)) {
        const auto fallback_size = fit_sprite(*frame, available_width, available_height);
        draw_sprite(display, window, gc, *frame, left + (right - left - fallback_size.width) / 2,
                    sprite_top, fallback_size, color(display, "#101e2a"));
    }
    text(display, window, gc,
         "PRESENTATION ONLY",
         left + 12, bottom - 13, color(display, "#748294"));
}

int simulation_stage_index(SimulationStage stage) {
    switch (stage) {
        case SimulationStage::Requested: return 0;
        case SimulationStage::Resolved: return 1;
        case SimulationStage::Accepted: return 2;
        case SimulationStage::Routing: return 2;
        case SimulationStage::Executing: return 3;
        case SimulationStage::Checkpointing: return 4;
        case SimulationStage::Persisted: return 5;
        case SimulationStage::Idle: return -1;
    }
    return -1;
}

void facts(
    Display* display,
    Window window,
    GC gc,
    int left,
    int top,
    int width,
    const std::vector<std::pair<std::string, std::string>>& values,
    unsigned long muted,
    unsigned long body,
    unsigned long border
) {
    line(display, window, gc, left, top, left + width, top, border);
    int y = top + 20;
    for (const auto& item : values) {
        text(display, window, gc, item.first, left, y, muted);
        text(display, window, gc, item.second, left + width / 2, y, body);
        line(display, window, gc, left, y + 10, left + width, y + 10, border);
        y += 30;
    }
}

void render(Display* display, Window window, GC gc, ShellState& state, int width, int height) {
    const auto bg = color(display, "#0a0e14");
    const auto raised = color(display, "#141c27");
    const auto rail = color(display, "#0d121a");
    const auto border = color(display, "#1e2a38");
    const auto border_strong = color(display, "#2b3a4d");
    const auto body = color(display, "#b9c5d1");
    const auto bright = color(display, "#e7edf3");
    const auto muted = color(display, "#748294");
    const auto deep = color(display, "#526071");
    const auto teal = color(display, "#55d5c5");
    const auto amber = color(display, "#f0ae68");
    const auto red = color(display, "#ed7875");

    fill(display, window, gc, 0, 0, width, height, bg);
    fill(display, window, gc, 0, 0, width, kTopHeight, bg);
    line(display, window, gc, 0, kTopHeight - 1, width, kTopHeight - 1, border);
    text(display, window, gc, "CANOPY FOUNDRY", 61, 27, bright);
    text(display, window, gc, "NATIVE LOCAL SHELL", 61, 45, muted);
    fill(display, window, gc, 24, 24, 9, 18, teal);
    fill(display, window, gc, 37, 20, 9, 22, teal);
    fill(display, window, gc, 50, 16, 9, 26, teal);
    fill(display, window, gc, width - 144, 29, 7, 7, teal);
    text(display, window, gc, "LOCAL ONLY", width - 128, 35, muted);
    text(display, window, gc, "NATIVE", width - 65, 35, body);

    fill(display, window, gc, 0, kTopHeight, kLeftWidth, height - kTopHeight, rail);
    line(display, window, gc, kLeftWidth - 1, kTopHeight, kLeftWidth - 1, height, border);
    text(display, window, gc, "WORKSPACE", 22, 92, deep);
    const View views[] = {View::Overview, View::Plan, View::Capabilities, View::Evidence};
    const char* labels[] = {"Overview", "Plan lens", "Capabilities", "Evidence"};
    for (int index = 0; index < 4; ++index) {
        const int top = 112 + index * 44;
        const bool active = state.view == views[index];
        if (active) {
            panel(display, window, gc, 13, top, kLeftWidth - 14, top + 35,
                  color(display, "#132b2f"), color(display, "#286967"));
            fill(display, window, gc, 13, top, 3, 35, teal);
        }
        text(display, window, gc, labels[index], 38, top + 22, active ? bright : muted);
        text(display, window, gc, "0" + std::to_string(index + 1), 166, top + 22, deep);
    }
    text(display, window, gc, "BOUNDARY", 22, 319, deep);
    if (state.view == View::Boundaries) {
        panel(display, window, gc, 13, 343, kLeftWidth - 14, 378,
              color(display, "#132b2f"), color(display, "#286967"));
        fill(display, window, gc, 13, 343, 3, 35, teal);
    }
    text(display, window, gc, "Ownership", 38, 365, state.view == View::Boundaries ? bright : muted);
    text(display, window, gc, "05", 166, 365, deep);

    text(display, window, gc, "PLAYGROUND", 22, 410, deep);
    if (state.view == View::Simulation) {
        panel(display, window, gc, 13, 426, kLeftWidth - 14, 461,
              color(display, "#132b2f"), color(display, "#286967"));
        fill(display, window, gc, 13, 426, 3, 35, color(display, "#55d5c5"));
    }
    text(display, window, gc, "Simulation", 38, 449,
         state.view == View::Simulation ? bright : muted);
    text(display, window, gc, "S", 166, 449, deep);
    panel(display, window, gc, 22, 480, 177, 518, raised, border_strong);
    text(display, window, gc, state.mode == WorkspaceMode::Ida ? "Train Brain" : "Back to IDA", 35, 505, body);
    text(display, window, gc, state.companion_visible ? "Hide IDA" : "Show IDA", 38, 544, muted);
    text(display, window, gc, "Help / Guide  H", 38, 582, muted);
    line(display, window, gc, 22, height - 109, kLeftWidth - 22, height - 109, border);
    text(display, window, gc, "STANDALONE SURFACE", 22, height - 85, teal);
    text(display, window, gc, "Local contracts only.", 22, height - 62, muted);
    text(display, window, gc, "planner foundation / v1", 22, height - 37, deep);

    const int left = kLeftWidth + 45;
    const int right = width - kRightWidth - 28;
    const auto ida_card = companion_card_size(right - left);
    text(display, window, gc, "CANOPY FOUNDRY / " + std::string(view_name(state.view)),
         left, 98, muted);
    if (state.guide_visible) {
        text(display, window, gc, "WELCOME TO FOUNDRY", left, 143, teal);
        text(display, window, gc, "Start with IDA. Choose Train Brain to open Weave.", left, 190, bright);
        draw_ida_card(display, window, gc, state, right - ida_card.width, 132, right, 132 + ida_card.height);
        text(display, window, gc, "1. Check: foundry capabilities --pretty", left, 250, body);
        text(display, window, gc, "Device/toolkit: nvidia-smi and nvcc --version.", left, 273, muted);
        text(display, window, gc, "2. Plan: see the Starter Guide example", left, 315, body);
        text(display, window, gc, "Check Run reports preflight only; it does not train.", left, 423, muted);
        text(display, window, gc, "One-step plan fixture only; it has no training data.", left, 338, muted);
        text(display, window, gc, "3. Load a request and resolve it in Train Brain.", left, 380, body);
        text(display, window, gc, "Checkpoint inspection requires native validation.", left, 403, muted);
        text(display, window, gc, "CUDA needs a supported GPU/toolkit. Fix a rejected plan and retry.", left, 445, amber);
        text(display, window, gc, "Open a local guide:", left, 490, body);
        panel(display, window, gc, left, 510, left + 145, 548, raised, border_strong);
        panel(display, window, gc, left + 155, 510, left + 300, 548, raised, border_strong);
        text(display, window, gc, "1  Starter guide", left + 10, 535, body);
        text(display, window, gc, "2  Worker guide", left + 165, 535, body);
        text(display, window, gc, state.status, left, 660, muted);
        text(display, window, gc, state.view == View::Plan ? "Here: resolve request or inspect resume." : "Here: choose Train Brain to open planning.", left, 550, amber);
        panel(display, window, gc, left, 580, left + 145, 620, raised, border_strong);
        text(display, window, gc, "Continue", left + 12, 605, bright);
        text(display, window, gc, "Reopen with H or Help / Guide", left + 160, 605, muted);
    } else if (state.view == View::Overview) {
        text(display, window, gc, "LOCAL EXECUTION / INSPECTABLE BY DEFAULT", left, 143, teal);
        text(display, window, gc, "Make the execution contract visible.", left, 190, bright);
        text(display, window, gc, "A native shell for local Foundry contracts.", left, 225, body);
        text(display, window, gc, "It shows the current document and evidence posture", left, 244, body);
        text(display, window, gc, "without launching training or contacting the Hub.", left, 263, body);
        draw_ida_card(display, window, gc, state, right - ida_card.width, 132, right, 132 + ida_card.height);
        panel(display, window, gc, left, 302, left + 145, 339,
              color(display, "#286e6c"), teal);
        text(display, window, gc, "Load local JSON", left + 13, 326, bright);
        panel(display, window, gc, left + 157, 302, left + 267, 339, raised, border_strong);
        text(display, window, gc, "Train Brain", left + 170, 326, body);
        panel(display, window, gc, left + 279, 302, left + 407, 339,
              color(display, "#286e6c"), teal);
        text(display, window, gc, "Run simulation", left + 291, 326, bright);
        text(display, window, gc, state.status, left, 375, muted);
        panel(display, window, gc, left, std::max(430, height - 205), left + 330, height - 24,
              color(display, "#142c30"), color(display, "#286967"));
        text(display, window, gc, "CURRENT LENS", left + 18, std::max(430, height - 205) + 25, teal);
        text(display, window, gc, "Request -> plan", left + 18, std::max(430, height - 205) + 54, bright);
        facts(display, window, gc, left + 18, std::max(430, height - 205) + 78, 294, {
            {"SOURCE", state.document.loaded ? state.document.name : "None loaded"},
            {"DOCUMENT", state.document.loaded ? state.document.type : "no document"},
            {"NETWORK", "disabled"},
        }, muted, body, border);
    } else if (state.view == View::Plan) {
        text(display, window, gc, state.mode == WorkspaceMode::TrainBrain ? "WEAVE / Plan" : "Plan lens", left, 155, bright);
        text(display, window, gc, "DISPLAYED CONTRACT IS NOT PROOF OF EXECUTION", left + 230, 155, amber);
        const int column = std::max(270, (right - left - 30) / 3);
        const int top = 204;
        for (int index = 0; index < 3; ++index) {
            const int panel_left = left + index * (column + 15);
            panel(display, window, gc, panel_left, top, panel_left + column, 560,
                  index == 1 ? color(display, state.mode == WorkspaceMode::TrainBrain ? "#302221" : "#142d30") : raised,
                  index == 1 ? color(display, state.mode == WorkspaceMode::TrainBrain ? "#935841" : "#286967") : border);
        }
        text(display, window, gc, "INPUT DOCUMENT", left + 18, top + 25, muted);
        text(display, window, gc, state.document.loaded ? state.document.name : "No local JSON loaded",
             left + 18, top + 53, body);
        facts(display, window, gc, left + 18, top + 82, column - 36, {
            {"SCHEMA", state.document.schema},
            {"BACKEND", state.document.backend},
            {"ARCH", state.document.required_arch},
            {"MODEL", state.document.model_contract},
            {"OPTIMIZER", state.document.optimizer},
        }, muted, body, border);
        const int middle = left + column + 15;
        text(display, window, gc, "RESOLVED SHAPE", middle + 18, top + 25,
             state.mode == WorkspaceMode::TrainBrain ? amber : teal);
        text(display, window, gc, "ExecutionPlan", middle + 18, top + 53, bright);
        facts(display, window, gc, middle + 18, top + 82, column - 36, {
            {"PLANNER", state.plan.planner_revision},
            {"PARALLEL", state.plan.parallel_mode},
            {"TRANSPORT", state.plan.transport},
            {"PLAN", state.plan.resolved ? (state.plan.accepted ? "accepted" : "rejected") : "not resolved"},
            {"RUNTIME", "verify separately"},
        }, muted, body, border);
        const int last = middle + column + 15;
        text(display, window, gc, "PRECISION ROLES", last + 18, top + 25, muted);
        text(display, window, gc, "Representation map", last + 18, top + 53, bright);
        facts(display, window, gc, last + 18, top + 82, column - 36, {
            {"MASTER", state.plan.master},
            {"FORWARD", state.plan.forward},
            {"BACKWARD DX", state.plan.backward},
            {"GRADIENTS", state.plan.gradients},
            {"ACCUMULATORS", state.plan.accumulators},
            {"OPTIMIZER", state.plan.optimizer_state},
        }, muted, body, border);
        panel(display, window, gc, left, 580, left + 180, 620, raised, border_strong);
        text(display, window, gc, "Resolve selected request", left + 10, 605, body);
        panel(display, window, gc, left, 628, left + 180, 668, raised, border_strong);
        text(display, window, gc, "Inspect resume", left + 10, 653, body);
        panel(display, window, gc, left + 195, 628, left + 340, 668, raised, border_strong);
        text(display, window, gc, "Refresh status", left + 205, 653, body);
        text(display, window, gc, state.status, left + 195, 605, muted);
        panel(display, window, gc, left, 676, left + 105, 716, raised, border_strong);
        text(display, window, gc, "Check Run", left + 10, 701, body);
        panel(display, window, gc, left + 117, 676, left + 207, 716, raised, border_strong);
        text(display, window, gc, "Run", left + 127, 701, body);
        panel(display, window, gc, left + 219, 676, left + 309, 716, raised, border_strong);
        text(display, window, gc, "Cancel", left + 229, 701, body);
        text(display, window, gc, "Execution unavailable; see Help.", left + 322, 701, amber);
    } else if (state.view == View::Capabilities) {
        text(display, window, gc, "Capabilities", left, 155, bright);
        text(display, window, gc, "CATALOG != RUNTIME", right - 145, 155, amber);
        const int top = 204;
        panel(display, window, gc, left, top, right, top + 360, raised, border);
        text(display, window, gc, "CAPABILITY        STATE          EVIDENCE", left + 18, top + 24, muted);
        const char* names[] = {"backend.cuda", "backend.opencl", "precision.bf16", "precision.nvfp4", "moe.route_plan", "parallel.host_staged"};
        const char* states[] = {"cataloged", "disabled", "contract", "experimental", "contract", "contract"};
        const char* evidence[] = {"public catalog", "public catalog", "planner-tested", "contract surface", "unit-tested", "planner surface"};
        for (int index = 0; index < 6; ++index) {
            const int y = top + 55 + index * 48;
            line(display, window, gc, left + 16, y - 17, right - 16, y - 17, border);
            text(display, window, gc, names[index], left + 18, y, bright);
            const auto state_color = std::string(states[index]) == "experimental"
                ? amber
                : std::string(states[index]) == "disabled" ? muted : teal;
            text(display, window, gc, states[index], left + 190, y, state_color);
            text(display, window, gc, evidence[index], left + 315, y, state_color);
        }
    } else if (state.view == View::Evidence) {
        text(display, window, gc, "Evidence stages", left, 155, bright);
        text(display, window, gc, "NO FLATTENING", right - 120, 155, teal);
        const int track_left = left + 25;
        const int track_right = right - 25;
        line(display, window, gc, track_left, 244, track_right, 244, border_strong);
        const char* labels[] = {"Requested", "Resolved", "Accepted", "Observed", "Persisted"};
        const char* details[] = {"operator input", "native planner", "admission gate", "runtime counters", "durable receipt"};
        for (int index = 0; index < 5; ++index) {
            const int x = track_left + (track_right - track_left) * index / 4;
            fill(display, window, gc, x - 8, 236, 16, 16, index == 0 ? color(display, "#3c3528") : bg);
            XSetForeground(display, gc, index == 0 ? amber : border_strong);
            XDrawArc(display, window, gc, x - 8, 236, 16, 16, 0, 360 * 64);
            text(display, window, gc, labels[index], x - 45, 275, bright);
            text(display, window, gc, details[index], x - 50, 294, muted);
        }
        panel(display, window, gc, left, 362, right, 520, color(display, "#34281e"), color(display, "#5b4631"));
        text(display, window, gc, "CURRENT POSTURE", left + 18, 388, amber);
        text(display, window, gc, state.document.loaded ? state.document.posture : "CONTRACT-ONLY", left + 18, 428, bright);
        text(display, window, gc, state.document.loaded
                  ? "Selected document shown for review; verify schema and integrity separately."
                  : "No runtime receipt is attached. This shell is not reporting an execution.",
             left + 18, 465, body);
    } else if (state.view == View::Simulation) {
        const int work_right = right - (state.companion_visible ? ida_card.width + 18 : 0);
        const int active_stage = simulation_stage_index(state.simulation.stage);
        text(display, window, gc,
             "SIMULATION: " + std::string(simulation_status_name(state.simulation)), left, 155, bright);
        text(display, window, gc, "SYNTHETIC WALKTHROUGH / NOT A RUNTIME RECEIPT", left, 180, amber);
        draw_ida_card(display, window, gc, state, right - ida_card.width, 132, right, 132 + ida_card.height);

        panel(display, window, gc, left, 204, work_right, 288, raised, border);
        text(display, window, gc, "REQUEST -> PLAN -> ROUTE -> RUN -> SAVE", left + 16, 228, muted);
        const char* stages[] = {"REQUEST", "PLAN", "ROUTE", "RUN", "SAVE", "DONE"};
        const char* compact_stages[] = {"REQ", "PLAN", "ROUTE", "RUN", "SAVE", "DONE"};
        const int track_left = left + 28;
        const int track_right = work_right - 28;
        const int label_width = std::max(0, (track_right - track_left) / 5);
        line(display, window, gc, track_left, 258, track_right, 258, border_strong);
        for (int index = 0; index < 6; ++index) {
            const int x = track_left + (track_right - track_left) * index / 5;
            const bool reached = active_stage >= index || (state.simulation.completed && index == 5);
            const bool current = active_stage == index;
            fill(display, window, gc, x - (current ? 9 : 7), 251,
                 (current ? 18 : 14), (current ? 18 : 14),
                 reached ? (current ? amber : teal) : bg);
            const char* label = label_width < 55 ? compact_stages[index] : stages[index];
            const int label_length = static_cast<int>(std::strlen(label));
            const int measured = stage_label_font ? XTextWidth(stage_label_font, label, label_length) : label_length * 6;
            const int cell_left = std::max(left, x - label_width / 2);
            const int cell_right = std::min(work_right, x + (label_width + 1) / 2);
            // Center within the allocated column, clipping at the column edges.
            text_clipped(display, window, gc, label, x - measured / 2, 282,
                         cell_left, cell_right - cell_left, reached ? bright : muted);
        }

        panel(display, window, gc, left, 306, work_right, 574,
              color(display, "#122a2d"), color(display, "#286967"));
        text(display, window, gc, "ROUTE BOARD", left + 18, 332, teal);
        text(display, window, gc, "synthetic device-resident route plan", left + 18, 355, body);
        const int source_x = left + 34;
        const int expert_x = work_right - 120;
        const int center_y = 420;
        const int pulse = static_cast<int>(state.simulation.tick % 18);
        fill(display, window, gc, source_x, center_y - 12, 24, 24, teal);
        text(display, window, gc, "TOK", source_x + 2, center_y + 5, bg);
        line(display, window, gc, source_x + 25, center_y, expert_x - 20, center_y, border_strong);
        for (int expert = 0; expert < 4; ++expert) {
            const int y = 379 + expert * 38;
            const bool hot = (expert + pulse / 5) % 4 == 0 &&
                (state.simulation.stage == SimulationStage::Routing ||
                 state.simulation.stage == SimulationStage::Executing);
            line(display, window, gc, expert_x - 20, center_y, expert_x, y, hot ? amber : border_strong);
            fill(display, window, gc, expert_x, y - 10, 24, 20,
                 hot ? amber : color(display, "#2e4a56"));
            text(display, window, gc, "E" + std::to_string(expert), expert_x + 5, y + 5,
                 hot ? bg : bright);
            text(display, window, gc,
                 std::to_string(state.simulation.expert_load[static_cast<std::size_t>(expert)]) + " tiles",
                 expert_x + 34, y + 5, hot ? amber : body);
        }
        text(display, window, gc,
             "LAYER " + std::to_string(state.simulation.active_layer) +
                 " / TOKEN " + std::to_string(state.simulation.active_token),
             left + 18, 548, bright);
        text(display, window, gc,
             "Route is generated for this simulation only; it is not a runtime trace.",
             left + 18, 568, muted);
        text(display, window, gc,
             "STAGE " + std::string(simulation_stage_name(state.simulation.stage)) +
                 "   PROGRESS " + std::to_string(state.simulation.progress_percent) + "%",
             left, height - 90, work_right - left, body);
        text(display, window, gc,
             "MEMORY " + std::to_string(state.simulation.memory_percent) + "% synthetic",
             left, height - 70, work_right - left, body);
        text(display, window, gc, "S: start  P/Space: pause or resume  R: reset",
             left, height - 42, work_right - left, muted);
    } else {
        text(display, window, gc, "What belongs where", left, 155, bright);
        const int gap = 15;
        const int column = (right - left - gap) / 2;
        panel(display, window, gc, left, 204, left + column, 382, color(display, "#142d30"), color(display, "#286967"));
        panel(display, window, gc, left + column + gap, 204, right, 382, color(display, "#372126"), color(display, "#5b373a"));
        text(display, window, gc, "FOUNDRY", left + 18, 230, teal);
        text(display, window, gc, "Local execution product", left + 18, 263, bright);
        text(display, window, gc, "Plan resolution | memory | precision | checkpoints", left + 18, 307, body);
        text(display, window, gc, "EXTERNAL", left + column + gap + 18, 230, red);
        text(display, window, gc, "Hub authority layer", left + column + gap + 18, 263, bright);
        text(display, window, gc, "Identity | lifecycle | audit | evaluation | promotion", left + column + gap + 18, 307, body);
        panel(display, window, gc, left, 408, right, 500, raised, border);
        text(display, window, gc, "THIS NATIVE SHELL", left + 18, 435, muted);
        text(display, window, gc, "Window lifecycle and local presentation only. No socket, worker, training launch, or cloud connection.",
             left + 18, 470, body);
    }

    fill(display, window, gc, width - kRightWidth, kTopHeight, kRightWidth, height - kTopHeight, rail);
    line(display, window, gc, width - kRightWidth, kTopHeight, width - kRightWidth, height, border);
    const int rail_left = width - kRightWidth + 26;
    text(display, window, gc, "SIGNAL", rail_left, 92, deep);
    panel(display, window, gc, width - kRightWidth + 18, 113, width - 18, 278,
          color(display, "#352a1f"), color(display, "#5b4631"));
    text(display, window, gc, "POSTURE", rail_left + 10, 140, amber);
    text(display, window, gc, state.document.loaded ? state.document.posture : "CONTRACT-ONLY",
         rail_left + 10, 183, bright);
    text(display, window, gc, state.document.loaded
              ? "Local document attached; runtime evidence remains separate."
              : "No runtime evidence attached.",
         rail_left + 10, 222, muted);
    fill(display, window, gc, rail_left + 10, 260, kRightWidth - 72, 4, border_strong);
    fill(display, window, gc, rail_left + 10, 260, 55, 4, amber);
    text(display, window, gc, "ACTIVE DOCUMENT", rail_left, 322, deep);
    text(display, window, gc, state.document.loaded ? state.document.name : "None loaded", rail_left, 354, body);
    text(display, window, gc, state.document.loaded ? state.document.type : "Waiting for local JSON", rail_left, 377, muted);
    line(display, window, gc, rail_left, 418, width - 26, 418, border);
    text(display, window, gc, "LOCAL GUARANTEES", rail_left, 446, deep);
    text(display, window, gc, "OK  No network calls", rail_left, 478, body);
    text(display, window, gc, "OK  No runtime claims", rail_left, 502, body);
    text(display, window, gc, "OK  No authority expansion", rail_left, 526, body);
    text(display, window, gc, "OK  Source stays local", rail_left, 550, body);
    text(display, window, gc, "CANOPY FOUNDRY", rail_left, height - 58, deep);
    text(display, window, gc, "PUBLIC LOCAL SURFACE", rail_left, height - 36, deep);
}

}  // namespace

int run_native_shell(ShellState& state) {
    Display* display = XOpenDisplay(nullptr);
    if (display == nullptr) return 2;
    const int screen = DefaultScreen(display);
    const unsigned long background = BlackPixel(display, screen);
    const Window window = XCreateSimpleWindow(
        display, RootWindow(display, screen), 0, 0, 1280, 800, 1,
        background, background);
    // Do not clear the visible window ahead of the completed back buffer.
    XSetWindowBackgroundPixmap(display, window, None);
    XStoreName(display, window, "Canopy Foundry - Native Shell");
    XSelectInput(
        display, window,
        ExposureMask | StructureNotifyMask | ButtonReleaseMask | KeyPressMask);
    XMapWindow(display, window);
    GC gc = XCreateGC(display, window, 0, nullptr);
    stage_label_font = XQueryFont(display, XGContextFromGC(gc));
    XSetGraphicsExposures(display, gc, False);
    bool running = true;
    int width = 1280;
    int height = 800;
    int buffer_width = 0;
    int buffer_height = 0;
    Pixmap buffer = None;
    bool needs_redraw = true;
    using Clock = std::chrono::steady_clock;
    auto next_tick = Clock::now() + std::chrono::milliseconds(100);
    while (running) {
        while (running && XPending(display) > 0) {
            XEvent event{};
            XNextEvent(display, &event);
            if (event.type == Expose || event.type == ConfigureNotify ||
                event.type == ButtonRelease || event.type == KeyPress) needs_redraw = true;
            if (event.type == ConfigureNotify) {
                width = event.xconfigure.width;
                height = event.xconfigure.height;
            } else if (event.type == ButtonRelease) {
                if (state.guide_visible) {
                    const int left = kLeftWidth + 45;
                    if (event.xbutton.y >= 510 && event.xbutton.y < 548 && event.xbutton.x >= left && event.xbutton.x < left + 145)
                        open_guide_document(state, GuideDocument::Starter);
                    else if (event.xbutton.y >= 510 && event.xbutton.y < 548 && event.xbutton.x >= left + 155 && event.xbutton.x < left + 300)
                        open_guide_document(state, GuideDocument::Foundry);
                    else if (event.xbutton.x >= left && event.xbutton.x < left + 145 &&
                        event.xbutton.y >= 580 && event.xbutton.y < 620) close_guide(state);
                    continue;
                }
                if (event.xbutton.x < kLeftWidth && event.xbutton.y >= 562 && event.xbutton.y < 595) {
                    state.guide_visible = true;
                } else if (event.xbutton.x < kLeftWidth) {
                    const int y = event.xbutton.y;
                    if (y >= 112 && y < 147) state.view = View::Overview;
                    else if (y >= 156 && y < 191) state.view = View::Plan;
                    else if (y >= 200 && y < 235) state.view = View::Capabilities;
                    else if (y >= 244 && y < 279) state.view = View::Evidence;
                else if (y >= 343 && y < 378) state.view = View::Boundaries;
                else if (y >= 426 && y < 461) state.view = View::Simulation;
                else if (y >= 480 && y < 518) switch_workspace_mode(state,
                    state.mode == WorkspaceMode::Ida ? WorkspaceMode::TrainBrain : WorkspaceMode::Ida);
                else if (y >= 525 && y < 552) state.companion_visible = !state.companion_visible;
                } else if (state.view == View::Overview &&
                           event.xbutton.x >= kLeftWidth + 45 &&
                       event.xbutton.x < kLeftWidth + 190 &&
                       event.xbutton.y >= 302 &&
                       event.xbutton.y < 350) {
                state.status = "Linux shell: pass --document PATH to inspect a JSON document.";
            } else if (state.view == View::Plan &&
                       event.xbutton.x >= kLeftWidth + 45 &&
                       event.xbutton.x < kLeftWidth + 45 + 180 &&
                       event.xbutton.y >= 580 && event.xbutton.y < 620) {
                resolve_selected_request(state);
            } else if (state.view == View::Plan &&
                       event.xbutton.x >= kLeftWidth + 45 &&
                       event.xbutton.x < kLeftWidth + 45 + 180 &&
                       event.xbutton.y >= 628 && event.xbutton.y < 668) {
                inspect_selected_resume(state);
            } else if (state.view == View::Plan &&
                       event.xbutton.x >= kLeftWidth + 45 + 195 &&
                       event.xbutton.x < kLeftWidth + 45 + 340 &&
                       event.xbutton.y >= 628 && event.xbutton.y < 668) {
                refresh_selected_run(state);
            } else if (state.view == View::Plan &&
                       event.xbutton.x >= kLeftWidth + 45 &&
                       event.xbutton.x < kLeftWidth + 45 + 105 &&
                       event.xbutton.y >= 676 && event.xbutton.y < 716) {
                check_selected_run(state);
            } else if (state.view == View::Plan &&
                       event.xbutton.x >= kLeftWidth + 45 + 117 &&
                       event.xbutton.x < kLeftWidth + 45 + 207 &&
                       event.xbutton.y >= 676 && event.xbutton.y < 716) {
                start_selected_run(state);
            } else if (state.view == View::Plan &&
                       event.xbutton.x >= kLeftWidth + 45 + 219 &&
                       event.xbutton.x < kLeftWidth + 45 + 309 &&
                       event.xbutton.y >= 676 && event.xbutton.y < 716) {
                cancel_selected_run(state);
            } else if (state.view == View::Overview &&
                       event.xbutton.x >= kLeftWidth + 45 + 157 &&
                       event.xbutton.x < kLeftWidth + 45 + 267 &&
                       event.xbutton.y >= 302 && event.xbutton.y < 350) {
                switch_workspace_mode(state, WorkspaceMode::TrainBrain);
            } else if (state.view == View::Overview &&
                       event.xbutton.x >= kLeftWidth + 45 + 279 &&
                       event.xbutton.x < kLeftWidth + 45 + 407 &&
                       event.xbutton.y >= 302 &&
                       event.xbutton.y < 350) {
                start_simulation(state.simulation);
                state.view = View::Simulation;
                }
            } else if (event.type == KeyPress) {
                const KeySym key = XLookupKeysym(&event.xkey, 0);
                if (key == XK_h || key == XK_H) {
                    if (state.guide_visible) close_guide(state);
                    else state.guide_visible = true;
                } else if (state.guide_visible) {
                    if (key == XK_1 || key == XK_2) open_guide_document(state, key == XK_1 ? GuideDocument::Starter : GuideDocument::Foundry);
                    else if (key == XK_Escape) close_guide(state);
                    continue;
                } else if (key == XK_Escape || key == XK_q) {
                    running = false;
                } else if (key >= XK_1 && key <= XK_5) {
                    state.view = static_cast<View>(key - XK_1);
                } else if (key == XK_s || key == XK_S) {
                    start_simulation(state.simulation);
                    state.view = View::Simulation;
                } else if (key == XK_p || key == XK_P || key == XK_space) {
                    toggle_simulation_pause(state.simulation);
                } else if (key == XK_r || key == XK_R) {
                    reset_simulation(state.simulation);
                }
            }
        }
        if (!running) break;
        const auto now = Clock::now();
        if (now >= next_tick) {
            const auto previous_animation = state.animation_tick++;
            const auto previous_simulation = state.simulation.tick;
            advance_simulation(state.simulation);
            const bool show_simulation = !state.guide_visible && state.view == View::Simulation;
            const bool show_companion = state.companion_visible &&
                (state.guide_visible || state.view == View::Overview || show_simulation);
            needs_redraw = needs_redraw ||
                (show_simulation && previous_simulation != state.simulation.tick) ||
                (show_companion && sprite_visual_changes(state.ida_sprite, previous_animation, state.animation_tick));
            next_tick = now + std::chrono::milliseconds(100);
        }
        if (needs_redraw && width > 0 && height > 0) {
            if (buffer == None || buffer_width != width || buffer_height != height) {
                if (buffer != None) XFreePixmap(display, buffer);
                buffer = XCreatePixmap(display, window, static_cast<unsigned int>(width),
                                       static_cast<unsigned int>(height), DefaultDepth(display, screen));
                buffer_width = width;
                buffer_height = height;
            }
            if (buffer != None) {
                render(display, buffer, gc, state, width, height);
                XCopyArea(display, buffer, window, gc, 0, 0,
                          static_cast<unsigned int>(width), static_cast<unsigned int>(height), 0, 0);
            } else {
                render(display, window, gc, state, width, height);
            }
            XFlush(display);
            needs_redraw = false;
        }

        fd_set descriptors;
        FD_ZERO(&descriptors);
        const int connection = ConnectionNumber(display);
        FD_SET(connection, &descriptors);
        const auto remaining = std::max(Clock::duration::zero(), next_tick - Clock::now());
        const auto micros = std::chrono::duration_cast<std::chrono::microseconds>(remaining).count();
        timeval timeout{};
        timeout.tv_sec = static_cast<long>(micros / 1000000);
        timeout.tv_usec = static_cast<long>(micros % 1000000);
        select(connection + 1, &descriptors, nullptr, nullptr, &timeout);
    }
    if (buffer != None) XFreePixmap(display, buffer);
    if (stage_label_font) XFreeFontInfo(nullptr, stage_label_font, 1);
    stage_label_font = nullptr;
    XFreeGC(display, gc);
    XDestroyWindow(display, window);
    XCloseDisplay(display);
    named_colors.clear();
    sprite_colors.clear();
    return 0;
}

}  // namespace canopy_foundry::shell

#endif
