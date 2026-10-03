#ifdef _WIN32

#include "canopy_foundry/shell/platform.hpp"

#include <windows.h>
#include <commdlg.h>

#include <algorithm>
#include <string>
#include <utility>
#include <vector>

namespace canopy_foundry::shell {
namespace {

constexpr int kTopHeight = 64;
constexpr int kLeftWidth = 204;
constexpr int kRightWidth = 316;
constexpr UINT_PTR kAnimationTimer = 1;

ShellState* g_state = nullptr;

std::wstring wide(const std::string& value) {
    if (value.empty()) return {};
    const int required = MultiByteToWideChar(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        nullptr, 0);
    if (required <= 0) return {};
    std::wstring result(static_cast<std::size_t>(required), L'\0');
    MultiByteToWideChar(
        CP_UTF8, 0, value.data(), static_cast<int>(value.size()),
        result.data(), required);
    return result;
}

COLORREF color(int red, int green, int blue) {
    return RGB(red, green, blue);
}

void fill(HDC dc, const RECT& rect, COLORREF value) {
    HBRUSH brush = CreateSolidBrush(value);
    FillRect(dc, &rect, brush);
    DeleteObject(brush);
}

void outline(HDC dc, const RECT& rect, COLORREF value, int radius = 8) {
    HPEN pen = CreatePen(PS_SOLID, 1, value);
    HGDIOBJ old_pen = SelectObject(dc, pen);
    HGDIOBJ old_brush = SelectObject(dc, GetStockObject(HOLLOW_BRUSH));
    RoundRect(dc, rect.left, rect.top, rect.right, rect.bottom, radius, radius);
    SelectObject(dc, old_brush);
    SelectObject(dc, old_pen);
    DeleteObject(pen);
}

void line(HDC dc, int x1, int y1, int x2, int y2, COLORREF value) {
    HPEN pen = CreatePen(PS_SOLID, 1, value);
    HGDIOBJ old_pen = SelectObject(dc, pen);
    MoveToEx(dc, x1, y1, nullptr);
    LineTo(dc, x2, y2);
    SelectObject(dc, old_pen);
    DeleteObject(pen);
}

void draw_text(
    HDC dc,
    const std::string& value,
    int x,
    int y,
    int width,
    int height,
    int size,
    COLORREF text_color,
    bool bold = false,
    UINT format = DT_LEFT | DT_TOP
) {
    const std::wstring text_value = wide(value);
    RECT rect{x, y, x + width, y + height};
    HFONT font = CreateFontW(
        -size, 0, 0, 0, bold ? FW_SEMIBOLD : FW_NORMAL,
        FALSE, FALSE, FALSE, DEFAULT_CHARSET, OUT_DEFAULT_PRECIS,
        CLIP_DEFAULT_PRECIS, CLEARTYPE_QUALITY, DEFAULT_PITCH,
        L"Segoe UI");
    HGDIOBJ old_font = SelectObject(dc, font);
    SetBkMode(dc, TRANSPARENT);
    SetTextColor(dc, text_color);
    DrawTextW(
        dc, text_value.c_str(), static_cast<int>(text_value.size()),
        &rect, format | DT_NOPREFIX);
    SelectObject(dc, old_font);
    DeleteObject(font);
}

void panel(HDC dc, int left, int top, int right, int bottom, COLORREF fill_color, COLORREF border_color) {
    const RECT rect{left, top, right, bottom};
    fill(dc, rect, fill_color);
    outline(dc, rect, border_color);
}

COLORREF sprite_color(const SpriteColor& value) {
    return color(value.red, value.green, value.blue);
}

void draw_sprite(
    HDC dc,
    const SpriteFrame& frame,
    int left,
    int top,
    int subpixel,
    COLORREF fallback
) {
    if (frame.width <= 0 || frame.height <= 0 || subpixel <= 0) return;
    const int cell_width = subpixel * 2;
    for (int y = 0; y < frame.height; ++y) {
        for (int x = 0; x < frame.width; ++x) {
            const auto& cell = frame.cells[static_cast<std::size_t>(y * frame.width + x)];
            for (int part_y = 0; part_y < 2; ++part_y) {
                for (int part_x = 0; part_x < 2; ++part_x) {
                    const int bit = 1 << (part_y * 2 + part_x);
                    const bool foreground = (cell.mask & bit) != 0;
                    if (!foreground && !cell.background.opaque) continue;
                    const COLORREF value = foreground
                        ? sprite_color(cell.foreground)
                        : sprite_color(cell.background);
                    fill(
                        dc,
                        RECT{
                            left + x * cell_width + part_x * subpixel,
                            top + y * cell_width + part_y * subpixel,
                            left + x * cell_width + (part_x + 1) * subpixel,
                            top + y * cell_width + (part_y + 1) * subpixel,
                        },
                        value == color(0, 0, 0) && !foreground ? fallback : value);
                }
            }
        }
    }
}

void draw_ida_card(HDC dc, int left, int top, int right, int bottom) {
    const int pulse = 40 + ((static_cast<int>(g_state->animation_tick % 8) * 7));
    panel(dc, left, top, right, bottom, color(16, 30, 42), color(40, 105, 103));
    draw_text(dc, "IDA CLI / LOCAL PRESENCE", left + 16, top + 14, right - left - 32, 18,
              9, color(85 + pulse / 8, 213, 197), true);
    const auto* frame = sprite_frame_at(g_state->ida_sprite, g_state->animation_tick);
    if (frame == nullptr) {
        draw_text(dc, "Presentation atlas unavailable.", left + 16, top + 72,
                  right - left - 32, 22, 10, color(185, 197, 209));
        draw_text(dc, "No execution claim is implied.", left + 16, top + 104,
                  right - left - 32, 22, 9, color(116, 130, 148));
        return;
    }
    const int available_width = std::max(1, right - left - 24);
    const int available_height = std::max(1, bottom - top - 68);
    const int subpixel = std::clamp(
        std::min(available_width / std::max(1, frame->width * 2),
                 available_height / std::max(1, frame->height * 2)),
        2,
        4);
    const int sprite_width = frame->width * subpixel * 2;
    const int sprite_left = left + (right - left - sprite_width) / 2;
    const int sprite_top = top + 40 + sprite_breathe_offset(g_state->animation_tick);
    draw_sprite(dc, *frame, sprite_left, sprite_top, subpixel, color(16, 30, 42));
    const int frame_number = static_cast<int>(
        (g_state->animation_tick / std::max(1, g_state->ida_sprite.frame_hold_ticks)) %
        std::max<std::size_t>(1, g_state->ida_sprite.frames.size())) + 1;
    draw_text(
        dc,
        "FRAME " + std::to_string(frame_number) + "/" +
            std::to_string(g_state->ida_sprite.frames.size()) + "  ·  PRESENTATION ONLY",
        left + 12,
        bottom - 25,
        right - left - 24,
        16,
        8,
        color(116, 130, 148),
        false,
        DT_CENTER);
}

void button(HDC dc, const std::string& label, int left, int top, int width, bool active = false) {
    panel(
        dc, left, top, left + width, top + 36,
        active ? color(40, 110, 108) : color(20, 29, 40),
        active ? color(85, 213, 197) : color(43, 58, 77));
    draw_text(dc, label, left + 12, top + 9, width - 24, 18, 11,
              active ? color(231, 237, 243) : color(185, 197, 209), true);
}

void draw_header(HDC dc, int width) {
    fill(dc, RECT{0, 0, width, kTopHeight}, color(10, 14, 20));
    line(dc, 0, kTopHeight - 1, width, kTopHeight - 1, color(30, 42, 56));
    fill(dc, RECT{20, 18, 49, 47}, color(20, 53, 57));
    outline(dc, RECT{20, 18, 49, 47}, color(85, 213, 197), 7);
    fill(dc, RECT{27, 31, 32, 42}, color(85, 213, 197));
    fill(dc, RECT{35, 26, 40, 42}, color(85, 213, 197));
    fill(dc, RECT{43, 22, 48, 42}, color(85, 213, 197));
    draw_text(dc, "CANOPY FOUNDRY", 61, 17, 260, 18, 12, color(231, 237, 243), true);
    draw_text(dc, "NATIVE LOCAL SHELL", 61, 36, 260, 14, 8, color(116, 130, 148));
    const int pulse = 70 + static_cast<int>((g_state->animation_tick % 6) * 8);
    fill(dc, RECT{width - 144, 30, width - 137, 37}, color(55, pulse, 185));
    draw_text(dc, "LOCAL ONLY", width - 128, 27, 77, 16, 9, color(116, 130, 148), true);
    line(dc, width - 44, 27, width - 44, 41, color(43, 58, 77));
    draw_text(dc, "NATIVE", width - 34, 27, 30, 16, 8, color(185, 197, 209), true, DT_RIGHT);
}

void draw_nav(HDC dc, int height) {
    fill(dc, RECT{0, kTopHeight, kLeftWidth, height}, color(13, 18, 26));
    line(dc, kLeftWidth - 1, kTopHeight, kLeftWidth - 1, height, color(30, 42, 56));
    draw_text(dc, "WORKSPACE", 22, 88, 150, 18, 9, color(82, 96, 113), true);

    const View views[] = {
        View::Overview, View::Plan, View::Capabilities, View::Evidence};
    const char* labels[] = {"Overview", "Plan lens", "Capabilities", "Evidence"};
    for (int index = 0; index < 4; ++index) {
        const int top = 112 + index * 44;
        const bool active = g_state->view == views[index];
        if (active) {
            panel(dc, 13, top, kLeftWidth - 14, top + 35,
                  color(19, 43, 47), color(40, 105, 103));
            fill(dc, RECT{13, top, 16, top + 35}, color(85, 213, 197));
        }
        draw_text(dc, labels[index], 38, top + 9, 112, 18, 11,
                  active ? color(231, 237, 243) : color(116, 130, 148), active);
        draw_text(dc, "0" + std::to_string(index + 1), 166, top + 10, 23, 16, 8,
                  color(82, 96, 113), false, DT_RIGHT);
    }
    draw_text(dc, "BOUNDARY", 22, 319, 150, 18, 9, color(82, 96, 113), true);
    if (g_state->view == View::Boundaries) {
        panel(dc, 13, 343, kLeftWidth - 14, 378,
              color(19, 43, 47), color(40, 105, 103));
        fill(dc, RECT{13, 343, 16, 378}, color(85, 213, 197));
    }
    draw_text(dc, "Ownership", 38, 352, 112, 18, 11,
              g_state->view == View::Boundaries
                  ? color(231, 237, 243)
                  : color(116, 130, 148),
              g_state->view == View::Boundaries);
    draw_text(dc, "05", 166, 353, 23, 16, 8, color(82, 96, 113), false, DT_RIGHT);

    draw_text(dc, "PLAYGROUND", 22, 402, 150, 18, 9, color(82, 96, 113), true);
    if (g_state->view == View::Simulation) {
        panel(dc, 13, 426, kLeftWidth - 14, 461,
              color(19, 43, 47), color(40, 105, 103));
        fill(dc, RECT{13, 426, 16, 461}, color(85, 213, 197));
    }
    draw_text(dc, "Simulation", 38, 435, 112, 18, 11,
              g_state->view == View::Simulation
                  ? color(231, 237, 243)
                  : color(116, 130, 148),
              g_state->view == View::Simulation);
    draw_text(dc, "S", 166, 436, 23, 16, 8, color(82, 96, 113), false, DT_RIGHT);

    line(dc, 22, height - 109, kLeftWidth - 22, height - 109, color(30, 42, 56));
    draw_text(dc, "STANDALONE SURFACE", 22, height - 89, 160, 16, 9, color(85, 213, 197), true);
    draw_text(dc, "Local contracts only.", 22, height - 65, 155, 18, 10, color(116, 130, 148));
    draw_text(dc, "planner foundation / v1", 22, height - 37, 160, 16, 8, color(82, 96, 113));
}

void draw_facts(HDC dc, int left, int top, int width, const std::vector<std::pair<std::string, std::string>>& facts) {
    line(dc, left, top, left + width, top, color(30, 42, 56));
    int y = top + 12;
    for (const auto& fact : facts) {
        draw_text(dc, fact.first, left, y, width / 2, 18, 10, color(116, 130, 148));
        draw_text(dc, fact.second, left + width / 2, y, width / 2, 18, 10, color(185, 197, 209), false, DT_RIGHT);
        line(dc, left, y + 24, left + width, y + 24, color(24, 34, 46));
        y += 36;
    }
}

void draw_overview(HDC dc, int left, int right, int height) {
    draw_text(dc, "CANOPY FOUNDRY / OVERVIEW", left, 92, right - left, 18, 9, color(116, 130, 148), true);
    draw_text(dc, "LOCAL EXECUTION / INSPECTABLE BY DEFAULT", left, 132, right - left, 18, 10, color(85, 213, 197), true);
    draw_text(dc, "Make the execution contract visible.", left, 161, 410, 92, 34, color(231, 237, 243), false, DT_WORDBREAK);
    draw_text(dc,
              "A native shell for local Foundry contracts. It shows the current document and evidence posture without launching training or contacting the Hub.",
              left, 265, 410, 68, 13, color(185, 197, 209), false, DT_WORDBREAK);
    draw_ida_card(dc, right - 224, 132, right, 340);
    button(dc, "Load local JSON", left, 352, 145, true);
    button(dc, "Plan lens", left + 157, 352, 110);
    button(dc, "Run simulation", left + 279, 352, 128);
    draw_text(dc, g_state->status, left, 408, 530, 40, 10, color(116, 130, 148), false, DT_WORDBREAK);

    const int card_top = std::max(470, height - 220);
    panel(dc, left, card_top, left + 330, height - 24, color(18, 40, 43), color(40, 105, 103));
    draw_text(dc, "CURRENT LENS", left + 18, card_top + 18, 180, 16, 9, color(85, 213, 197), true);
    draw_text(dc, "Request -> plan", left + 18, card_top + 47, 260, 24, 17, color(231, 237, 243), true);
    draw_facts(dc, left + 18, card_top + 88, 294, {
        {"SOURCE", g_state->document.loaded ? g_state->document.name : "None loaded"},
        {"DOCUMENT", g_state->document.loaded ? g_state->document.type : "no document"},
        {"NETWORK", "disabled"},
    });
}

void draw_plan(HDC dc, int left, int right) {
    draw_text(dc, "CANOPY FOUNDRY / PLAN LENS", left, 92, right - left, 18, 9, color(116, 130, 148), true);
    draw_text(dc, "Plan lens", left, 122, 500, 48, 36, color(231, 237, 243), false);
    draw_text(dc, "DISPLAYED CONTRACT IS NOT PROOF OF EXECUTION", left + 230, 140, 300, 18, 9, color(240, 174, 104), true);
    const int column = std::max(270, (right - left - 30) / 3);
    const int top = 204;
    for (int index = 0; index < 3; ++index) {
        panel(dc, left + index * (column + 15), top, left + index * (column + 15) + column, 560,
              index == 1 ? color(20, 45, 48) : color(20, 28, 39),
              index == 1 ? color(40, 105, 103) : color(30, 42, 56));
    }
    draw_text(dc, "INPUT DOCUMENT", left + 18, top + 18, column - 36, 16, 9, color(116, 130, 148), true);
    draw_text(dc, g_state->document.loaded ? g_state->document.name : "No local JSON loaded",
              left + 18, top + 45, column - 36, 22, 14, color(231, 237, 243), true, DT_WORDBREAK);
    draw_facts(dc, left + 18, top + 92, column - 36, {
        {"SCHEMA", g_state->document.schema},
        {"BACKEND", g_state->document.backend},
        {"ARCH", g_state->document.required_arch},
        {"MODEL", g_state->document.model_contract},
        {"OPTIMIZER", g_state->document.optimizer},
    });

    const int middle = left + column + 15;
    draw_text(dc, "RESOLVED SHAPE", middle + 18, top + 18, column - 36, 16, 9, color(85, 213, 197), true);
    draw_text(dc, "ExecutionPlan", middle + 18, top + 45, column - 36, 24, 17, color(231, 237, 243), true);
    draw_facts(dc, middle + 18, top + 92, column - 36, {
        {"PLANNER", g_state->document.planner_revision},
        {"PARALLEL", g_state->document.parallel_mode},
        {"TRANSPORT", g_state->document.transport},
        {"OWNER", "native planner"},
        {"RUNTIME", "verify separately"},
    });

    const int last = middle + column + 15;
    draw_text(dc, "PRECISION ROLES", last + 18, top + 18, column - 36, 16, 9, color(116, 130, 148), true);
    draw_text(dc, "Representation map", last + 18, top + 45, column - 36, 24, 15, color(231, 237, 243), true);
    draw_facts(dc, last + 18, top + 92, column - 36, {
        {"MASTER", "explicit role"},
        {"FORWARD", "execution view"},
        {"BACKWARD DX", "execution view"},
        {"GRADIENTS", "separate"},
        {"ACCUMULATORS", "separate"},
    });
}

void draw_capabilities(HDC dc, int left, int right) {
    draw_text(dc, "CANOPY FOUNDRY / CAPABILITIES", left, 92, right - left, 18, 9, color(116, 130, 148), true);
    draw_text(dc, "Capabilities", left, 122, 500, 48, 36, color(231, 237, 243), false);
    draw_text(dc, "CATALOG != RUNTIME", right - 150, 143, 150, 16, 9, color(240, 174, 104), true, DT_RIGHT);
    const int top = 204;
    const int width = right - left;
    panel(dc, left, top, right, top + 382, color(20, 28, 39), color(30, 42, 56));
    draw_text(dc, "CAPABILITY", left + 18, top + 15, width / 3, 16, 9, color(116, 130, 148), true);
    draw_text(dc, "STATE", left + width / 3, top + 15, 110, 16, 9, color(116, 130, 148), true);
    draw_text(dc, "EVIDENCE", left + width / 3 + 110, top + 15, 135, 16, 9, color(116, 130, 148), true);
    const char* names[] = {"backend.cuda", "backend.opencl", "precision.bf16", "precision.nvfp4", "moe.route_plan", "parallel.host_staged"};
    const char* states[] = {"cataloged", "disabled", "contract", "experimental", "contract", "contract"};
    const char* evidence[] = {"public catalog", "public catalog", "planner-tested", "contract surface", "unit-tested", "planner surface"};
    for (int index = 0; index < 6; ++index) {
        const int y = top + 49 + index * 53;
        line(dc, left + 16, y - 11, right - 16, y - 11, color(24, 34, 46));
        draw_text(dc, names[index], left + 18, y, width / 3, 18, 10, color(231, 237, 243), false);
        const COLORREF state_color = std::string(states[index]) == "experimental"
            ? color(240, 174, 104)
            : std::string(states[index]) == "disabled"
                ? color(116, 130, 148)
                : color(85, 213, 197);
        draw_text(dc, states[index], left + width / 3, y, 110, 18, 10, state_color, true);
        draw_text(dc, evidence[index], left + width / 3 + 110, y, 145, 18, 10, state_color);
    }
}

void draw_evidence(HDC dc, int left, int right) {
    draw_text(dc, "CANOPY FOUNDRY / EVIDENCE", left, 92, right - left, 18, 9, color(116, 130, 148), true);
    draw_text(dc, "Evidence stages", left, 122, 500, 48, 36, color(231, 237, 243), false);
    draw_text(dc, "NO FLATTENING", right - 120, 143, 120, 16, 9, color(85, 213, 197), true, DT_RIGHT);
    const char* labels[] = {"Requested", "Resolved", "Accepted", "Observed", "Persisted"};
    const char* details[] = {"operator input", "native planner", "admission gate", "runtime counters", "durable receipt"};
    const int track_left = left + 24;
    const int track_right = right - 24;
    line(dc, track_left, 248, track_right, 248, color(43, 58, 77));
    for (int index = 0; index < 5; ++index) {
        const int x = track_left + (track_right - track_left) * index / 4;
        const bool current = index == 0;
        fill(dc, RECT{x - 9, 239, x + 9, 257}, current ? color(60, 53, 40) : color(10, 14, 20));
        outline(dc, RECT{x - 9, 239, x + 9, 257}, current ? color(240, 174, 104) : color(43, 58, 77), 9);
        draw_text(dc, labels[index], x - 55, 275, 110, 18, 10, color(231, 237, 243), true, DT_CENTER);
        draw_text(dc, details[index], x - 65, 297, 130, 28, 9, color(116, 130, 148), false, DT_CENTER | DT_WORDBREAK);
    }
    panel(dc, left, 370, right, 560, color(52, 40, 30), color(91, 70, 49));
    draw_text(dc, "CURRENT POSTURE", left + 18, 390, 220, 16, 9, color(240, 174, 104), true);
    draw_text(dc, g_state->document.loaded ? g_state->document.posture : "CONTRACT-ONLY",
              left + 18, 421, 310, 28, 21, color(231, 237, 243), true);
    draw_text(dc, g_state->document.loaded
                  ? "The selected document is shown for review. Verify schema and integrity before treating it as evidence."
                  : "No runtime receipt is attached. This shell is not reporting an execution.",
              left + 18, 463, right - left - 36, 48, 12, color(185, 197, 209), false, DT_WORDBREAK);
}

void draw_boundaries(HDC dc, int left, int right) {
    draw_text(dc, "CANOPY FOUNDRY / OWNERSHIP", left, 92, right - left, 18, 9, color(116, 130, 148), true);
    draw_text(dc, "What belongs where", left, 122, 500, 48, 36, color(231, 237, 243), false);
    const int gap = 15;
    const int column = (right - left - gap) / 2;
    panel(dc, left, 204, left + column, 390, color(20, 45, 48), color(40, 105, 103));
    panel(dc, left + column + gap, 204, right, 390, color(55, 33, 38), color(91, 55, 58));
    draw_text(dc, "FOUNDRY", left + 18, 224, column - 36, 16, 9, color(85, 213, 197), true);
    draw_text(dc, "Local execution product", left + 18, 254, column - 36, 24, 16, color(231, 237, 243), true);
    draw_text(dc, "Plan resolution\nMemory and precision\nKernels and routing\nCheckpoint integrity",
              left + 18, 299, column - 36, 70, 12, color(185, 197, 209), false, DT_WORDBREAK);
    draw_text(dc, "EXTERNAL", left + column + gap + 18, 224, column - 36, 16, 9, color(237, 120, 117), true);
    draw_text(dc, "Hub authority layer", left + column + gap + 18, 254, column - 36, 24, 16, color(231, 237, 243), true);
    draw_text(dc, "Identity and organization scope\nLifecycle and quotas\nEvaluation and audit\nPromotion and public projections",
              left + column + gap + 18, 299, column - 36, 70, 12, color(185, 197, 209), false, DT_WORDBREAK);
    panel(dc, left, 416, right, 535, color(20, 28, 39), color(30, 42, 56));
    draw_text(dc, "THIS NATIVE SHELL", left + 18, 436, 220, 16, 9, color(116, 130, 148), true);
    draw_text(dc, "Window lifecycle and local presentation only. No socket, worker, training launch, or cloud connection.",
              left + 18, 465, right - left - 36, 40, 12, color(185, 197, 209), false, DT_WORDBREAK);
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

void draw_simulation(HDC dc, int left, int right, int height) {
    const auto bg = color(10, 14, 20);
    const auto raised = color(20, 28, 39);
    const auto border = color(30, 42, 56);
    const auto border_strong = color(43, 58, 77);
    const auto body = color(185, 197, 209);
    const auto bright = color(231, 237, 243);
    const auto muted = color(116, 130, 148);
    const auto teal = color(85, 213, 197);
    const auto amber = color(240, 174, 104);
    const int work_right = right - 242;
    const int active_stage = simulation_stage_index(g_state->simulation.stage);

    draw_text(dc, "CANOPY FOUNDRY / EXECUTION THEATER", left, 92, right - left, 18, 9, muted, true);
    draw_text(dc, "SIMULATION / SYNTHETIC WALKTHROUGH", left, 122, work_right - left, 18, 10, amber, true);
    draw_ida_card(dc, right - 224, 132, right, 340);

    panel(dc, left, 132, work_right, 218, raised, border);
    draw_text(dc, "REQUEST -> PLAN -> ROUTE -> RUN -> SAVE", left + 16, 150,
              work_right - left - 32, 16, 9, muted, true);
    const char* stages[] = {"REQUEST", "PLAN", "ROUTE", "RUN", "SAVE", "DONE"};
    const int track_left = left + 28;
    const int track_right = work_right - 28;
    line(dc, track_left, 184, track_right, 184, border_strong);
    for (int index = 0; index < 6; ++index) {
        const int x = track_left + (track_right - track_left) * index / 5;
        const bool reached = active_stage >= index || (g_state->simulation.completed && index == 5);
        const bool current = active_stage == index;
        fill(dc, RECT{x - (current ? 9 : 7), 177 - (current ? 2 : 0),
                      x + (current ? 9 : 7), 191 + (current ? 2 : 0)},
             reached ? (current ? amber : teal) : bg);
        outline(dc, RECT{x - 9, 175, x + 9, 193}, reached ? (current ? amber : teal) : border_strong, 9);
        draw_text(dc, stages[index], x - 42, 198, 84, 14, 8,
                  reached ? bright : muted, true, DT_CENTER);
    }

    panel(dc, left, 236, work_right, 505, color(18, 40, 43), color(40, 105, 103));
    draw_text(dc, "ROUTE BOARD", left + 18, 256, 170, 16, 9, teal, true);
    draw_text(dc, "synthetic device-resident route plan", left + 18, 278,
              work_right - left - 36, 16, 10, body);
    const int source_x = left + 34;
    const int expert_x = work_right - 120;
    const int center_y = 340;
    const int pulse = static_cast<int>(g_state->simulation.tick % 18);
    fill(dc, RECT{source_x, center_y - 12, source_x + 24, center_y + 12}, teal);
    draw_text(dc, "TOK", source_x - 1, center_y - 6, 26, 14, 8, bg, true, DT_CENTER);
    line(dc, source_x + 25, center_y, expert_x - 20, center_y, border_strong);
    for (int expert = 0; expert < 4; ++expert) {
        const int y = 302 + expert * 38;
        const bool hot = (expert + pulse / 5) % 4 == 0 &&
            (g_state->simulation.stage == SimulationStage::Routing ||
             g_state->simulation.stage == SimulationStage::Executing);
        line(dc, expert_x - 20, center_y, expert_x, y, hot ? amber : border_strong);
        fill(dc, RECT{expert_x, y - 10, expert_x + 24, y + 10}, hot ? amber : color(46, 74, 86));
        draw_text(dc, "E" + std::to_string(expert), expert_x, y - 6, 24, 14, 8,
                  hot ? bg : bright, true, DT_CENTER);
        draw_text(dc, std::to_string(g_state->simulation.expert_load[static_cast<std::size_t>(expert)]) +
                      " tiles", expert_x + 34, y - 7, 76, 16, 9, hot ? amber : body);
    }
    draw_text(dc, "LAYER " + std::to_string(g_state->simulation.active_layer) +
                  "  /  TOKEN " + std::to_string(g_state->simulation.active_token),
              left + 18, 454, work_right - left - 36, 16, 10, bright, true);
    draw_text(dc, "The route is generated for this simulation only; it is not a runtime trace.",
              left + 18, 478, work_right - left - 36, 16, 9, muted);

    const int footer_top = std::min(540, std::max(520, height - 166));
    panel(dc, left, footer_top, work_right, height - 24, raised, border);
    draw_text(dc, "SIMULATED EVENT TAPE", left + 18, footer_top + 22,
              work_right - left - 36, 16, 9, amber, true);
    draw_text(dc, "STAGE  " + std::string(simulation_stage_name(g_state->simulation.stage)),
              left + 18, footer_top + 50, 180, 18, 12, bright, true);
    draw_text(dc, "PROGRESS  " + std::to_string(g_state->simulation.progress_percent) + "%",
              left + 210, footer_top + 50, 150, 18, 10, teal, true);
    draw_text(dc, "MEMORY  " + std::to_string(g_state->simulation.memory_percent) + "% synthetic pressure",
              left + 18, footer_top + 79, work_right - left - 36, 16, 9, body);
    fill(dc, RECT{left + 18, footer_top + 103, work_right - 18, footer_top + 108}, border_strong);
    fill(dc, RECT{left + 18, footer_top + 103,
                  left + 18 + (work_right - left - 36) * g_state->simulation.memory_percent / 100,
                  footer_top + 108}, amber);
    draw_text(dc, "S start/restart   P pause   R reset   SPACE pause/resume",
              left + 18, height - 43, work_right - left - 36, 16, 9, muted);
}

void draw_center(HDC dc, int width, int height) {
    const int left = kLeftWidth + 45;
    const int right = width - kRightWidth - 28;
    switch (g_state->view) {
        case View::Overview:
            draw_overview(dc, left, right, height);
            break;
        case View::Plan:
            draw_plan(dc, left, right);
            break;
        case View::Capabilities:
            draw_capabilities(dc, left, right);
            break;
        case View::Evidence:
            draw_evidence(dc, left, right);
            break;
        case View::Boundaries:
            draw_boundaries(dc, left, right);
            break;
        case View::Simulation:
            draw_simulation(dc, left, right, height);
            break;
    }
}

void draw_right_rail(HDC dc, int width, int height) {
    const int left = width - kRightWidth;
    fill(dc, RECT{left, kTopHeight, width, height}, color(13, 18, 26));
    line(dc, left, kTopHeight, left, height, color(30, 42, 56));
    draw_text(dc, "SIGNAL", left + 26, 88, 180, 16, 9, color(82, 96, 113), true);
    panel(dc, left + 18, 113, width - 18, 278, color(53, 42, 31), color(91, 70, 49));
    draw_text(dc, "POSTURE", left + 36, 134, 140, 16, 9, color(240, 174, 104), true);
    draw_text(dc, g_state->document.loaded ? g_state->document.posture : "CONTRACT-ONLY",
              left + 36, 171, kRightWidth - 72, 30, 20, color(231, 237, 243), true);
    draw_text(dc, g_state->document.loaded
                  ? "Local document attached. Native runtime evidence remains separate."
                  : "No runtime evidence attached.",
              left + 36, 218, kRightWidth - 72, 40, 11, color(116, 130, 148), false, DT_WORDBREAK);
    fill(dc, RECT{left + 36, 260, width - 36, 264}, color(43, 58, 77));
    fill(dc, RECT{left + 36, 260, left + 91, 264}, color(240, 174, 104));

    draw_text(dc, "ACTIVE DOCUMENT", left + 26, 322, 210, 16, 9, color(82, 96, 113), true);
    draw_text(dc, g_state->document.loaded ? g_state->document.name : "None loaded",
              left + 26, 352, kRightWidth - 52, 20, 12, color(185, 197, 209), true, DT_WORDBREAK);
    draw_text(dc, g_state->document.loaded ? g_state->document.type : "Waiting for local JSON",
              left + 26, 380, kRightWidth - 52, 20, 10, color(116, 130, 148));
    line(dc, left + 26, 418, width - 26, 418, color(30, 42, 56));
    draw_text(dc, "LOCAL GUARANTEES", left + 26, 446, 210, 16, 9, color(82, 96, 113), true);
    draw_text(dc, "OK  No network calls\nOK  No runtime claims\nOK  No authority expansion\nOK  Source stays local",
              left + 26, 476, kRightWidth - 52, 92, 10, color(185, 197, 209), false, DT_WORDBREAK);
    draw_text(dc, "CANOPY FOUNDRY", left + 26, height - 58, 210, 16, 9, color(82, 96, 113), true);
    draw_text(dc, "PUBLIC LOCAL SURFACE", left + 26, height - 36, 210, 16, 8, color(82, 96, 113));
}

void open_document(HWND window) {
    wchar_t path_buffer[MAX_PATH] = {};
    OPENFILENAMEW dialog{};
    dialog.lStructSize = sizeof(dialog);
    dialog.hwndOwner = window;
    dialog.lpstrFilter = L"JSON documents\0*.json\0All files\0*.*\0\0";
    dialog.lpstrFile = path_buffer;
    dialog.nMaxFile = MAX_PATH;
    dialog.lpstrInitialDir = g_state->frontend_root.c_str();
    dialog.Flags = OFN_FILEMUSTEXIST | OFN_PATHMUSTEXIST | OFN_NOCHANGEDIR;
    if (!GetOpenFileNameW(&dialog)) return;

    const int size = WideCharToMultiByte(
        CP_UTF8, 0, path_buffer, -1, nullptr, 0, nullptr, nullptr);
    if (size <= 1) return;
    std::string utf8(static_cast<std::size_t>(size), '\0');
    WideCharToMultiByte(CP_UTF8, 0, path_buffer, -1, utf8.data(), size, nullptr, nullptr);
    utf8.pop_back();
    std::string error;
    if (!load_document(*g_state, std::filesystem::path(utf8), error)) {
        MessageBoxW(window, wide(error).c_str(), L"Foundry shell", MB_OK | MB_ICONWARNING);
        return;
    }
    InvalidateRect(window, nullptr, FALSE);
}

void select_view(int y) {
    if (y >= 112 && y < 147) g_state->view = View::Overview;
    else if (y >= 156 && y < 191) g_state->view = View::Plan;
    else if (y >= 200 && y < 235) g_state->view = View::Capabilities;
    else if (y >= 244 && y < 279) g_state->view = View::Evidence;
    else if (y >= 343 && y < 378) g_state->view = View::Boundaries;
    else if (y >= 426 && y < 461) g_state->view = View::Simulation;
}

LRESULT CALLBACK window_proc(HWND window, UINT message, WPARAM w_param, LPARAM l_param) {
    switch (message) {
        case WM_PAINT: {
            PAINTSTRUCT paint_struct{};
            HDC dc = BeginPaint(window, &paint_struct);
            RECT client{};
            GetClientRect(window, &client);
            fill(dc, client, color(10, 14, 20));
            draw_header(dc, client.right);
            draw_nav(dc, client.bottom);
            draw_center(dc, client.right, client.bottom);
            draw_right_rail(dc, client.right, client.bottom);
            EndPaint(window, &paint_struct);
            return 0;
        }
        case WM_LBUTTONUP: {
            const int x = static_cast<short>(LOWORD(l_param));
            const int y = static_cast<short>(HIWORD(l_param));
            if (x < kLeftWidth) {
                select_view(y);
                InvalidateRect(window, nullptr, FALSE);
                return 0;
            }
            const int center_left = kLeftWidth + 45;
            if ((g_state->view == View::Overview && x >= center_left && x < center_left + 145 && y >= 352 && y < 400) ||
                (g_state->view == View::Plan && x >= center_left && x < center_left + 145 && y >= 204 && y < 255)) {
                open_document(window);
                return 0;
            }
            if (g_state->view == View::Overview &&
                x >= center_left + 279 && x < center_left + 407 && y >= 352 && y < 400) {
                start_simulation(g_state->simulation);
                g_state->view = View::Simulation;
                InvalidateRect(window, nullptr, FALSE);
                return 0;
            }
            return 0;
        }
        case WM_KEYDOWN:
            if (w_param == '1') g_state->view = View::Overview;
            else if (w_param == '2') g_state->view = View::Plan;
            else if (w_param == '3') g_state->view = View::Capabilities;
            else if (w_param == '4') g_state->view = View::Evidence;
            else if (w_param == '5') g_state->view = View::Boundaries;
            else if (w_param == 'S') {
                start_simulation(g_state->simulation);
                g_state->view = View::Simulation;
            } else if (w_param == 'P' || w_param == VK_SPACE) {
                toggle_simulation_pause(g_state->simulation);
            } else if (w_param == 'R') {
                reset_simulation(g_state->simulation);
            }
            if (w_param == 'L') {
                open_document(window);
                return 0;
            } else if (w_param == VK_ESCAPE) {
                DestroyWindow(window);
                return 0;
            }
            InvalidateRect(window, nullptr, FALSE);
            return 0;
        case WM_TIMER:
            if (w_param == kAnimationTimer && g_state != nullptr) {
                ++g_state->animation_tick;
                advance_simulation(g_state->simulation);
                InvalidateRect(window, nullptr, FALSE);
                return 0;
            }
            return 0;
        case WM_DESTROY:
            KillTimer(window, kAnimationTimer);
            PostQuitMessage(0);
            return 0;
        default:
            return DefWindowProcW(window, message, w_param, l_param);
    }
}

}  // namespace

int run_native_shell(ShellState& state) {
    g_state = &state;
    const wchar_t class_name[] = L"CanopyFoundryNativeShell";
    WNDCLASSW window_class{};
    window_class.hInstance = GetModuleHandleW(nullptr);
    window_class.lpfnWndProc = window_proc;
    window_class.lpszClassName = class_name;
    window_class.hCursor = LoadCursorW(nullptr, MAKEINTRESOURCEW(32512));
    window_class.hbrBackground = static_cast<HBRUSH>(GetStockObject(BLACK_BRUSH));
    if (RegisterClassW(&window_class) == 0) return 2;

    HWND window = CreateWindowExW(
        0, class_name, L"Canopy Foundry - Native Shell",
        WS_OVERLAPPEDWINDOW, CW_USEDEFAULT, CW_USEDEFAULT,
        1280, 800, nullptr, nullptr, window_class.hInstance, nullptr);
    if (window == nullptr) return 2;
    ShowWindow(window, SW_SHOW);
    UpdateWindow(window);
    SetTimer(window, kAnimationTimer, 100, nullptr);

    MSG message{};
    while (GetMessageW(&message, nullptr, 0, 0) > 0) {
        TranslateMessage(&message);
        DispatchMessageW(&message);
    }
    g_state = nullptr;
    return static_cast<int>(message.wParam);
}

}  // namespace canopy_foundry::shell

#endif
