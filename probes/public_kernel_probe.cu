#include "ida_native/kernels.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

namespace {

bool check_cuda(cudaError_t error, const char* operation) {
    if (error == cudaSuccess) return true;
    std::fprintf(stderr, "%s: %s\n", operation, cudaGetErrorString(error));
    return false;
}

float bf16(float value) {
    return __bfloat162float(__float2bfloat16(value));
}

float sigmoid(float value) {
    if (value >= 0.0f) return 1.0f / (1.0f + std::exp(-value));
    const float e = std::exp(value);
    return e / (1.0f + e);
}

bool close_enough(float got, float want, float tolerance = 0.06f) {
    return std::fabs(got - want) <= tolerance;
}

}  // namespace

int main() {
    const int B = 1;
    const int S = 4;
    const int H = 2;
    const int max_positions = 4;
    const int N = 4;

    std::vector<__nv_bfloat16> hidden(B * S * H, __float2bfloat16(0.0f));
    std::vector<__nv_bfloat16> position(max_positions * H);
    for (int pos = 0; pos < max_positions; ++pos) {
        for (int h = 0; h < H; ++h) {
            position[pos * H + h] = __float2bfloat16(
                10.0f * static_cast<float>(pos) + static_cast<float>(h + 1));
        }
    }
    // The second sample begins at offset two, so positions are 0,1 again.
    const std::vector<std::uint16_t> segs = {0, 0, 2, 2};
    std::vector<__nv_bfloat16> grad_hidden(B * S * H, __float2bfloat16(1.0f));
    std::vector<float> grad_position(max_positions * H, 0.0f);

    const std::vector<__nv_bfloat16> gate = {
        __float2bfloat16(-2.0f), __float2bfloat16(-0.5f),
        __float2bfloat16(0.5f), __float2bfloat16(2.0f)};
    const std::vector<__nv_bfloat16> up = {
        __float2bfloat16(1.0f), __float2bfloat16(-2.0f),
        __float2bfloat16(0.25f), __float2bfloat16(3.0f)};
    std::vector<__nv_bfloat16> grad_out(N, __float2bfloat16(1.0f));

    __nv_bfloat16* d_hidden = nullptr;
    __nv_bfloat16* d_position = nullptr;
    std::uint16_t* d_segs = nullptr;
    __nv_bfloat16* d_grad_hidden = nullptr;
    float* d_grad_position = nullptr;
    __nv_bfloat16* d_gate = nullptr;
    __nv_bfloat16* d_up = nullptr;
    __nv_bfloat16* d_out = nullptr;
    __nv_bfloat16* d_grad_out = nullptr;
    __nv_bfloat16* d_grad_gate = nullptr;
    __nv_bfloat16* d_grad_up = nullptr;

    const auto alloc = [](auto** ptr, std::size_t bytes) {
        return cudaMalloc(reinterpret_cast<void**>(ptr), bytes) == cudaSuccess;
    };
    if (!alloc(&d_hidden, hidden.size() * sizeof(__nv_bfloat16)) ||
        !alloc(&d_position, position.size() * sizeof(__nv_bfloat16)) ||
        !alloc(&d_segs, segs.size() * sizeof(std::uint16_t)) ||
        !alloc(&d_grad_hidden, grad_hidden.size() * sizeof(__nv_bfloat16)) ||
        !alloc(&d_grad_position, grad_position.size() * sizeof(float)) ||
        !alloc(&d_gate, gate.size() * sizeof(__nv_bfloat16)) ||
        !alloc(&d_up, up.size() * sizeof(__nv_bfloat16)) ||
        !alloc(&d_out, N * sizeof(__nv_bfloat16)) ||
        !alloc(&d_grad_out, grad_out.size() * sizeof(__nv_bfloat16)) ||
        !alloc(&d_grad_gate, N * sizeof(__nv_bfloat16)) ||
        !alloc(&d_grad_up, N * sizeof(__nv_bfloat16))) {
        std::fprintf(stderr, "public_kernel_probe: allocation failed\n");
        return 2;
    }

    bool ok = true;
    ok = ok && check_cuda(cudaMemcpy(d_hidden, hidden.data(),
        hidden.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy hidden");
    ok = ok && check_cuda(cudaMemcpy(d_position, position.data(),
        position.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy position");
    ok = ok && check_cuda(cudaMemcpy(d_segs, segs.data(),
        segs.size() * sizeof(std::uint16_t), cudaMemcpyHostToDevice), "copy segs");
    ok = ok && check_cuda(cudaMemcpy(d_grad_hidden, grad_hidden.data(),
        grad_hidden.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy grad hidden");
    ok = ok && check_cuda(cudaMemset(d_grad_position, 0,
        grad_position.size() * sizeof(float)), "zero grad position");
    ok = ok && check_cuda(cudaMemcpy(d_gate, gate.data(),
        gate.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy gate");
    ok = ok && check_cuda(cudaMemcpy(d_up, up.data(),
        up.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy up");
    ok = ok && check_cuda(cudaMemcpy(d_grad_out, grad_out.data(),
        grad_out.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice), "copy grad out");
    if (!ok) return 3;

    ida_native::position_embedding_forward(
        d_hidden, d_position, d_segs, B, S, H, max_positions, nullptr);
    ida_native::position_embedding_backward(
        d_grad_hidden, d_segs, d_grad_position, B, S, H, max_positions, nullptr);
    ida_native::swiglu_forward(d_gate, d_up, d_out, N, nullptr);
    ida_native::swiglu_backward(
        d_grad_out, d_gate, d_up, d_grad_gate, d_grad_up, N, nullptr);
    ok = check_cuda(cudaDeviceSynchronize(), "synchronize public kernels");
    if (!ok) return 4;

    std::vector<__nv_bfloat16> got_hidden(hidden.size());
    std::vector<float> got_grad_position(grad_position.size());
    std::vector<__nv_bfloat16> got_out(N);
    std::vector<__nv_bfloat16> got_grad_gate(N);
    std::vector<__nv_bfloat16> got_grad_up(N);
    ok = ok && check_cuda(cudaMemcpy(got_hidden.data(), d_hidden,
        got_hidden.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "read hidden");
    ok = ok && check_cuda(cudaMemcpy(got_grad_position.data(), d_grad_position,
        got_grad_position.size() * sizeof(float), cudaMemcpyDeviceToHost), "read position grad");
    ok = ok && check_cuda(cudaMemcpy(got_out.data(), d_out,
        got_out.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "read swiglu");
    ok = ok && check_cuda(cudaMemcpy(got_grad_gate.data(), d_grad_gate,
        got_grad_gate.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "read gate grad");
    ok = ok && check_cuda(cudaMemcpy(got_grad_up.data(), d_grad_up,
        got_grad_up.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost), "read up grad");

    for (int s = 0; s < S; ++s) {
        const int pos = s < 2 ? s : s - 2;
        for (int h = 0; h < H; ++h) {
            const float want = 10.0f * static_cast<float>(pos) + h + 1.0f;
            if (!close_enough(__bfloat162float(got_hidden[s * H + h]), want)) ok = false;
        }
    }
    for (int pos = 0; pos < max_positions; ++pos) {
        for (int h = 0; h < H; ++h) {
            const float want = pos < 2 ? 2.0f : 0.0f;
            if (!close_enough(got_grad_position[pos * H + h], want)) ok = false;
        }
    }
    for (int i = 0; i < N; ++i) {
        const float gate_value = __bfloat162float(gate[i]);
        const float up_value = __bfloat162float(up[i]);
        const float sig = sigmoid(gate_value);
        const float silu_value = gate_value * sig;
        const float want_out = bf16(silu_value * up_value);
        const float want_gate = bf16(up_value * sig *
            (1.0f + gate_value * (1.0f - sig)));
        const float want_up = bf16(silu_value);
        if (!close_enough(__bfloat162float(got_out[i]), want_out) ||
            !close_enough(__bfloat162float(got_grad_gate[i]), want_gate) ||
            !close_enough(__bfloat162float(got_grad_up[i]), want_up)) {
            ok = false;
        }
    }

    cudaFree(d_hidden);
    cudaFree(d_position);
    cudaFree(d_segs);
    cudaFree(d_grad_hidden);
    cudaFree(d_grad_position);
    cudaFree(d_gate);
    cudaFree(d_up);
    cudaFree(d_out);
    cudaFree(d_grad_out);
    cudaFree(d_grad_gate);
    cudaFree(d_grad_up);

    std::puts(ok ? "public_kernel_probe: PASS" : "public_kernel_probe: FAIL");
    return ok ? 0 : 1;
}
