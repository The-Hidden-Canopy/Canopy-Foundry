#include "ida_native/cpu_trainer.hpp"

#include "ida_native/cpu_kernels.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <functional>
#include <limits>
#include <random>
#include <stdexcept>
#include <thread>
#include <vector>

namespace ida_native {

namespace {

enum ParameterIndex : std::size_t {
    Embed,
    AttnNorm,
    QProj,
    KProj,
    VProj,
    OProj,
    FfnNorm,
    GateProj,
    UpProj,
    DownProj,
    FinalNorm,
    LmHead,
    ParameterCount,
};

struct Parameter {
    const char* name{nullptr};
    std::vector<float> value;
    std::vector<float> gradient;
    std::vector<float> moment;
    std::vector<float> variance;
};

struct Model {
    int hidden{0};
    int intermediate{0};
    int vocab{0};
    int tokens{0};
    std::array<Parameter, ParameterCount> parameters{};

    std::vector<float> x, normed, q, k, v, attn, h1, ffn_normed;
    std::vector<float> gate, up, ffn, h2, final_normed, logits, dlogits, row_loss;
    std::vector<float> d_k, d_v;
};

template <typename T>
std::vector<T> read_binary(const std::filesystem::path& path, std::size_t count) {
    std::ifstream input(path, std::ios::binary);
    if (!input) throw std::runtime_error("unable to open dataset file: " + path.string());
    std::vector<T> values(count);
    input.read(reinterpret_cast<char*>(values.data()),
               static_cast<std::streamsize>(values.size() * sizeof(T)));
    if (!input) throw std::runtime_error("dataset file is shorter than expected: " + path.string());
    return values;
}

float safe_rms(const float* values, int count) {
    float sum = 0.0f;
    for (int i = 0; i < count; ++i) sum += values[i] * values[i];
    return std::sqrt(sum / static_cast<float>(count) + 1.0e-6f);
}

float silu(float value) {
    return value / (1.0f + std::exp(-value));
}

float silu_grad(float value) {
    const float sigmoid = 1.0f / (1.0f + std::exp(-value));
    return sigmoid * (1.0f + value * (1.0f - sigmoid));
}

std::size_t default_thread_count() {
    const unsigned count = std::thread::hardware_concurrency();
    return std::max<std::size_t>(1, count == 0 ? 1u : count);
}

void parallel_for(
    std::size_t count,
    std::size_t requested_threads,
    const std::function<void(std::size_t, std::size_t)>& function
) {
    if (count == 0) return;
    const std::size_t workers = std::min(count, std::max<std::size_t>(1, requested_threads));
    if (workers == 1) {
        function(0, count);
        return;
    }
    std::vector<std::thread> threads;
    threads.reserve(workers);
    for (std::size_t worker = 0; worker < workers; ++worker) {
        const std::size_t begin = count * worker / workers;
        const std::size_t end = count * (worker + 1) / workers;
        threads.emplace_back([&, begin, end] { function(begin, end); });
    }
    for (auto& thread : threads) thread.join();
}

void validate_request(const NativeRequest& request) {
    const auto& m = request.model;
    if (request.device.runtime != "cpu")
        throw std::runtime_error("CPU request runtime mismatch");
    if (request.device.required_arch != "host" && request.device.required_arch != "auto")
        throw std::runtime_error("CPU smoke target requires device.required_arch=host or auto");
    if (request.precision_profile != "fp32" && request.precision_profile != "cpu_fp32")
        throw std::runtime_error("CPU smoke target supports precision_profile=fp32 only");
    throw std::runtime_error("CPU smoke target is disabled: Adam and AdamW are disabled and no CPU Lion path is enabled");
    if (request.attention_backend != "scalar_flash")
        throw std::runtime_error("CPU smoke target supports scalar_flash attention only");
    if (request.input.sequence_length != 2048 || request.input.batch_size != 1)
        throw std::runtime_error("CPU smoke target requires batch_size=1 and sequence_length=2048");
    if (m.layers != 1 || m.heads != 1 || m.hidden_size <= 0 || m.hidden_size > 256 ||
        m.intermediate_size <= 0 || m.intermediate_size > 1024 || m.vocab_size <= 1 ||
        m.vocab_size > 65536) {
        throw std::runtime_error(
            "CPU smoke target requires one layer, one head, hidden_size<=256, "
            "intermediate_size<=1024, and vocab_size<=65536");
    }
    if (m.num_cognitive_routes != 0 || m.top_k_routes != 0 ||
        m.num_personality_experts != 0 || m.top_k_experts != 0 ||
        m.use_personality_residual_experts || m.moe_native_fp4 || m.rope_theta != 0.0f ||
        request.pss_pred_rank > 0) {
        throw std::runtime_error(
            "CPU smoke target does not support MoE, RoPE, PSS, or other optional blocks");
    }
    if (!request.parent.init_from_model.empty() || !request.parent.resume_from_checkpoint.empty())
        throw std::runtime_error("CPU smoke target is fresh-training-only; resume is not supported");
    if (request.training.max_steps <= 0 || request.training.grad_accumulation <= 0)
        throw std::runtime_error("training.max_steps and training.grad_accumulation must be positive");
}

void allocate_model(Model& model) {
    const std::size_t H = static_cast<std::size_t>(model.hidden);
    const std::size_t I = static_cast<std::size_t>(model.intermediate);
    const std::size_t V = static_cast<std::size_t>(model.vocab);
    const std::size_t T = static_cast<std::size_t>(model.tokens);
    const std::array<const char*, ParameterCount> names = {
        "embed", "attn_norm", "q_proj", "k_proj", "v_proj", "o_proj",
        "ffn_norm", "gate_proj", "up_proj", "down_proj", "final_norm", "lm_head",
    };
    const std::array<std::size_t, ParameterCount> counts = {
        V * H, H, H * H, H * H, H * H, H * H, H, H * I, H * I, I * H, H, V * H,
    };
    for (std::size_t i = 0; i < ParameterCount; ++i) {
        auto& parameter = model.parameters[i];
        parameter.name = names[i];
        parameter.value.resize(counts[i]);
        parameter.gradient.assign(counts[i], 0.0f);
        parameter.moment.assign(counts[i], 0.0f);
        parameter.variance.assign(counts[i], 0.0f);
    }
    model.x.resize(T * H); model.normed.resize(T * H); model.q.resize(T * H);
    model.k.resize(T * H); model.v.resize(T * H); model.attn.resize(T * H);
    model.h1.resize(T * H); model.ffn_normed.resize(T * H);
    model.gate.resize(T * I); model.up.resize(T * I); model.ffn.resize(T * I);
    model.h2.resize(T * H); model.final_normed.resize(T * H);
    model.logits.resize(T * V); model.dlogits.resize(T * V); model.row_loss.resize(T);
    model.d_k.assign(T * H, 0.0f); model.d_v.assign(T * H, 0.0f);
}

void initialize_model(Model& model, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    auto initialize = [&](ParameterIndex index, float scale) {
        std::uniform_real_distribution<float> distribution(-scale, scale);
        for (float& value : model.parameters[index].value) value = distribution(generator);
    };
    auto ones = [&](ParameterIndex index) {
        std::fill(model.parameters[index].value.begin(), model.parameters[index].value.end(), 1.0f);
    };
    const float h_scale = 1.0f / std::sqrt(static_cast<float>(model.hidden));
    const float i_scale = 1.0f / std::sqrt(static_cast<float>(model.intermediate));
    initialize(Embed, 0.02f);
    ones(AttnNorm);
    initialize(QProj, h_scale); initialize(KProj, h_scale);
    initialize(VProj, h_scale); initialize(OProj, h_scale);
    ones(FfnNorm);
    initialize(GateProj, i_scale); initialize(UpProj, i_scale);
    initialize(DownProj, h_scale);
    ones(FinalNorm);
    initialize(LmHead, h_scale);
}

double checksum(const std::vector<float>& values) {
    double result = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i)
        result += static_cast<double>(values[i]) * static_cast<double>((i % 17) + 1);
    return result;
}

void reset_gradients(Model& model) {
    for (auto& parameter : model.parameters)
        std::fill(parameter.gradient.begin(), parameter.gradient.end(), 0.0f);
    std::fill(model.d_k.begin(), model.d_k.end(), 0.0f);
    std::fill(model.d_v.begin(), model.d_v.end(), 0.0f);
}

void forward_qkv(Model& model, const std::vector<std::uint32_t>& tokens, std::size_t threads) {
    const int H = model.hidden;
    const std::size_t token_count = static_cast<std::size_t>(model.tokens);
    const auto& embed = model.parameters[Embed].value;
    const auto& attn_norm = model.parameters[AttnNorm].value;
    const auto& q_proj = model.parameters[QProj].value;
    const auto& k_proj = model.parameters[KProj].value;
    const auto& v_proj = model.parameters[VProj].value;
    parallel_for(token_count, threads, [&](std::size_t begin, std::size_t end) {
        std::vector<float> values(H);
        std::vector<float> normalized(H);
        for (std::size_t t = begin; t < end; ++t) {
            const std::size_t row = t * static_cast<std::size_t>(H);
            const std::size_t embedding = static_cast<std::size_t>(tokens[t]) * H;
            for (int h = 0; h < H; ++h) values[h] = embed[embedding + h];
            const float rms = safe_rms(values.data(), H);
            for (int h = 0; h < H; ++h) {
                model.x[row + h] = values[h];
                normalized[h] = values[h] / rms * attn_norm[h];
                model.normed[row + h] = normalized[h];
            }
            for (int h = 0; h < H; ++h) {
                float q_value = 0.0f, k_value = 0.0f, v_value = 0.0f;
                for (int d = 0; d < H; ++d) {
                    q_value += normalized[d] * q_proj[static_cast<std::size_t>(d) * H + h];
                    k_value += normalized[d] * k_proj[static_cast<std::size_t>(d) * H + h];
                    v_value += normalized[d] * v_proj[static_cast<std::size_t>(d) * H + h];
                }
                model.q[row + h] = q_value;
                model.k[row + h] = k_value;
                model.v[row + h] = v_value;
            }
        }
    });
}

void forward_attention_ffn(Model& model, std::size_t threads) {
    const int H = model.hidden;
    const int I = model.intermediate;
    const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(H));
    const auto& q_proj = model.parameters[QProj].value;
    const auto& k_proj = model.parameters[KProj].value;
    const auto& v_proj = model.parameters[VProj].value;
    const auto& o_proj = model.parameters[OProj].value;
    const auto& ffn_norm = model.parameters[FfnNorm].value;
    const auto& gate_proj = model.parameters[GateProj].value;
    const auto& up_proj = model.parameters[UpProj].value;
    const auto& down_proj = model.parameters[DownProj].value;
    parallel_for(static_cast<std::size_t>(model.tokens), threads, [&](std::size_t begin, std::size_t end) {
        std::vector<float> attention(H), residual(H);
        for (std::size_t t = begin; t < end; ++t) {
            const std::size_t row = t * static_cast<std::size_t>(H);
            float max_score = -std::numeric_limits<float>::max();
            for (std::size_t j = 0; j <= t; ++j) {
                const float score = cpu_dot(model.q.data() + row, model.k.data() + j * H, H) * inv_sqrt_h;
                max_score = std::max(max_score, score);
            }
            float denominator = 0.0f;
            for (std::size_t j = 0; j <= t; ++j) {
                const float score = cpu_dot(model.q.data() + row, model.k.data() + j * H, H) * inv_sqrt_h;
                denominator += std::exp(score - max_score);
            }
            for (int d = 0; d < H; ++d) {
                float value = 0.0f;
                for (std::size_t j = 0; j <= t; ++j) {
                    const float score = cpu_dot(model.q.data() + row, model.k.data() + j * H, H) * inv_sqrt_h;
                    value += (std::exp(score - max_score) / denominator) * model.v[j * H + d];
                }
                attention[d] = value;
                model.attn[row + d] = value;
            }
            for (int h = 0; h < H; ++h) {
                float value = model.x[row + h];
                for (int d = 0; d < H; ++d) value += attention[d] * o_proj[static_cast<std::size_t>(d) * H + h];
                residual[h] = value;
                model.h1[row + h] = value;
            }
            const float rms = safe_rms(residual.data(), H);
            for (int h = 0; h < H; ++h) model.ffn_normed[row + h] = residual[h] / rms * ffn_norm[h];
            for (int i = 0; i < I; ++i) {
                float gate_value = 0.0f, up_value = 0.0f;
                for (int h = 0; h < H; ++h) {
                    gate_value += model.ffn_normed[row + h] * gate_proj[static_cast<std::size_t>(h) * I + i];
                    up_value += model.ffn_normed[row + h] * up_proj[static_cast<std::size_t>(h) * I + i];
                }
                model.gate[t * I + i] = gate_value;
                model.up[t * I + i] = up_value;
                model.ffn[t * I + i] = silu(gate_value) * up_value;
            }
            for (int h = 0; h < H; ++h) {
                float value = residual[h];
                for (int i = 0; i < I; ++i) value += model.ffn[t * I + i] * down_proj[static_cast<std::size_t>(i) * H + h];
                model.h2[row + h] = value;
            }
        }
    });
}

void forward_logits(Model& model, std::size_t threads) {
    const int H = model.hidden;
    const int V = model.vocab;
    const auto& final_norm = model.parameters[FinalNorm].value;
    const auto& lm_head = model.parameters[LmHead].value;
    parallel_for(static_cast<std::size_t>(model.tokens), threads, [&](std::size_t begin, std::size_t end) {
        std::vector<float> normalized(H);
        for (std::size_t t = begin; t < end; ++t) {
            const std::size_t row = t * static_cast<std::size_t>(H);
            const float rms = safe_rms(model.h2.data() + row, H);
            for (int h = 0; h < H; ++h) {
                normalized[h] = model.h2[row + h] / rms * final_norm[h];
                model.final_normed[row + h] = normalized[h];
            }
            for (int token = 0; token < V; ++token)
                model.logits[t * V + token] = cpu_dot(normalized.data(), lm_head.data() + static_cast<std::size_t>(token) * H, H);
        }
    });
}

void loss_and_gradient(Model& model, const std::vector<std::int32_t>& labels, std::size_t threads) {
    const int V = model.vocab;
    parallel_for(static_cast<std::size_t>(model.tokens), threads, [&](std::size_t begin, std::size_t end) {
        for (std::size_t t = begin; t < end; ++t) {
            const float* logits = model.logits.data() + t * static_cast<std::size_t>(V);
            float maximum = -std::numeric_limits<float>::max();
            for (int token = 0; token < V; ++token) maximum = std::max(maximum, logits[token]);
            float denominator = 0.0f;
            for (int token = 0; token < V; ++token) denominator += std::exp(logits[token] - maximum);
            model.row_loss[t] = std::log(denominator) + maximum - logits[labels[t]];
            for (int token = 0; token < V; ++token) {
                const float probability = std::exp(logits[token] - maximum) / denominator;
                model.dlogits[t * V + token] = probability - (token == labels[t] ? 1.0f : 0.0f);
            }
        }
    });
}

void backward(Model& model, const std::vector<std::uint32_t>& tokens) {
    const int H = model.hidden;
    const int I = model.intermediate;
    const int V = model.vocab;
    const float inv_sqrt_h = 1.0f / std::sqrt(static_cast<float>(H));
    const auto& embed = model.parameters[Embed].value;
    const auto& attn_norm = model.parameters[AttnNorm].value;
    const auto& q_proj = model.parameters[QProj].value;
    const auto& k_proj = model.parameters[KProj].value;
    const auto& v_proj = model.parameters[VProj].value;
    const auto& o_proj = model.parameters[OProj].value;
    const auto& ffn_norm = model.parameters[FfnNorm].value;
    const auto& gate_proj = model.parameters[GateProj].value;
    const auto& up_proj = model.parameters[UpProj].value;
    const auto& down_proj = model.parameters[DownProj].value;
    const auto& final_norm = model.parameters[FinalNorm].value;
    const auto& lm_head = model.parameters[LmHead].value;
    auto& g_embed = model.parameters[Embed].gradient;
    auto& g_attn_norm = model.parameters[AttnNorm].gradient;
    auto& g_q_proj = model.parameters[QProj].gradient;
    auto& g_k_proj = model.parameters[KProj].gradient;
    auto& g_v_proj = model.parameters[VProj].gradient;
    auto& g_o_proj = model.parameters[OProj].gradient;
    auto& g_ffn_norm = model.parameters[FfnNorm].gradient;
    auto& g_gate_proj = model.parameters[GateProj].gradient;
    auto& g_up_proj = model.parameters[UpProj].gradient;
    auto& g_down_proj = model.parameters[DownProj].gradient;
    auto& g_final_norm = model.parameters[FinalNorm].gradient;
    auto& g_lm_head = model.parameters[LmHead].gradient;

    std::vector<float> dy(H), dh2(H), dh1(H), dnormf(H), dattn(H), dqcur(H), dnorma(H);
    std::vector<float> h2v(H), h1v(H), xv(H);
    for (int t = model.tokens - 1; t >= 0; --t) {
        const std::size_t row = static_cast<std::size_t>(t) * H;
        for (int h = 0; h < H; ++h) h2v[h] = model.h2[row + h];
        const float final_rms = safe_rms(h2v.data(), H);
        float final_dot = 0.0f;
        for (int h = 0; h < H; ++h) {
            float value = 0.0f;
            for (int token = 0; token < V; ++token) {
                const float gradient = model.dlogits[static_cast<std::size_t>(t) * V + token];
                value += gradient * lm_head[static_cast<std::size_t>(token) * H + h];
                g_lm_head[static_cast<std::size_t>(token) * H + h] += gradient * model.final_normed[row + h];
            }
            dy[h] = value;
            final_dot += value * final_norm[h] * h2v[h];
            g_final_norm[h] += value * h2v[h] / final_rms;
        }
        for (int h = 0; h < H; ++h) {
            dh2[h] = (dy[h] * final_norm[h] - h2v[h] * final_dot /
                      (static_cast<float>(H) * final_rms * final_rms)) / final_rms;
            dh1[h] = dh2[h];
            dnormf[h] = 0.0f;
        }

        for (int i = 0; i < I; ++i) {
            float value = 0.0f;
            for (int h = 0; h < H; ++h) value += dh2[h] * down_proj[static_cast<std::size_t>(i) * H + h];
            for (int h = 0; h < H; ++h)
                g_down_proj[static_cast<std::size_t>(i) * H + h] +=
                    dh2[h] * model.ffn[static_cast<std::size_t>(t) * I + i];
            const float gate_value = model.gate[static_cast<std::size_t>(t) * I + i];
            const float up_value = model.up[static_cast<std::size_t>(t) * I + i];
            const float dgate = value * up_value * silu_grad(gate_value);
            const float dup = value * silu(gate_value);
            for (int h = 0; h < H; ++h) {
                const float input = model.ffn_normed[row + h];
                g_gate_proj[static_cast<std::size_t>(h) * I + i] += dgate * input;
                g_up_proj[static_cast<std::size_t>(h) * I + i] += dup * input;
                dnormf[h] += dgate * gate_proj[static_cast<std::size_t>(h) * I + i] +
                             dup * up_proj[static_cast<std::size_t>(h) * I + i];
            }
        }
        for (int h = 0; h < H; ++h) h1v[h] = model.h1[row + h];
        const float ffn_rms = safe_rms(h1v.data(), H);
        float ffn_dot = 0.0f;
        for (int h = 0; h < H; ++h) ffn_dot += dnormf[h] * ffn_norm[h] * h1v[h];
        for (int h = 0; h < H; ++h) {
            const float dh = (dnormf[h] * ffn_norm[h] - h1v[h] * ffn_dot /
                              (static_cast<float>(H) * ffn_rms * ffn_rms)) / ffn_rms;
            dh1[h] += dh;
            g_ffn_norm[h] += dnormf[h] * h1v[h] / ffn_rms;
            dattn[h] = 0.0f;
            dnorma[h] = 0.0f;
            dqcur[h] = 0.0f;
        }
        for (int d = 0; d < H; ++d) {
            for (int h = 0; h < H; ++h) {
                g_o_proj[static_cast<std::size_t>(d) * H + h] += dh1[h] * model.attn[row + d];
                dattn[d] += dh1[h] * o_proj[static_cast<std::size_t>(d) * H + h];
            }
        }

        float max_score = -std::numeric_limits<float>::max();
        for (int j = 0; j <= t; ++j)
            max_score = std::max(max_score, cpu_dot(model.q.data() + row, model.k.data() + static_cast<std::size_t>(j) * H, H) * inv_sqrt_h);
        float denominator = 0.0f;
        for (int j = 0; j <= t; ++j)
            denominator += std::exp(cpu_dot(model.q.data() + row, model.k.data() + static_cast<std::size_t>(j) * H, H) * inv_sqrt_h - max_score);
        float weighted_dot = 0.0f;
        for (int j = 0; j <= t; ++j) {
            const float probability = std::exp(cpu_dot(model.q.data() + row, model.k.data() + static_cast<std::size_t>(j) * H, H) * inv_sqrt_h - max_score) / denominator;
            weighted_dot += probability * cpu_dot(dattn.data(), model.v.data() + static_cast<std::size_t>(j) * H, H);
        }
        for (int j = 0; j <= t; ++j) {
            const std::size_t key_row = static_cast<std::size_t>(j) * H;
            const float probability = std::exp(cpu_dot(model.q.data() + row, model.k.data() + key_row, H) * inv_sqrt_h - max_score) / denominator;
            const float dot_value = cpu_dot(dattn.data(), model.v.data() + key_row, H);
            const float dscore = probability * (dot_value - weighted_dot);
            for (int d = 0; d < H; ++d) {
                dqcur[d] += dscore * model.k[key_row + d] * inv_sqrt_h;
                model.d_k[key_row + d] += dscore * model.q[row + d] * inv_sqrt_h;
                model.d_v[key_row + d] += probability * dattn[d];
            }
        }
        for (int h = 0; h < H; ++h) {
            const float input = model.normed[row + h];
            for (int d = 0; d < H; ++d) {
                g_q_proj[static_cast<std::size_t>(h) * H + d] += input * dqcur[d];
                g_k_proj[static_cast<std::size_t>(h) * H + d] += input * model.d_k[row + d];
                g_v_proj[static_cast<std::size_t>(h) * H + d] += input * model.d_v[row + d];
                dnorma[h] += dqcur[d] * q_proj[static_cast<std::size_t>(h) * H + d] +
                             model.d_k[row + d] * k_proj[static_cast<std::size_t>(h) * H + d] +
                             model.d_v[row + d] * v_proj[static_cast<std::size_t>(h) * H + d];
            }
        }
        for (int h = 0; h < H; ++h) xv[h] = model.x[row + h];
        const float attn_rms = safe_rms(xv.data(), H);
        float attn_dot = 0.0f;
        for (int h = 0; h < H; ++h) attn_dot += dnorma[h] * attn_norm[h] * xv[h];
        const std::size_t embedding = static_cast<std::size_t>(tokens[t]) * H;
        for (int h = 0; h < H; ++h) {
            const float dx = (dnorma[h] * attn_norm[h] - xv[h] * attn_dot /
                              (static_cast<float>(H) * attn_rms * attn_rms)) / attn_rms;
            g_attn_norm[h] += dnorma[h] * xv[h] / attn_rms;
            g_embed[embedding + h] += dx + dh1[h];
        }
    }
}

void adamw(Model& model, int step, int accumulation, double learning_rate) {
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float beta1_power = std::pow(beta1, static_cast<float>(step));
    const float beta2_power = std::pow(beta2, static_cast<float>(step));
    const float epsilon = 1.0e-8f;
    const float weight_decay = 0.01f;
    const float gradient_scale = 1.0f / static_cast<float>(accumulation);
    for (auto& parameter : model.parameters) {
        for (std::size_t i = 0; i < parameter.value.size(); ++i) {
            const float gradient = parameter.gradient[i] * gradient_scale;
            const float moment = beta1 * parameter.moment[i] + (1.0f - beta1) * gradient;
            const float variance = beta2 * parameter.variance[i] + (1.0f - beta2) * gradient * gradient;
            parameter.moment[i] = moment;
            parameter.variance[i] = variance;
            const float mhat = moment / (1.0f - beta1_power);
            const float vhat = variance / (1.0f - beta2_power);
            parameter.value[i] -= static_cast<float>(learning_rate) *
                (mhat / (std::sqrt(vhat) + epsilon) + weight_decay * parameter.value[i]);
        }
    }
}

}  // namespace

CPUTrainResult run_cpu_training(const NativeRequest& request, CPUTrainCallback on_step) {
    validate_request(request);
    const int token_count = request.input.sequence_length;
    const std::size_t expected_tokens = static_cast<std::size_t>(token_count);
    if (std::filesystem::file_size(request.input.token_blocks) != expected_tokens * sizeof(std::uint32_t) ||
        std::filesystem::file_size(request.input.label_blocks) != expected_tokens * sizeof(std::int32_t)) {
        throw std::runtime_error("CPU smoke dataset must contain exactly one 2048-token sequence");
    }
    const auto tokens = read_binary<std::uint32_t>(request.input.token_blocks, expected_tokens);
    const auto labels = read_binary<std::int32_t>(request.input.label_blocks, expected_tokens);
    for (int t = 0; t < token_count; ++t) {
        if (tokens[t] >= static_cast<std::uint32_t>(request.model.vocab_size))
            throw std::runtime_error("token id is outside model.vocab_size");
        if (labels[t] < 0 || labels[t] >= request.model.vocab_size)
            throw std::runtime_error("label id is outside model.vocab_size");
    }

    Model model{};
    model.hidden = request.model.hidden_size;
    model.intermediate = request.model.intermediate_size;
    model.vocab = request.model.vocab_size;
    model.tokens = token_count;
    allocate_model(model);
    initialize_model(model, static_cast<std::uint64_t>(request.seed));
    const double initial_checksum = checksum(model.parameters[LmHead].value);
    const std::size_t threads = default_thread_count();
    CPUTrainResult result{};
    result.threads = threads;
    result.kernel_variant = cpu_kernel_variant();
    result.device = "cpu";
    const int accumulation = std::max(1, request.training.grad_accumulation);
    const auto started = std::chrono::steady_clock::now();
    for (int step = 1; step <= request.training.max_steps; ++step) {
        reset_gradients(model);
        float average_loss = 0.0f;
        for (int micro = 0; micro < accumulation; ++micro) {
            forward_qkv(model, tokens, threads);
            forward_attention_ffn(model, threads);
            forward_logits(model, threads);
            loss_and_gradient(model, labels, threads);
            backward(model, tokens);
            for (float value : model.row_loss) average_loss += value;
        }
        average_loss /= static_cast<float>(token_count * accumulation);
        adamw(model, step, accumulation, request.training.learning_rate);
        result.steps = step;
        result.final_loss = average_loss;
        result.tokens_processed += static_cast<std::size_t>(token_count * accumulation);
        const double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - started).count();
        result.tokens_per_second = elapsed > 0.0
            ? static_cast<double>(result.tokens_processed) / elapsed : 0.0;
        if (on_step) on_step({step, average_loss, result.tokens_per_second, result.tokens_processed});
    }
    result.parameters_changed = initial_checksum != checksum(model.parameters[LmHead].value);
    return result;
}

}  // namespace ida_native
