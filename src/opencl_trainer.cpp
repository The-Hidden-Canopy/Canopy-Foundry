#include "ida_native/opencl_trainer.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iterator>
#include <random>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace ida_native {

namespace {

struct Mem {
    cl_mem value{nullptr};
    Mem() = default;
    Mem(const Mem&) = delete;
    Mem& operator=(const Mem&) = delete;
    ~Mem() = default;
};

struct Parameter {
    const char* name;
    Mem* value;
    Mem* gradient;
    Mem* moment;
    Mem* variance;
    std::size_t count;
};

struct Model {
    int hidden{0};
    int intermediate{0};
    int vocab{0};
    int tokens{0};

    Mem embed, attn_norm, q_proj, k_proj, v_proj, o_proj;
    Mem ffn_norm, gate_proj, up_proj, down_proj, final_norm, lm_head;
    Mem x, normed, q, k, v, attn, h1, ffn_normed, gate, up, ffn, h2;
    Mem final_normed, logits, dlogits, row_loss;
    Mem d_q, d_k, d_v;
    Mem g_embed, g_attn_norm, g_q_proj, g_k_proj, g_v_proj, g_o_proj;
    Mem g_ffn_norm, g_gate_proj, g_up_proj, g_down_proj, g_final_norm, g_lm_head;
    Mem m_embed, m_attn_norm, m_q_proj, m_k_proj, m_v_proj, m_o_proj;
    Mem m_ffn_norm, m_gate_proj, m_up_proj, m_down_proj, m_final_norm, m_lm_head;
    Mem v_embed, v_attn_norm, v_q_proj, v_k_proj, v_v_proj, v_o_proj;
    Mem v_ffn_norm, v_gate_proj, v_up_proj, v_down_proj, v_final_norm, v_lm_head;
    std::vector<Parameter> parameters;
};

void release(Mem& mem) {
    if (mem.value != nullptr) {
        clReleaseMemObject(mem.value);
        mem.value = nullptr;
    }
}

void release_model(Model& model) {
    Mem* memories[] = {
        &model.embed, &model.attn_norm, &model.q_proj, &model.k_proj, &model.v_proj,
        &model.o_proj, &model.ffn_norm, &model.gate_proj, &model.up_proj, &model.down_proj,
        &model.final_norm, &model.lm_head, &model.x, &model.normed, &model.q, &model.k,
        &model.v, &model.attn, &model.h1, &model.ffn_normed, &model.gate, &model.up,
        &model.ffn, &model.h2, &model.final_normed, &model.logits, &model.dlogits,
        &model.row_loss, &model.d_q, &model.d_k, &model.d_v, &model.g_embed,
        &model.g_attn_norm, &model.g_q_proj, &model.g_k_proj, &model.g_v_proj, &model.g_o_proj,
        &model.g_ffn_norm, &model.g_gate_proj, &model.g_up_proj, &model.g_down_proj,
        &model.g_final_norm, &model.g_lm_head, &model.m_embed, &model.m_attn_norm,
        &model.m_q_proj, &model.m_k_proj, &model.m_v_proj, &model.m_o_proj, &model.m_ffn_norm,
        &model.m_gate_proj, &model.m_up_proj, &model.m_down_proj, &model.m_final_norm,
        &model.m_lm_head, &model.v_embed, &model.v_attn_norm, &model.v_q_proj, &model.v_k_proj,
        &model.v_v_proj, &model.v_o_proj, &model.v_ffn_norm, &model.v_gate_proj, &model.v_up_proj,
        &model.v_down_proj, &model.v_final_norm, &model.v_lm_head,
    };
    for (Mem* memory : memories) release(*memory);
}

cl_mem make_buffer(const OpenCLRuntime& runtime, std::size_t count) {
    if (count == 0) throw std::runtime_error("attempted to allocate an empty OpenCL buffer");
    cl_int status = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(runtime.context, CL_MEM_READ_WRITE,
                                   count * sizeof(float), nullptr, &status);
    check_opencl(status, "clCreateBuffer");
    return buffer;
}

cl_mem make_uint_buffer(const OpenCLRuntime& runtime, std::size_t count) {
    cl_int status = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY,
                                   count * sizeof(std::uint32_t), nullptr, &status);
    check_opencl(status, "clCreateBuffer(tokens)");
    return buffer;
}

cl_mem make_int_buffer(const OpenCLRuntime& runtime, std::size_t count) {
    cl_int status = CL_SUCCESS;
    cl_mem buffer = clCreateBuffer(runtime.context, CL_MEM_READ_ONLY,
                                   count * sizeof(std::int32_t), nullptr, &status);
    check_opencl(status, "clCreateBuffer(labels)");
    return buffer;
}

void write_floats(const OpenCLRuntime& runtime, cl_mem destination,
                  const std::vector<float>& values) {
    check_opencl(clEnqueueWriteBuffer(runtime.queue, destination, CL_TRUE, 0,
                                      values.size() * sizeof(float), values.data(),
                                      0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(float)");
}

void write_tokens(const OpenCLRuntime& runtime, cl_mem destination,
                  const std::vector<std::uint32_t>& values) {
    check_opencl(clEnqueueWriteBuffer(runtime.queue, destination, CL_TRUE, 0,
                                      values.size() * sizeof(std::uint32_t), values.data(),
                                      0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(tokens)");
}

void write_labels(const OpenCLRuntime& runtime, cl_mem destination,
                  const std::vector<std::int32_t>& values) {
    check_opencl(clEnqueueWriteBuffer(runtime.queue, destination, CL_TRUE, 0,
                                      values.size() * sizeof(std::int32_t), values.data(),
                                      0, nullptr, nullptr),
                 "clEnqueueWriteBuffer(labels)");
}

void zero_buffer(const OpenCLRuntime& runtime, Mem& memory, std::size_t count) {
    const float zero = 0.0f;
    check_opencl(clEnqueueFillBuffer(runtime.queue, memory.value, &zero, sizeof(zero), 0,
                                     count * sizeof(float), 0, nullptr, nullptr),
                 "clEnqueueFillBuffer(zero)");
}

template <typename T>
void set_arg(cl_kernel kernel, int& index, const T& value) {
    check_opencl(clSetKernelArg(kernel, static_cast<cl_uint>(index++), sizeof(T), &value),
                 "clSetKernelArg(value)");
}

void set_arg(cl_kernel kernel, int& index, cl_mem value) {
    check_opencl(clSetKernelArg(kernel, static_cast<cl_uint>(index++), sizeof(value), &value),
                 "clSetKernelArg(buffer)");
}

void enqueue_1d(const OpenCLRuntime& runtime, cl_kernel kernel, std::size_t count) {
    const std::size_t global_size = count;
    check_opencl(clEnqueueNDRangeKernel(runtime.queue, kernel, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel");
}

void enqueue_serial(const OpenCLRuntime& runtime, cl_kernel kernel) {
    const std::size_t global_size = 1;
    check_opencl(clEnqueueNDRangeKernel(runtime.queue, kernel, 1, nullptr,
                                        &global_size, nullptr, 0, nullptr, nullptr),
                 "clEnqueueNDRangeKernel(serial)");
}

float read_one(const OpenCLRuntime& runtime, cl_mem buffer) {
    float value = 0.0f;
    check_opencl(clEnqueueReadBuffer(runtime.queue, buffer, CL_TRUE, 0, sizeof(value),
                                     &value, 0, nullptr, nullptr),
                 "clEnqueueReadBuffer(scalar)");
    return value;
}

double checksum(const std::vector<float>& values) {
    double result = 0.0;
    for (std::size_t i = 0; i < values.size(); ++i)
        result += static_cast<double>(values[i]) * static_cast<double>((i % 17) + 1);
    return result;
}

std::vector<float> read_vector(const OpenCLRuntime& runtime, cl_mem buffer, std::size_t count) {
    std::vector<float> values(count);
    check_opencl(clEnqueueReadBuffer(runtime.queue, buffer, CL_TRUE, 0,
                                     count * sizeof(float), values.data(),
                                     0, nullptr, nullptr),
                 "clEnqueueReadBuffer(vector)");
    return values;
}

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

void validate_request(const NativeRequest& request) {
    const auto& m = request.model;
    if (request.device.runtime != "opencl") throw std::runtime_error("OpenCL request runtime mismatch");
    if (request.device.required_arch != "gfx1036" && request.device.required_arch != "auto")
        throw std::runtime_error("OpenCL smoke target requires device.required_arch=gfx1036 or auto");
    if (request.precision_profile != "fp32" && request.precision_profile != "opencl_fp32")
        throw std::runtime_error("OpenCL smoke target supports precision_profile=fp32 only");
    throw std::runtime_error("OpenCL smoke target is disabled: Adam and AdamW are disabled and no OpenCL Lion path is enabled");
    if (request.attention_backend != "scalar_flash")
        throw std::runtime_error("OpenCL smoke target supports scalar_flash attention only");
    if (request.input.sequence_length != 2048 || request.input.batch_size != 1)
        throw std::runtime_error("OpenCL smoke target requires batch_size=1 and sequence_length=2048");
    if (m.layers != 1 || m.heads != 1 || m.hidden_size <= 0 || m.hidden_size > 256 ||
        m.intermediate_size <= 0 || m.intermediate_size > 1024 || m.vocab_size <= 1 ||
        m.vocab_size > 65536) {
        throw std::runtime_error(
            "OpenCL smoke target requires one layer, one head, hidden_size<=256, "
            "intermediate_size<=1024, and vocab_size<=65536");
    }
    if (m.num_cognitive_routes != 0 || m.top_k_routes != 0 ||
        m.num_personality_experts != 0 || m.top_k_experts != 0 ||
        m.use_personality_residual_experts || m.moe_native_fp4 || m.rope_theta != 0.0f ||
        request.pss_pred_rank > 0) {
        throw std::runtime_error(
            "OpenCL smoke target does not support MoE, RoPE, PSS, or other optional blocks");
    }
    if (!request.parent.init_from_model.empty() || !request.parent.resume_from_checkpoint.empty())
        throw std::runtime_error("OpenCL smoke target is fresh-training-only; resume is not supported");
    if (request.training.max_steps <= 0 || request.training.grad_accumulation <= 0)
        throw std::runtime_error("training.max_steps and training.grad_accumulation must be positive");
}

void allocate_parameters(const OpenCLRuntime& runtime, Model& model) {
    const std::size_t H = static_cast<std::size_t>(model.hidden);
    const std::size_t I = static_cast<std::size_t>(model.intermediate);
    const std::size_t V = static_cast<std::size_t>(model.vocab);
    const std::size_t T = static_cast<std::size_t>(model.tokens);

    auto alloc = [&](Mem& memory, std::size_t count) { memory.value = make_buffer(runtime, count); };
    alloc(model.embed, V * H); alloc(model.attn_norm, H); alloc(model.q_proj, H * H);
    alloc(model.k_proj, H * H); alloc(model.v_proj, H * H); alloc(model.o_proj, H * H);
    alloc(model.ffn_norm, H); alloc(model.gate_proj, H * I); alloc(model.up_proj, H * I);
    alloc(model.down_proj, I * H); alloc(model.final_norm, H); alloc(model.lm_head, V * H);
    alloc(model.x, T * H); alloc(model.normed, T * H); alloc(model.q, T * H);
    alloc(model.k, T * H); alloc(model.v, T * H); alloc(model.attn, T * H);
    alloc(model.h1, T * H); alloc(model.ffn_normed, T * H); alloc(model.gate, T * I);
    alloc(model.up, T * I); alloc(model.ffn, T * I); alloc(model.h2, T * H);
    alloc(model.final_normed, T * H); alloc(model.logits, T * V); alloc(model.dlogits, T * V);
    alloc(model.row_loss, T); alloc(model.d_q, T * H); alloc(model.d_k, T * H); alloc(model.d_v, T * H);

    auto alloc_triplet = [&](Mem& gradient, Mem& moment, Mem& variance, std::size_t count) {
        alloc(gradient, count); alloc(moment, count); alloc(variance, count);
        zero_buffer(runtime, gradient, count); zero_buffer(runtime, moment, count);
        zero_buffer(runtime, variance, count);
    };
    alloc_triplet(model.g_embed, model.m_embed, model.v_embed, V * H);
    alloc_triplet(model.g_attn_norm, model.m_attn_norm, model.v_attn_norm, H);
    alloc_triplet(model.g_q_proj, model.m_q_proj, model.v_q_proj, H * H);
    alloc_triplet(model.g_k_proj, model.m_k_proj, model.v_k_proj, H * H);
    alloc_triplet(model.g_v_proj, model.m_v_proj, model.v_v_proj, H * H);
    alloc_triplet(model.g_o_proj, model.m_o_proj, model.v_o_proj, H * H);
    alloc_triplet(model.g_ffn_norm, model.m_ffn_norm, model.v_ffn_norm, H);
    alloc_triplet(model.g_gate_proj, model.m_gate_proj, model.v_gate_proj, H * I);
    alloc_triplet(model.g_up_proj, model.m_up_proj, model.v_up_proj, H * I);
    alloc_triplet(model.g_down_proj, model.m_down_proj, model.v_down_proj, I * H);
    alloc_triplet(model.g_final_norm, model.m_final_norm, model.v_final_norm, H);
    alloc_triplet(model.g_lm_head, model.m_lm_head, model.v_lm_head, V * H);

    model.parameters = {
        {"embed", &model.embed, &model.g_embed, &model.m_embed, &model.v_embed, V * H},
        {"attn_norm", &model.attn_norm, &model.g_attn_norm, &model.m_attn_norm, &model.v_attn_norm, H},
        {"q_proj", &model.q_proj, &model.g_q_proj, &model.m_q_proj, &model.v_q_proj, H * H},
        {"k_proj", &model.k_proj, &model.g_k_proj, &model.m_k_proj, &model.v_k_proj, H * H},
        {"v_proj", &model.v_proj, &model.g_v_proj, &model.m_v_proj, &model.v_v_proj, H * H},
        {"o_proj", &model.o_proj, &model.g_o_proj, &model.m_o_proj, &model.v_o_proj, H * H},
        {"ffn_norm", &model.ffn_norm, &model.g_ffn_norm, &model.m_ffn_norm, &model.v_ffn_norm, H},
        {"gate_proj", &model.gate_proj, &model.g_gate_proj, &model.m_gate_proj, &model.v_gate_proj, H * I},
        {"up_proj", &model.up_proj, &model.g_up_proj, &model.m_up_proj, &model.v_up_proj, H * I},
        {"down_proj", &model.down_proj, &model.g_down_proj, &model.m_down_proj, &model.v_down_proj, I * H},
        {"final_norm", &model.final_norm, &model.g_final_norm, &model.m_final_norm, &model.v_final_norm, H},
        {"lm_head", &model.lm_head, &model.g_lm_head, &model.m_lm_head, &model.v_lm_head, V * H},
    };
}

void initialize_weights(const OpenCLRuntime& runtime, Model& model, std::uint64_t seed) {
    std::mt19937_64 generator(seed);
    auto init = [&](Mem& memory, std::size_t count, float scale) {
        std::vector<float> values(count);
        std::uniform_real_distribution<float> distribution(-scale, scale);
        for (float& value : values) value = distribution(generator);
        write_floats(runtime, memory.value, values);
    };
    auto ones = [&](Mem& memory, std::size_t count) {
        write_floats(runtime, memory.value, std::vector<float>(count, 1.0f));
    };
    const float h_scale = 1.0f / std::sqrt(static_cast<float>(model.hidden));
    const float i_scale = 1.0f / std::sqrt(static_cast<float>(model.intermediate));
    init(model.embed, static_cast<std::size_t>(model.vocab) * model.hidden, 0.02f);
    ones(model.attn_norm, model.hidden); init(model.q_proj, model.hidden * model.hidden, h_scale);
    init(model.k_proj, model.hidden * model.hidden, h_scale);
    init(model.v_proj, model.hidden * model.hidden, h_scale);
    init(model.o_proj, model.hidden * model.hidden, h_scale);
    ones(model.ffn_norm, model.hidden); init(model.gate_proj, model.hidden * model.intermediate, i_scale);
    init(model.up_proj, model.hidden * model.intermediate, i_scale);
    init(model.down_proj, model.intermediate * model.hidden, h_scale);
    ones(model.final_norm, model.hidden); init(model.lm_head, model.vocab * model.hidden, h_scale);
}

void set_forward_qkv(cl_kernel kernel, const Model& m, cl_mem tokens) {
    int a = 0;
    set_arg(kernel, a, tokens); set_arg(kernel, a, m.embed.value); set_arg(kernel, a, m.attn_norm.value);
    set_arg(kernel, a, m.q_proj.value); set_arg(kernel, a, m.k_proj.value); set_arg(kernel, a, m.v_proj.value);
    set_arg(kernel, a, m.x.value); set_arg(kernel, a, m.normed.value); set_arg(kernel, a, m.q.value);
    set_arg(kernel, a, m.k.value); set_arg(kernel, a, m.v.value); set_arg(kernel, a, m.tokens); set_arg(kernel, a, m.hidden);
}

void set_forward_attn(cl_kernel kernel, const Model& m) {
    int a = 0;
    set_arg(kernel, a, m.x.value); set_arg(kernel, a, m.q.value); set_arg(kernel, a, m.k.value); set_arg(kernel, a, m.v.value);
    set_arg(kernel, a, m.o_proj.value); set_arg(kernel, a, m.ffn_norm.value); set_arg(kernel, a, m.gate_proj.value);
    set_arg(kernel, a, m.up_proj.value); set_arg(kernel, a, m.down_proj.value); set_arg(kernel, a, m.attn.value);
    set_arg(kernel, a, m.h1.value); set_arg(kernel, a, m.ffn_normed.value); set_arg(kernel, a, m.gate.value);
    set_arg(kernel, a, m.up.value); set_arg(kernel, a, m.ffn.value); set_arg(kernel, a, m.h2.value);
    set_arg(kernel, a, m.tokens); set_arg(kernel, a, m.hidden); set_arg(kernel, a, m.intermediate);
}

void set_forward_logits(cl_kernel kernel, const Model& m) {
    int a = 0;
    set_arg(kernel, a, m.h2.value); set_arg(kernel, a, m.final_norm.value); set_arg(kernel, a, m.lm_head.value);
    set_arg(kernel, a, m.final_normed.value); set_arg(kernel, a, m.logits.value);
    set_arg(kernel, a, m.tokens); set_arg(kernel, a, m.hidden); set_arg(kernel, a, m.vocab);
}

void set_loss(cl_kernel kernel, const Model& m, cl_mem labels) {
    int a = 0;
    set_arg(kernel, a, m.logits.value); set_arg(kernel, a, labels); set_arg(kernel, a, m.dlogits.value);
    set_arg(kernel, a, m.row_loss.value); set_arg(kernel, a, m.tokens); set_arg(kernel, a, m.vocab);
}

void set_backward(cl_kernel kernel, const Model& m, cl_mem tokens) {
    int a = 0;
    const cl_mem args[] = {
        tokens, m.embed.value, m.attn_norm.value, m.q_proj.value, m.k_proj.value, m.v_proj.value,
        m.o_proj.value, m.ffn_norm.value, m.gate_proj.value, m.up_proj.value, m.down_proj.value,
        m.final_norm.value, m.lm_head.value, m.x.value, m.normed.value, m.q.value, m.k.value,
        m.v.value, m.attn.value, m.h1.value, m.ffn_normed.value, m.gate.value, m.up.value,
        m.ffn.value, m.h2.value, m.final_normed.value, m.dlogits.value, m.d_q.value, m.d_k.value,
        m.d_v.value, m.g_embed.value, m.g_attn_norm.value, m.g_q_proj.value, m.g_k_proj.value,
        m.g_v_proj.value, m.g_o_proj.value, m.g_ffn_norm.value, m.g_gate_proj.value, m.g_up_proj.value,
        m.g_down_proj.value, m.g_final_norm.value, m.g_lm_head.value,
    };
    for (cl_mem value : args) set_arg(kernel, a, value);
    set_arg(kernel, a, m.tokens); set_arg(kernel, a, m.hidden); set_arg(kernel, a, m.intermediate); set_arg(kernel, a, m.vocab);
}

void reset_gradients(const OpenCLRuntime& runtime, Model& m) {
    const std::size_t H = static_cast<std::size_t>(m.hidden);
    const std::size_t I = static_cast<std::size_t>(m.intermediate);
    const std::size_t V = static_cast<std::size_t>(m.vocab);
    Mem* gradients[] = {
        &m.g_embed, &m.g_attn_norm, &m.g_q_proj, &m.g_k_proj, &m.g_v_proj, &m.g_o_proj,
        &m.g_ffn_norm, &m.g_gate_proj, &m.g_up_proj, &m.g_down_proj, &m.g_final_norm, &m.g_lm_head,
        &m.d_q, &m.d_k, &m.d_v,
    };
    const std::size_t counts[] = {
        V * H, H, H * H, H * H, H * H, H * H, H, H * I, H * I, I * H, H, V * H,
        static_cast<std::size_t>(m.tokens) * H, static_cast<std::size_t>(m.tokens) * H,
        static_cast<std::size_t>(m.tokens) * H,
    };
    for (std::size_t i = 0; i < std::size(gradients); ++i)
        zero_buffer(runtime, *gradients[i], counts[i]);
}

void set_adamw(cl_kernel kernel, const Parameter& p, int step, int accumulation,
               double learning_rate) {
    int a = 0;
    const float beta1 = 0.9f;
    const float beta2 = 0.999f;
    const float beta1_power = std::pow(beta1, static_cast<float>(step));
    const float beta2_power = std::pow(beta2, static_cast<float>(step));
    const float epsilon = 1.0e-8f;
    const float weight_decay = 0.01f;
    const float gradient_scale = 1.0f / static_cast<float>(accumulation);
    set_arg(kernel, a, p.value->value); set_arg(kernel, a, p.moment->value);
    set_arg(kernel, a, p.variance->value); set_arg(kernel, a, p.gradient->value);
    const int count = static_cast<int>(p.count);
    set_arg(kernel, a, count); set_arg(kernel, a, static_cast<float>(learning_rate));
    set_arg(kernel, a, beta1); set_arg(kernel, a, beta2); set_arg(kernel, a, beta1_power);
    set_arg(kernel, a, beta2_power); set_arg(kernel, a, epsilon); set_arg(kernel, a, weight_decay);
    set_arg(kernel, a, gradient_scale);
}

}  // namespace

OpenCLTrainResult run_opencl_training(
    const NativeRequest& request,
    OpenCLRuntime& runtime,
    const std::filesystem::path& kernel_source,
    OpenCLTrainCallback on_step
) {
    validate_request(request);
    const int T = request.input.sequence_length;
    const std::size_t token_bytes = std::filesystem::file_size(request.input.token_blocks);
    if (token_bytes != static_cast<std::size_t>(T) * sizeof(std::uint32_t) ||
        std::filesystem::file_size(request.input.label_blocks) != token_bytes) {
        throw std::runtime_error("OpenCL smoke dataset must contain exactly one 2048-token sequence");
    }
    const auto tokens = read_binary<std::uint32_t>(request.input.token_blocks, T);
    const auto labels = read_binary<std::int32_t>(request.input.label_blocks, T);
    for (int t = 0; t < T; ++t) {
        if (tokens[t] >= static_cast<std::uint32_t>(request.model.vocab_size))
            throw std::runtime_error("token id is outside model.vocab_size");
        if (labels[t] < 0 || labels[t] >= request.model.vocab_size)
            throw std::runtime_error("label id is outside model.vocab_size");
    }

    cl_program program = build_opencl_program(runtime, kernel_source);
    Model model{};
    model.hidden = request.model.hidden_size;
    model.intermediate = request.model.intermediate_size;
    model.vocab = request.model.vocab_size;
    model.tokens = T;
    cl_mem token_buffer = nullptr;
    cl_mem label_buffer = nullptr;
    cl_kernel qkv = nullptr;
    cl_kernel attn = nullptr;
    cl_kernel logits = nullptr;
    cl_kernel loss = nullptr;
    cl_kernel backward = nullptr;
    cl_kernel adamw = nullptr;
    try {
        allocate_parameters(runtime, model);
        initialize_weights(runtime, model, static_cast<std::uint64_t>(request.seed));
        const double initial_lm_checksum = checksum(
            read_vector(runtime, model.lm_head.value,
                        static_cast<std::size_t>(model.vocab) * model.hidden));
        token_buffer = make_uint_buffer(runtime, tokens.size());
        label_buffer = make_int_buffer(runtime, labels.size());
        write_tokens(runtime, token_buffer, tokens);
        write_labels(runtime, label_buffer, labels);
        qkv = clCreateKernel(program, "forward_qkv", nullptr);
        attn = clCreateKernel(program, "forward_attn_ffn", nullptr);
        logits = clCreateKernel(program, "forward_logits", nullptr);
        loss = clCreateKernel(program, "loss_and_gradient", nullptr);
        backward = clCreateKernel(program, "backward_full", nullptr);
        adamw = clCreateKernel(program, "adamw_update", nullptr);
        if (!qkv || !attn || !logits || !loss || !backward || !adamw)
            throw std::runtime_error("unable to create one or more OpenCL smoke kernels");

        set_forward_qkv(qkv, model, token_buffer);
        set_forward_attn(attn, model);
        set_forward_logits(logits, model);
        set_loss(loss, model, label_buffer);
        set_backward(backward, model, token_buffer);

        OpenCLTrainResult result{};
        result.device = runtime.device_name;
        const auto started = std::chrono::steady_clock::now();
        const int accumulation = std::max(1, request.training.grad_accumulation);
        for (int step = 1; step <= request.training.max_steps; ++step) {
            reset_gradients(runtime, model);
            float average_loss = 0.0f;
            for (int micro = 0; micro < accumulation; ++micro) {
                enqueue_1d(runtime, qkv, T);
                enqueue_1d(runtime, attn, T);
                enqueue_1d(runtime, logits, T);
                enqueue_1d(runtime, loss, T);
                enqueue_serial(runtime, backward);
                const auto row_losses = read_vector(runtime, model.row_loss.value, T);
                for (float row_loss : row_losses) average_loss += row_loss;
            }
            average_loss /= static_cast<float>(T * accumulation);
            for (const Parameter& parameter : model.parameters) {
                set_adamw(adamw, parameter, step, accumulation, request.training.learning_rate);
                enqueue_1d(runtime, adamw, parameter.count);
            }
            check_opencl(clFinish(runtime.queue), "clFinish(step)");
            result.steps = step;
            result.final_loss = average_loss;
            result.tokens_processed += static_cast<std::size_t>(T * accumulation);
            const double elapsed = std::chrono::duration<double>(
                std::chrono::steady_clock::now() - started).count();
            result.tokens_per_second = elapsed > 0.0
                ? static_cast<double>(result.tokens_processed) / elapsed : 0.0;
            if (on_step) on_step({step, average_loss, result.tokens_per_second, result.tokens_processed});
        }
        const double final_lm_checksum = checksum(
            read_vector(runtime, model.lm_head.value,
                        static_cast<std::size_t>(model.vocab) * model.hidden));
        result.parameters_changed = initial_lm_checksum != final_lm_checksum;
        release_model(model);
        if (adamw) clReleaseKernel(adamw);
        if (backward) clReleaseKernel(backward);
        if (loss) clReleaseKernel(loss);
        if (logits) clReleaseKernel(logits);
        if (attn) clReleaseKernel(attn);
        if (qkv) clReleaseKernel(qkv);
        if (label_buffer) clReleaseMemObject(label_buffer);
        if (token_buffer) clReleaseMemObject(token_buffer);
        clReleaseProgram(program);
        return result;
    } catch (...) {
        release_model(model);
        if (adamw) clReleaseKernel(adamw);
        if (backward) clReleaseKernel(backward);
        if (loss) clReleaseKernel(loss);
        if (logits) clReleaseKernel(logits);
        if (attn) clReleaseKernel(attn);
        if (qkv) clReleaseKernel(qkv);
        if (label_buffer) clReleaseMemObject(label_buffer);
        if (token_buffer) clReleaseMemObject(token_buffer);
        clReleaseProgram(program);
        throw;
    }
}

}  // namespace ida_native
