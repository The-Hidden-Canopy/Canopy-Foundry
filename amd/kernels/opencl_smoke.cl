// Portable FP32 dense smoke kernels.
//
// This file intentionally keeps the math scalar and architecture-neutral. It
// is a correctness/reference path for the OpenCL backend, not a replacement
// for the CUDA FP8/WGMMA kernels. The backward kernel uses one work-item so
// that gradient accumulation is deterministic and does not require optional
// floating-point atomic extensions.

#define EPSILON 1.0e-6f

inline float safe_rms(__private const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] * x[i];
    return sqrt(sum / (float)n + EPSILON);
}

inline float safe_rms_global(__global const float* x, int n) {
    float sum = 0.0f;
    for (int i = 0; i < n; ++i) sum += x[i] * x[i];
    return sqrt(sum / (float)n + EPSILON);
}

inline float silu(float x) {
    return x / (1.0f + exp(-x));
}

inline float silu_grad(float x) {
    const float s = 1.0f / (1.0f + exp(-x));
    return s * (1.0f + x * (1.0f - s));
}

__kernel void forward_qkv(
    __global const uint* tokens,
    __global const float* embed,
    __global const float* attn_norm,
    __global const float* q_proj,
    __global const float* k_proj,
    __global const float* v_proj,
    __global float* x,
    __global float* normed,
    __global float* q,
    __global float* k,
    __global float* v,
    int token_count,
    int hidden
) {
    const int t = (int)get_global_id(0);
    if (t >= token_count) return;
    const uint token = tokens[t];
    float xv[256];
    float nv[256];
    const float rms_scale = 1.0f;
    for (int h = 0; h < hidden; ++h) xv[h] = embed[(size_t)token * hidden + h];
    const float rms = safe_rms(xv, hidden);
    for (int h = 0; h < hidden; ++h) {
        x[(size_t)t * hidden + h] = xv[h];
        nv[h] = xv[h] / rms * attn_norm[h] * rms_scale;
        normed[(size_t)t * hidden + h] = nv[h];
    }
    for (int h = 0; h < hidden; ++h) {
        float qv = 0.0f;
        float kv = 0.0f;
        float vv = 0.0f;
        for (int d = 0; d < hidden; ++d) {
            qv += nv[d] * q_proj[(size_t)d * hidden + h];
            kv += nv[d] * k_proj[(size_t)d * hidden + h];
            vv += nv[d] * v_proj[(size_t)d * hidden + h];
        }
        q[(size_t)t * hidden + h] = qv;
        k[(size_t)t * hidden + h] = kv;
        v[(size_t)t * hidden + h] = vv;
    }
}

__kernel void forward_attn_ffn(
    __global const float* x,
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* o_proj,
    __global const float* ffn_norm,
    __global const float* gate_proj,
    __global const float* up_proj,
    __global const float* down_proj,
    __global float* attn,
    __global float* h1,
    __global float* ffn_normed,
    __global float* gate,
    __global float* up,
    __global float* ffn,
    __global float* h2,
    int token_count,
    int hidden,
    int intermediate
) {
    const int t = (int)get_global_id(0);
    if (t >= token_count) return;
    float av[256];
    float h1v[256];
    const float inv_sqrt_h = 1.0f / sqrt((float)hidden);
    float max_score = -3.402823466e+38f;
    for (int j = 0; j <= t; ++j) {
        float score = 0.0f;
        for (int d = 0; d < hidden; ++d)
            score += q[(size_t)t * hidden + d] * k[(size_t)j * hidden + d];
        max_score = fmax(max_score, score * inv_sqrt_h);
    }
    float denom = 0.0f;
    for (int j = 0; j <= t; ++j) {
        float score = 0.0f;
        for (int d = 0; d < hidden; ++d)
            score += q[(size_t)t * hidden + d] * k[(size_t)j * hidden + d];
        denom += exp(score * inv_sqrt_h - max_score);
    }
    for (int d = 0; d < hidden; ++d) {
        float value = 0.0f;
        for (int j = 0; j <= t; ++j) {
            float score = 0.0f;
            for (int qd = 0; qd < hidden; ++qd)
                score += q[(size_t)t * hidden + qd] * k[(size_t)j * hidden + qd];
            value += (exp(score * inv_sqrt_h - max_score) / denom) *
                     v[(size_t)j * hidden + d];
        }
        av[d] = value;
        attn[(size_t)t * hidden + d] = value;
    }
    for (int h = 0; h < hidden; ++h) {
        float value = x[(size_t)t * hidden + h];
        for (int d = 0; d < hidden; ++d)
            value += av[d] * o_proj[(size_t)d * hidden + h];
        h1v[h] = value;
        h1[(size_t)t * hidden + h] = value;
    }
    const float rms = safe_rms(h1v, hidden);
    for (int h = 0; h < hidden; ++h) {
        const float value = h1v[h] / rms * ffn_norm[h];
        ffn_normed[(size_t)t * hidden + h] = value;
    }
    for (int i = 0; i < intermediate; ++i) {
        float gate_value = 0.0f;
        float up_value = 0.0f;
        for (int h = 0; h < hidden; ++h) {
            gate_value += ffn_normed[(size_t)t * hidden + h] *
                          gate_proj[(size_t)h * intermediate + i];
            up_value += ffn_normed[(size_t)t * hidden + h] *
                        up_proj[(size_t)h * intermediate + i];
        }
        gate[(size_t)t * intermediate + i] = gate_value;
        up[(size_t)t * intermediate + i] = up_value;
        ffn[(size_t)t * intermediate + i] = silu(gate_value) * up_value;
    }
    for (int h = 0; h < hidden; ++h) {
        float value = h1v[h];
        for (int i = 0; i < intermediate; ++i)
            value += ffn[(size_t)t * intermediate + i] * down_proj[(size_t)i * hidden + h];
        h2[(size_t)t * hidden + h] = value;
    }
}

__kernel void forward_logits(
    __global const float* h2,
    __global const float* final_norm,
    __global const float* lm_head,
    __global float* final_normed,
    __global float* logits,
    int token_count,
    int hidden,
    int vocab
) {
    const int t = (int)get_global_id(0);
    if (t >= token_count) return;
    float hv[256];
    const float rms = safe_rms_global(h2 + (size_t)t * hidden, hidden);
    for (int h = 0; h < hidden; ++h) {
        hv[h] = h2[(size_t)t * hidden + h] / rms * final_norm[h];
        final_normed[(size_t)t * hidden + h] = hv[h];
    }
    for (int token = 0; token < vocab; ++token) {
        float value = 0.0f;
        for (int h = 0; h < hidden; ++h)
            value += hv[h] * lm_head[(size_t)token * hidden + h];
        logits[(size_t)t * vocab + token] = value;
    }
}

__kernel void loss_and_gradient(
    __global const float* logits,
    __global const int* labels,
    __global float* dlogits,
    __global float* row_loss,
    int token_count,
    int vocab,
    int valid_count
) {
    const int t = (int)get_global_id(0);
    if (t >= token_count) return;
    const int label = labels[t];
    if (label < 0) {
        row_loss[t] = 0.0f;
        for (int token = 0; token < vocab; ++token)
            dlogits[(size_t)t * vocab + token] = 0.0f;
        return;
    }
    float max_logit = -3.402823466e+38f;
    for (int token = 0; token < vocab; ++token)
        max_logit = fmax(max_logit, logits[(size_t)t * vocab + token]);
    float denom = 0.0f;
    for (int token = 0; token < vocab; ++token)
        denom += exp(logits[(size_t)t * vocab + token] - max_logit);
    const float target_logit = logits[(size_t)t * vocab + label];
    row_loss[t] = log(denom) + max_logit - target_logit;
    const float inv_valid = 1.0f / (float)valid_count;
    for (int token = 0; token < vocab; ++token) {
        const float probability = exp(logits[(size_t)t * vocab + token] - max_logit) / denom;
        dlogits[(size_t)t * vocab + token] =
            (probability - (token == label ? 1.0f : 0.0f)) * inv_valid;
    }
}

__kernel void backward_full(
    __global const uint* tokens,
    __global const float* embed,
    __global const float* attn_norm,
    __global const float* q_proj,
    __global const float* k_proj,
    __global const float* v_proj,
    __global const float* o_proj,
    __global const float* ffn_norm,
    __global const float* gate_proj,
    __global const float* up_proj,
    __global const float* down_proj,
    __global const float* final_norm,
    __global const float* lm_head,
    __global const float* x,
    __global const float* normed,
    __global const float* q,
    __global const float* k,
    __global const float* v,
    __global const float* attn,
    __global const float* h1,
    __global const float* ffn_normed,
    __global const float* gate,
    __global const float* up,
    __global const float* ffn,
    __global const float* h2,
    __global const float* final_normed,
    __global const float* dlogits,
    __global float* d_q,
    __global float* d_k,
    __global float* d_v,
    __global float* g_embed,
    __global float* g_attn_norm,
    __global float* g_q_proj,
    __global float* g_k_proj,
    __global float* g_v_proj,
    __global float* g_o_proj,
    __global float* g_ffn_norm,
    __global float* g_gate_proj,
    __global float* g_up_proj,
    __global float* g_down_proj,
    __global float* g_final_norm,
    __global float* g_lm_head,
    int token_count,
    int hidden,
    int intermediate,
    int vocab
) {
    if (get_global_id(0) != 0) return;
    float dy[256];
    float dh2[256];
    float dh1[256];
    float dnormf[256];
    float dnorma[256];
    float dattn[256];
    float dqcur[256];
    const float inv_sqrt_h = 1.0f / sqrt((float)hidden);

    for (int t = token_count - 1; t >= 0; --t) {
        float h2v[256];
        for (int h = 0; h < hidden; ++h) h2v[h] = h2[(size_t)t * hidden + h];
        const float final_rms = safe_rms(h2v, hidden);
        for (int h = 0; h < hidden; ++h) {
            float value = 0.0f;
            for (int token = 0; token < vocab; ++token) {
                const float dl = dlogits[(size_t)t * vocab + token];
                value += dl * lm_head[(size_t)token * hidden + h];
                g_lm_head[(size_t)token * hidden + h] +=
                    dl * final_normed[(size_t)t * hidden + h];
            }
            dy[h] = value;
            g_final_norm[h] += value * h2v[h] / final_rms;
        }
        for (int h = 0; h < hidden; ++h) {
            // Match CUDA's current RMSNorm backward contract: the native
            // kernel intentionally uses the first-order term and omits the
            // mean-correction term (see native/kernels/rmsnorm.cu).
            dh2[h] = dy[h] * final_norm[h] / final_rms;
            dh1[h] = dh2[h];
            dnormf[h] = 0.0f;
        }

        for (int i = 0; i < intermediate; ++i) {
            float value = 0.0f;
            for (int h = 0; h < hidden; ++h)
                value += dh2[h] * down_proj[(size_t)i * hidden + h];
            for (int h = 0; h < hidden; ++h)
                g_down_proj[(size_t)i * hidden + h] +=
                    dh2[h] * ffn[(size_t)t * intermediate + i];
            const float gate_value = gate[(size_t)t * intermediate + i];
            const float up_value = up[(size_t)t * intermediate + i];
            const float dgate = value * up_value * silu_grad(gate_value);
            const float dup = value * silu(gate_value);
            for (int h = 0; h < hidden; ++h) {
                const float input = ffn_normed[(size_t)t * hidden + h];
                g_gate_proj[(size_t)h * intermediate + i] += dgate * input;
                g_up_proj[(size_t)h * intermediate + i] += dup * input;
                dnormf[h] += dgate * gate_proj[(size_t)h * intermediate + i] +
                             dup * up_proj[(size_t)h * intermediate + i];
            }
        }
        float h1v[256];
        for (int h = 0; h < hidden; ++h) h1v[h] = h1[(size_t)t * hidden + h];
        const float ffn_rms = safe_rms(h1v, hidden);
        for (int h = 0; h < hidden; ++h) {
            const float dh = dnormf[h] * ffn_norm[h] / ffn_rms;
            dh1[h] += dh;
            g_ffn_norm[h] += dnormf[h] * h1v[h] / ffn_rms;
            dattn[h] = 0.0f;
            dnorma[h] = 0.0f;
            dqcur[h] = 0.0f;
        }
        for (int d = 0; d < hidden; ++d) {
            for (int h = 0; h < hidden; ++h) {
                g_o_proj[(size_t)d * hidden + h] +=
                    dh1[h] * attn[(size_t)t * hidden + d];
                dattn[d] += dh1[h] * o_proj[(size_t)d * hidden + h];
            }
        }

        float max_score = -3.402823466e+38f;
        for (int j = 0; j <= t; ++j) {
            float score = 0.0f;
            for (int d = 0; d < hidden; ++d)
                score += q[(size_t)t * hidden + d] * k[(size_t)j * hidden + d];
            max_score = fmax(max_score, score * inv_sqrt_h);
        }
        float denom = 0.0f;
        for (int j = 0; j <= t; ++j) {
            float score = 0.0f;
            for (int d = 0; d < hidden; ++d)
                score += q[(size_t)t * hidden + d] * k[(size_t)j * hidden + d];
            denom += exp(score * inv_sqrt_h - max_score);
        }
        float weighted_dot = 0.0f;
        for (int j = 0; j <= t; ++j) {
            float score = 0.0f;
            for (int d = 0; d < hidden; ++d)
                score += q[(size_t)t * hidden + d] * k[(size_t)j * hidden + d];
            const float probability = exp(score * inv_sqrt_h - max_score) / denom;
            float value = 0.0f;
            for (int d = 0; d < hidden; ++d)
                value += dattn[d] * v[(size_t)j * hidden + d];
            weighted_dot += probability * value;
        }
        for (int j = 0; j <= t; ++j) {
            float score = 0.0f;
            for (int d = 0; d < hidden; ++d)
                score += q[(size_t)t * hidden + d] * k[(size_t)j * hidden + d];
            const float probability = exp(score * inv_sqrt_h - max_score) / denom;
            float dot_value = 0.0f;
            for (int d = 0; d < hidden; ++d)
                dot_value += dattn[d] * v[(size_t)j * hidden + d];
            const float dscore = probability * (dot_value - weighted_dot);
            for (int d = 0; d < hidden; ++d) {
                dqcur[d] += dscore * k[(size_t)j * hidden + d] * inv_sqrt_h;
                d_k[(size_t)j * hidden + d] +=
                    dscore * q[(size_t)t * hidden + d] * inv_sqrt_h;
                d_v[(size_t)j * hidden + d] += probability * dattn[d];
            }
        }
        for (int h = 0; h < hidden; ++h) {
            const float input = normed[(size_t)t * hidden + h];
            for (int d = 0; d < hidden; ++d) {
                g_q_proj[(size_t)h * hidden + d] += input * dqcur[d];
                g_k_proj[(size_t)h * hidden + d] += input * d_k[(size_t)t * hidden + d];
                g_v_proj[(size_t)h * hidden + d] += input * d_v[(size_t)t * hidden + d];
                dnorma[h] += dqcur[d] * q_proj[(size_t)h * hidden + d] +
                             d_k[(size_t)t * hidden + d] * k_proj[(size_t)h * hidden + d] +
                             d_v[(size_t)t * hidden + d] * v_proj[(size_t)h * hidden + d];
            }
        }
        float xv[256];
        for (int h = 0; h < hidden; ++h) xv[h] = x[(size_t)t * hidden + h];
        const float attn_rms = safe_rms(xv, hidden);
        const uint token = tokens[t];
        for (int h = 0; h < hidden; ++h) {
            const float dx = dnorma[h] * attn_norm[h] / attn_rms;
            const float total_dx = dx + dh1[h];
            g_attn_norm[h] += dnorma[h] * xv[h] / attn_rms;
            g_embed[(size_t)token * hidden + h] += total_dx;
        }
    }
}

__kernel void adamw_update(
    __global float* parameter,
    __global float* moment,
    __global float* variance,
    __global const float* gradient,
    int count,
    float learning_rate,
    float beta1,
    float beta2,
    float beta1_power,
    float beta2_power,
    float epsilon,
    float weight_decay,
    float gradient_scale
) {
    const int i = (int)get_global_id(0);
    if (i >= count) return;
    const float gradient_value = gradient[i] * gradient_scale;
    const float m = beta1 * moment[i] + (1.0f - beta1) * gradient_value;
    const float v = beta2 * variance[i] + (1.0f - beta2) * gradient_value * gradient_value;
    moment[i] = m;
    variance[i] = v;
    const float mhat = m / (1.0f - beta1_power);
    const float vhat = v / (1.0f - beta2_power);
    parameter[i] -= learning_rate * (mhat / (sqrt(vhat) + epsilon) +
                                     weight_decay * parameter[i]);
}
