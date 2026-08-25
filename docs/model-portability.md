# Local model portability

Canopy Foundry can train from a user-owned local checkpoint after it has been
converted to the native SafeTensors contract. The repository contains no model
weights, tokenizer, dataset, credentials, PyTorch runtime, or personal
configuration. Checkpoint conversion is an external, machine-local step; no
Python ML framework is part of this repository.

## Supported conversion paths

The native runtime has explicit, fail-closed contracts for:

- GPT-2-compatible checkpoints use `hf_gpt2_native_v1`. The path preserves
  Conv1D orientation, learned absolute positions, LayerNorm weight/bias,
  attention and projection bias, dense feed-forward bias, and GELU/GELU-new.
- Llama-shaped dense checkpoints, including SmolLM-shaped checkpoints, use
  the native dense contract with rotary positions, grouped-query attention,
  and optional tied embeddings.
- Mixtral/Qwen-shaped top-k SwiGLU checkpoints use the generic MoE contract.
  The current contract covers the observed Mixtral-shaped portability surface
  only. Qwen-MoE always-on shared-expert tensors are rejected until a
  separately evidenced contract is implemented.

The external conversion step must reject missing tensors, partial bias sets,
incompatible shapes, ambiguous SafeTensors layouts, and unsupported model
families. It must never fill a missing weight with zeros. Its output should be
`model.safetensors` plus `native_model_manifest.json`, kept outside this
repository and bound locally by an opaque model ID in the governed worker
request. Do not add a converter, model weights, or framework lockfile here.

## GPT-2-style blank-slate config

`[configs/examples/gpt2_small.json](../configs/examples/gpt2_small.json)` is a
public shape template, not a personal profile and not a claim that weights are
included. Copy it into the ignored local config area and change the model,
tokenizer, dataset, and training values for your own run. The worker forwards
the model contract fields rather than selecting a hidden profile.

The native request accepts user-owned fields for architecture contract, hidden
and intermediate sizes, layer/head counts, normalization, activation, position
embeddings, bias flags, vocabulary size, context length, tied embeddings, and
generic MoE dimensions/top-k policy. Public CPU/OpenCL smoke targets reject
these optional CUDA model paths; they remain explicit smoke tests rather than
silent feature fallbacks.

## Boundary and telemetry posture

Private observability is compile-time disabled by default. Public checkpoint
manifests do not contain ontology or analytics paths, and the native timeline
writer is a no-op in a public build. Progress emitted to the worker remains
bounded and path-scrubbed. Training data, checkpoints, logs, and model
references remain machine-local.
