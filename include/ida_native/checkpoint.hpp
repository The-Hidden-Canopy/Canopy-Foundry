#pragma once

#include <atomic>
#include <filesystem>
#include <string>
#include <thread>

#include <cuda_runtime.h>

#include "ida_native/request.hpp"
#include "ida_native/trainer.hpp"

namespace ida_native {

void write_checkpoint_artifacts(
    const NativeRequest& request,
    const SmokeStepResult& result
);

void write_final_artifacts(
    const NativeRequest& request,
    const SmokeStepResult& result
);

// ── Real weight persistence (safetensors, BF16) ─────────────────────────────
//
// Tensor names follow the native lattice contract (ida_lattice_native_v1):
//   embed_tokens.weight [V,H], layers.{l}.{attn_norm,q_proj,k_proj,v_proj,
//   o_proj,ffn_norm,gate_proj,up_proj,down_proj}.weight, final_norm.weight [H],
//   lm_head.weight [V,H].  Row-major native GEMM orientation as documented in
//   trainer.hpp — native linear-layer orientation.
//
// save: writes {output_dir}/model.safetensors atomically (tmp + rename) plus
// {output_dir}/native_model_manifest.json (the file the Python request builder
// resolves for parent lineage).  Returns false on any I/O or CUDA error.
// `cumulative_opt_steps` (2026-07-23): stamped into native_model_manifest.json
// alongside the same value save_lattice_opt_safetensors records in
// optimizer_state.safetensors' own metadata for the SAME checkpoint event —
// model.safetensors and optimizer_state.safetensors are two separate atomic
// tmp+rename writes, so a kill between them leaves a torn generation pair.
// load_lattice_weights_safetensors's own `out_cumulative_opt_steps` lets the
// resume path compare both files' recorded generation and hard-fail on
// disagreement rather than silently resuming from a mismatched pair. -1
// means "not tracked at this save point" (skips the manifest field's
// meaningfulness, not the write itself).
bool save_lattice_weights_safetensors(
    const NativeRequest& request,
    const LatticeWeights& w,
    cudaStream_t stream,
    std::string& error,
    int cumulative_opt_steps = -1
);

// Persist the two peer-pipeline shards as one canonical weight body.  The
// payload remains a regular model.safetensors file so a later child can load
// it through the normal parent-lineage path.  Optimizer state is deliberately
// excluded: exact mid-burn resume remains unsupported until both stage states
// can be committed as one generation.
bool save_model_parallel_lattice_weights_safetensors(
    const NativeRequest& request,
    const LatticeWeights& first_stage,
    int first_device,
    cudaStream_t first_stream,
    const LatticeWeights& second_stage,
    int second_device,
    cudaStream_t second_stream,
    std::string& error,
    int cumulative_opt_steps,
    int pipeline_split_layer
);

// load: `dir_or_file` may be a directory (uses dir/model.safetensors) or a
// .safetensors path.  Every expected tensor must be present with exact dtype
// BF16 and exact shape — any mismatch fails hard (scale-mismatch wedges must
// never be silently absorbed).  Returns false with `error` set on failure.
//
// `expected_spec_hash` (Phase 3, 2026-07-22): non-empty only for a genuine
// same-lineage resume (parent.resume_from_checkpoint), never for a
// cross-lineage parent load (init_from_model) — a child version legitimately
// has a different BurnSpec than its parent, so that call site always passes
// the default empty string (no-op). When non-empty and the manifest records
// its own spec_hash, a mismatch is a hard failure: resuming into a different
// BurnSpec's optimizer/cursor/curriculum state would silently corrupt
// training dynamics, unlike a LRSS/attn_window mismatch which at least fails
// on a real shape difference — this is the identity-level version of that
// same discipline.
// `out_cumulative_opt_steps` (2026-07-23, checkpoint-atomicity fix): when
// non-null, receives the manifest's own recorded cumulative_opt_steps (-1 if
// the manifest predates this field, e.g. an older checkpoint) so the resume
// caller can compare it against optimizer_state.safetensors' own recorded
// generation (populated separately by load_lattice_opt_safetensors) and
// hard-fail on a torn/mismatched pair. Not checked inside this function
// itself, since at THIS call site the optimizer state hasn't been loaded
// yet — the caller performs the actual comparison once both loads return.
bool load_lattice_weights_safetensors(
    const std::filesystem::path& dir_or_file,
    const LatticeWeights& w,
    cudaStream_t stream,
    std::string& error,
    const std::string& expected_spec_hash = "",
    int* out_cumulative_opt_steps = nullptr
);

// ── Optimizer state + resume cursor persistence (Phase 3, 2026-07-22) ───────
//
// Prior to this, a checkpoint was a WEIGHT SNAPSHOT only — Adam/Lion moments,
// the dataset read cursor, and the curriculum grad-accum ramp's cumulative
// step count were never persisted, so any resume silently re-warmed the
// optimizer from zero and re-read the dataset from row 0 regardless of how
// far the interrupted burn had actually progressed (see the now-superseded
// native_optimizer_placeholder.txt/native_rng_state_placeholder.txt text this
// mechanism replaces). Mirrors save_lattice_weights_safetensors/
// load_lattice_weights_safetensors exactly: same atomic tmp+rename write
// pattern, same directory-or-file load convention, written to
// {output_dir}/optimizer_state.safetensors alongside model.safetensors.
//
// Tensor names: optimizer.{name}.m / optimizer.{name}.v (v absent for Lion,
// which never allocates it — see LatticeOptState/OptStateTensor). dtype is
// F32 or BF16 depending on optimizer_state_bf16, matching whichever
// precision the resident buffers were actually allocated at (no implicit
// cast on save or load — a precision-profile change across a resume is a
// re-genesis event, same posture as save_lattice_weights_safetensors' own
// precision_profile check).
struct ResumeState {
    int cumulative_opt_steps{0};
    // Micro-step (data-exposure) position, distinct unit from
    // cumulative_opt_steps -- this is what accum_at()'s curriculum ramp and
    // the training loop's own exit condition (micro_done < total_micro) are
    // keyed on, seeded on resume instead of always starting the ramp/loop
    // at 0 regardless of how far the interrupted burn had progressed.
    int cumulative_micro_steps{0};
    std::size_t dataset_cursor{0};
};

bool save_lattice_opt_safetensors(
    const NativeRequest& request,
    const LatticeWeights& w,
    const LatticeOptState& opt,
    const ResumeState& resume_state,
    cudaStream_t stream,
    std::string& error
);

// Loads directly into the ALREADY-ALLOCATED opt buffers (shapes must match
// exactly — allocate_lattice_opt must have already run against the same
// request). Populates resume_state from the file's own metadata. Returns
// false with `error` set on any I/O, shape, or dtype mismatch.
bool load_lattice_opt_safetensors(
    const std::filesystem::path& dir_or_file,
    const LatticeWeights& w,
    LatticeOptState& opt,
    ResumeState& resume_state,
    cudaStream_t stream,
    std::string& error
);

// ── Standalone PSS predictor state (per-family×seat weights repo) ────────────
//
// The prediction head's lifetime is deliberately decoupled from the body
// checkpoint: its job is to accumulate training across burns, while bodies
// re-genesis. Saved as its own tiny safetensors file (ida_pss_pred_state_v1,
// tensors pss_predictor.{down,up}.weight) with an identity manifest in
// __metadata__ and a .provenance.json sidecar for the push/pull harness.
//
// Load semantics differ from the body loader on purpose: an identity mismatch
// (hidden_size / vocab_size / rank / family / seat) is a RESET — fresh init
// kept, reason reported — never a hard failure. The head predicts a specific
// body lineage's tail-FFN behavior over a specific token space; state from
// any other identity is noise, and refusing to train because stale noise
// exists on disk would invert the priority. Two deliberate levers:
//   IDA_NATIVE_PSS_PRED_STATE_RESET=1            force fresh init (skip load)
//   IDA_NATIVE_PSS_PRED_STATE_KEEP_ON_GENESIS=1  allow resume onto a
//     fresh-genesis body (default: genesis resets the head — a body with no
//     parent is a body the saved state has never observed). This is the
//     cross-genesis resume probe's lever, not a production default.
enum class PssPredStateLoad { kResumed, kReset, kFresh };

bool save_pss_pred_state_safetensors(
    const NativeRequest& request,
    const LatticeWeights& w,
    const std::filesystem::path& out_path,
    int cumulative_opt_steps,
    cudaStream_t stream,
    std::string& error
);

PssPredStateLoad load_pss_pred_state_safetensors(
    const NativeRequest& request,
    LatticeWeights& w,
    const std::filesystem::path& init_path,
    cudaStream_t stream,
    std::string& detail
);

// ── Native-side burn_timeline.jsonl phase events ─────────────────────────────
//
// 2026-07-21: train_student_native.py's Python-side "training" phase wraps
// wait_for_burn_completion(), which just polls the status file — accurate
// for the engine's TOTAL wall time, but the whole run is one opaque black
// box from Python's side. This writes phase-boundary rows directly from the
// engine into the SAME repo_root/artifacts/telemetry/burn_timeline.jsonl
// file, using the identical row schema src/ida_train/telemetry/timeline.py's
// write_event() produces, so summarize_timeline() picks these up for free —
// no new file, no IPC back to Python. Phase names are deliberately distinct
// from the Python-side "training" bucket (native_init, native_train_loop,
// native_checkpoint_save, native_pss_state_save) so they compose as a
// breakdown of "training" rather than double-counting into it.
//
// Never throws; any I/O failure is silently swallowed (a telemetry write
// must never break a burn) — same contract as the Python writer.
void write_native_timeline_event(
    const NativeRequest& request,
    const std::string& phase,
    const std::string& event,          // "start" | "end"
    double duration_s = -1.0           // negative = omit the field
);

// ── Async periodic weight persistence (pinned-DDR5 staged) ───────────────────
//
// begin_async_weight_save stages the full weight set into a pinned host
// buffer via a dedicated copy stream, then hands the file write (tmp +
// rename, same atomic protocol as the sync save) to a background thread.
// Ordering contract with the training stream:
//   1. copies are ordered AFTER all work queued on train_stream at call time
//      (the optimizer step that produced these weights), and
//   2. train_stream is made to wait on copy completion before any later
//      kernel runs — so the next optimizer step cannot mutate weights while
//      they are still being staged.  The D2H of the whole body (~90 ms for
//      the 2.3 GiB AI model over pinned PCIe) is the only training-visible
//      cost; the multi-second file write overlaps with training entirely.
// If a previous save is still writing, the call is skipped (best-effort
// periodic durability; saves_skipped counts these).  Never throws.
struct AsyncWeightSaver {
    void*        pinned{nullptr};       // cudaHostAlloc, whole weight body
    std::size_t  capacity{0};
    cudaStream_t copy_stream{nullptr};
    cudaEvent_t  ev_ready{nullptr};     // train stream reached save point
    cudaEvent_t  ev_copy_done{nullptr}; // pinned buffer holds the snapshot
    std::thread  writer;
    std::atomic<bool> busy{false};
    int saves_completed{0};
    int saves_skipped{0};
    std::string last_error;  // written by the writer thread before busy=false
};

// `cumulative_opt_steps`: same generation stamp as
// save_lattice_weights_safetensors' own parameter (2026-07-23) -- see that
// declaration's comment. Captured by value into the background writer
// thread, so it reflects the step count AT THE MOMENT this call was made,
// not whatever opt_step the caller may have advanced to by the time the
// thread actually runs.
bool begin_async_weight_save(
    const NativeRequest& request,
    const LatticeWeights& w,
    cudaStream_t train_stream,
    AsyncWeightSaver& saver,
    std::string& error,
    int cumulative_opt_steps = -1
);

// Blocks until any in-flight write finishes and releases saver resources.
// Safe to call multiple times / on a never-used saver.
void drain_async_weight_saver(AsyncWeightSaver& saver);

}  // namespace ida_native
