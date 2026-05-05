# VGGT-TTT

A linear-memory drop-in for VGGT's global attention. Replaces the quadratic global-attention layers in [VGGT](https://github.com/facebookresearch/vggt) with a [LaCT](https://github.com/a1600012888/LaCT)-style fast-weight GLU block (per-head `FastWeightGluMLPMultihead`), distilled from the frozen VGGT teacher. Frame count scales linearly in VRAM, scene state persists across chunks via detached fast weights, and the original VGGT prediction heads (camera, depth, world points) ride on top unchanged.

## What changed vs vanilla VGGT

- **Global attention → LaCT block** (frame-wise attention unchanged).
- **2D RoPE preserved** on the LaCT path (matches VGGT's global attn).
- **Cross-chunk fast-weight state** (`w0,w1,w2`) detached between chunks; optional TBPTT window keeps gradients flowing across the last *K* chunks.
- **Slim checkpoints** (`lact_state_dict()`) — only the trainable LaCT params, ~200 MB.

## Results (DL3DV-Evaluation, 19 scenes)

VGGT teacher vs VGGT-TTT student (LaCT, stage-1 distill), Blackwell RTX Pro 6000 96 GB:

| Frames | Student VRAM | Teacher VRAM | Student secs | Teacher secs |  AbsRel | δ < 1.25 |
|-------:|-------------:|-------------:|-------------:|-------------:|--------:|---------:|
|     16 |      14.4 GB |      20.3 GB |          2.0 |          1.3 |   0.171 |    0.729 |
|     32 |      15.8 GB |      27.7 GB |          3.8 |          3.4 |   0.177 |    0.706 |
|     48 |      17.7 GB |      35.2 GB |          5.7 |          6.2 |   0.188 |    0.686 |
|     64 |      20.5 GB |      42.8 GB |          7.6 |          9.8 |   0.192 |    0.675 |
|     96 |      26.2 GB |      59.9 GB |         11.4 |         19.7 |   0.208 |    0.654 |
|    128 |      31.8 GB |  **76.9 GB** |         15.2 |     **32.5** |   0.221 |    0.629 |

- **VRAM**: student is linear (~0.16 GB/frame); teacher's super-linear curve is heading for OOM around N≈150 on a 96 GB card.
- **Latency**: tokens/sec is flat at ~20k for the student across all *N*. At N=128 the student is **2.1× faster** at **41% of teacher VRAM**.
- **Distillation fidelity**: depth tracks the teacher (AbsRel, δ<1.25 vs teacher predictions) and degrades gracefully with sequence length.

Crossover is around N≈48: short sequences favor the teacher (well-fused global attention), long sequences favor the student. The story is long-context, not short-burst.

Raw numbers in [`bench.json`](bench.json); reproduce via [`scripts/benchmark_vs_vggt.py`](scripts/benchmark_vs_vggt.py).

## Architecture

```
images ── DINOv2 patch embed (frozen) ──┐
                                        ▼
   ┌─ frame-wise attention (frozen, per-frame, RoPE) ─┐
   │                                                   │
   │   ┌─ LaCT block (cross-frame, RoPE on q,k) ─┐   │  × 24
   │   │   fast weights w0,w1,w2 persist          │   │
   │   │   between chunks (detached for memory)   │   │
   │   └───────────────────────────────────────────┘   │
   └────────────────────────────────────────────────────┘
                                        ▼
        camera / depth / point heads (frozen, fp32)
```

## Project layout

| Path | Role |
|---|---|
| `model/lact_ttt_glu.py` | tttLRM `FastWeightGluMLPMultihead` with optional 2D RoPE on q,k |
| `model/lact_block.py` | LayerNorm + GLU TTT + residual; padding-aware mini-batch; optional TBPTT |
| `model/ttt_aggregator.py` | VGGT aggregator with LaCT in place of global attention |
| `model/vggt_ttt.py` | Top-level model + slim `lact_state_dict()` save/load |
| `model/io_utils.py` | Safe checkpoint I/O |
| `model/losses.py` | Distillation + consistency losses |
| `pipeline/input_pipeline.py` | Image/video preprocessing |
| `scripts/script_common.py` | Blackwell-friendly defaults (TF32, fused AdamW, compile guard) |
| `scripts/run_inference.py` | Inference CLI |
| `scripts/benchmark_vs_vggt.py` | Teacher-vs-student benchmark with HF streaming |
| `configs/default.yaml` | Reference defaults (CLI scripts use argparse) |

Training scripts (`finetune.py`, `dl3dv_streaming.py`, `download.py`, `evaluate_stage1.py`) are not included in this public repo; they are data- and infrastructure-specific.

## Install

```bash
git clone https://github.com/<you>/vggt_ttt && cd vggt_ttt
pip install -e .
```

Install [VGGT](https://github.com/facebookresearch/vggt) separately (the model loads `facebook/VGGT-1B` from HuggingFace on first call).

## Inference

```bash
python scripts/run_inference.py \
    --input my_video.mp4 --fps 2 \
    --lact_ckpt /path/to/vggt_ttt_lact_stage1.pt \
    --out ./out
```

The slim LaCT checkpoint (~200 MB) is loaded on top of `facebook/VGGT-1B`; everything else (frame-wise blocks, prediction heads) comes from the public VGGT release.

## Training (stage 1 distillation)

The LaCT blocks are the only trainable parameters; everything else (DINOv2 patch embed, frame-wise attention, prediction heads) stays frozen at VGGT's pretrained weights. The student learns to mimic the teacher's per-layer aggregated tokens so the frozen heads keep working unchanged.

**Setup**
- Teacher: frozen `facebook/VGGT-1B`, `eval()`, no grad.
- Student: `VGGT_TTT(teacher, share_frame_blocks=True)` — student reuses the teacher's frame-wise blocks, only the 24 LaCT blocks (replacing global attention) are trainable. LaCT `c_proj` is zero-initialized so the student starts as a near-identity copy of the teacher.
- Both run on the same GPU; with `share_frame_blocks=True` no weights are duplicated.

**Loss** (per training step on a single video clip):
- **Token alignment**: cosine + L2 between student `aggregated_tokens_list[ℓ]` and teacher's, at each layer ℓ that any head reads from (`depth_head.intermediate_layer_idx ∪ point_head.intermediate_layer_idx ∪ {last}`). This is the dominant signal — it forces the LaCT path to reconstruct the global-attention output the heads expect.
- **Head consistency**: small auxiliary terms on the heads themselves — pose (rotation geodesic + translation), depth (L1 in log-space), point (L2). Optional; switched on once token alignment is below a threshold.

**Optimization**
- bf16 autocast in the trunk; heads + pose decoding stay fp32.
- Fused AdamW (`fused=True` on Blackwell), lr=1e-4, cosine decay, 5 epochs.
- `chunk_size` matches the per-frame token budget the LaCT block sees per cross-frame mixing call; we used 16. Cross-chunk fast weights `w0,w1,w2` are detached between chunks so memory is bounded.
- Gradient checkpointing at the GLU boundary (`gradient_checkpoint=True`) for sequences past ~64 frames.
- Optional `--tbptt_window N>0` keeps gradients flowing across the last *N* chunks within a clip.
- `reset_memory()` is called at every scene boundary so fast weights don't leak across videos.

**Data**
- DL3DV-ALL streamed from HuggingFace (one tar at a time, ~9 GB each, deleted after the scene is consumed). Random `seq_len` clip per scene; 32 frames at 518² in stage 1.

**Checkpoints**
- Only the LaCT params are saved (`model.lact_state_dict()`), ~200 MB. Reload with `model.load_lact_state_dict(state, strict=True)` on top of a fresh `VGGT_TTT.from_pretrained(...)`.

The actual training scripts (`scripts/finetune.py`, `scripts/dl3dv_streaming.py`) are dataset- and infra-specific (Colab + Google Drive checkpointing) and are not included in this repo. The model code above is enough to reproduce: instantiate teacher + student, freeze everything except LaCT, optimize the per-layer token alignment loss, save with `lact_state_dict()`.

## Benchmark

Streams `DL3DV/DL3DV-Evaluation` from HuggingFace (one ~9 GB tar at a time, deleted after each scene) so you don't need 500 GB of disk:

```bash
huggingface-cli login   # accept terms at huggingface.co/datasets/DL3DV/DL3DV-Evaluation

python scripts/benchmark_vs_vggt.py \
    --hf_repo DL3DV/DL3DV-Evaluation \
    --hf_cache /tmp/_dl3dv_eval_cache \
    --lact_ckpt /path/to/vggt_ttt_lact_stage1.pt \
    --num_scenes 20 --seq_lens 16,32,48,64,96,128 \
    --json_out bench.json
```

Or pass `--data_dir /path/to/extracted` to use a local copy.

## Loading the model

```python
import torch
from model.vggt_ttt import VGGT_TTT

model = VGGT_TTT.from_pretrained(share_frame_blocks=True, chunk_size=16).cuda().eval()
state = torch.load("vggt_ttt_lact_stage1.pt", map_location="cuda")
model.load_lact_state_dict(state, strict=True)

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    model.reset_memory()              # clear fast weights at scene boundary
    out = model(images)               # (B, V, 3, H, W) → camera/depth/point dict
```

Notes:
- `share_frame_blocks=True` reuses the teacher's frozen frame blocks (no duplicate weights, ~2 GB saved).
- `reset_memory()` clears the LaCT fast-weight state between scenes; the model carries scene state across chunks within a single forward.
- Heads run in fp32; the rest runs in bf16 under autocast.
- `--grad_ckpt` only needed beyond ~128 frames at 518².

## References

- VGGT — https://github.com/facebookresearch/vggt
- LaCT — https://github.com/a1600012888/LaCT
- tttLRM — https://cwchenwang.github.io/tttLRM/
- DL3DV-Evaluation — https://huggingface.co/datasets/DL3DV/DL3DV-Evaluation
