# VGGT-TTT

A linear-memory drop-in for VGGT's global attention. Replaces the quadratic global-attention layers in [VGGT](https://github.com/facebookresearch/vggt) with a [LaCT](https://github.com/a1600012888/LaCT)-style fast-weight GLU block (per-head `FastWeightGluMLPMultihead`), distilled from the frozen VGGT teacher. Frame count scales linearly in VRAM, scene state persists across chunks via detached fast weights, and the original VGGT prediction heads (camera, depth, world points) ride on top unchanged.

## What changed vs vanilla VGGT

- **Global attention → LaCT block** (frame-wise attention unchanged).
- **2D RoPE preserved** on the LaCT path (matches VGGT's global attn).
- **Cross-chunk fast-weight state** (`w0,w1,w2`) detached between chunks; optional TBPTT window keeps gradients flowing across the last *K* chunks.
- **Slim checkpoints** (`lact_state_dict()`) — only the trainable LaCT params, ~200 MB.

## Results (DL3DV-Evaluation, `--num_scenes 20`)

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

Raw numbers in [`bench.json`](bench.json); reproduce via [`scripts/benchmark_vs_vggt.py`](scripts/benchmark_vs_vggt.py). Each `summary` row’s `n_scenes` is how many scenes contributed at that `N` (default `--num_scenes 20`; fewer if a scene has too few frames or a row is skipped).

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
| `model/io_utils.py` | Safe checkpoint I/O (`torch_load_checkpoint`) |
| `model/losses.py` | `distillation_loss` (stage 1) + `consistency_loss` (stage 2) |
| `pipeline/input_pipeline.py` | Image/video preprocessing |
| `scripts/script_common.py` | CUDA defaults (TF32, fused AdamW, compile guard, AMP helpers) |
| `scripts/run_inference.py` | Inference CLI (video → poses / depth / points) |
| `scripts/upload_vggt_lact_hf.py` | Upload LaCT checkpoint + Hub model card to [`akrao9/VGGT-LACT`](https://huggingface.co/akrao9/VGGT-LACT) |
| `huggingface/VGGT-LACT/README.md` | Model card source for the Hub repo above |
| `scripts/benchmark_vs_vggt.py` | Teacher vs student: VRAM, latency, depth vs teacher, pose vs GT |
| `scripts/finetune.py` | Stage 1 distillation or stage 2 consistency training |
| `scripts/dl3dv_streaming.py` | DL3DV HF tar streaming helpers for `finetune.py` |
| `scripts/download.py` | DL3DV subset download / listing (Hugging Face) |
| `scripts/evaluate_stage1.py` | Held-out `distillation_loss` metrics + optional baseline comparison |
| `configs/default.yaml` | Reference defaults (CLIs use argparse) |

## Install

```bash
git clone https://github.com/Akrao9/vggt_ttt && cd vggt_ttt
pip install -e .
```

Install [VGGT](https://github.com/facebookresearch/vggt) separately (the model loads `facebook/VGGT-1B` from HuggingFace on first call).

## Inference

**Pretrained LaCT (stage 1)** lives on the Hub: [`akrao9/VGGT-LACT`](https://huggingface.co/akrao9/VGGT-LACT) (`vggt_ttt_lact_stage1.pt`, ~200 MB). Example download:

```python
from huggingface_hub import hf_hub_download
ckpt = hf_hub_download("akrao9/VGGT-LACT", "vggt_ttt_lact_stage1.pt")
```

```bash
python scripts/run_inference.py \
    --input my_video.mp4 --fps 2 \
    --checkpoint /path/to/vggt_ttt_lact_stage1.pt \
    --out ./out
```

The slim LaCT checkpoint is loaded on top of `facebook/VGGT-1B`; everything else (frame-wise blocks, prediction heads) comes from the public VGGT release.

## Training (stage 1 distillation)

The LaCT blocks are the only trainable parameters; everything else (DINOv2 patch embed, frame-wise attention, prediction heads) stays frozen at VGGT's pretrained weights. Stage 1 aligns **student head outputs to the frozen teacher** on the same images so the pretrained heads still see statistics they were built for.

**Setup**
- Teacher: frozen `facebook/VGGT-1B`, `eval()`, no grad.
- Student: `VGGT_TTT(teacher, share_frame_blocks=True)` — student reuses the teacher's frame-wise blocks, only the 24 LaCT blocks (replacing global attention) are trainable. LaCT `c_proj` is zero-initialized so the student starts as a near-identity copy of the teacher.
- Both run on the same GPU; with `share_frame_blocks=True` no weights are duplicated.

**Loss** — implemented as `distillation_loss` in [`model/losses.py`](model/losses.py) (each term is skipped if the matching key is missing on either side):

- **`pose_enc`**: decomposed **L1** on the camera pose encoding vs the teacher’s `pose_enc`, plus a penalty when the student predicts non-finite values (`finite_camera_distillation_loss`).
- **`depth` / `world_points`**: **masked L1** vs the teacher on finite entries (`finite_masked_l1_loss`), with the same non-finite penalty pattern.

Weights default to `weight_pose=1`, `weight_depth=1`, `weight_point=0.5`. There is **no** separate token-level cosine/L2 on `aggregated_tokens_list` in this repo’s public loss code.

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
- Only LaCT weights are saved: call **`lact_state_dict()` on your `VGGT_TTT` instance** (same tensors as `torch.save(model.lact_state_dict(), path)`), ~200 MB. Reload with `model.load_lact_state_dict(state, strict=True)` after `VGGT_TTT.from_pretrained(...)`.

Implementation entry point: [`scripts/finetune.py`](scripts/finetune.py) — stage 1 runs paired teacher/student forwards and **`distillation_loss(student_out, teacher_out)`** (see [`model/losses.py`](model/losses.py)).

### Train (local scenes)

Requires an extracted DL3DV-style tree under `--data_dir` (see dataset docs). HuggingFace login is not required for local paths.

```bash
python scripts/finetune.py --data_dir /path/to/extracted_scenes --stage 1 --epochs 5 \
    --save_dir ./checkpoints --chunk_size 16 --seq_len 32
```

### Train (stream DL3DV-ALL from Hugging Face)

Accept the dataset terms, then `huggingface-cli login`. Downloads one scene tar at a time into `--dl3dv_local_dir` (default: `~/.cache/vggt_ttt/dl3dv_stream`).

```bash
python scripts/finetune.py --stream_dl3dv --stage 1 --epochs 5 --seq_len 32 \
    --save_dir ./checkpoints --chunk_size 16 \
    --dl3dv_resolution 480P --stream_scenes_per_epoch 8
```

Use `--teacher_device cpu` if GPU memory is tight (slower teacher forward).

### Evaluate (same loss as training)

[`scripts/evaluate_stage1.py`](scripts/evaluate_stage1.py) reports `distillation_loss` scalars on held-out data. `--compare_baseline` adds an untrained-LaCT run on the same clips for a quick “checkpoint vs random” delta.

```bash
python scripts/evaluate_stage1.py \
    --data_dir /path/to/DL3DV-Evaluation \
    --checkpoint ./checkpoints/vggt_ttt_lact_stage1.pt \
    --compare_baseline --num_scenes 5 --json_out ./eval_stage1.json
```

## Benchmark

Streams [`DL3DV/DL3DV-Evaluation`](https://huggingface.co/datasets/DL3DV/DL3DV-Evaluation) from Hugging Face (one ~9 GB tar at a time, deleted after each scene) so you do not need the full dataset on disk. Default `--hf_cache` is `~/.cache/vggt_ttt/dl3dv_eval`.

```bash
huggingface-cli login   # accept terms at huggingface.co/datasets/DL3DV/DL3DV-Evaluation

python scripts/benchmark_vs_vggt.py \
    --hf_repo DL3DV/DL3DV-Evaluation \
    --hf_cache ~/.cache/vggt_ttt/dl3dv_eval \
    --lact_ckpt /path/to/vggt_ttt_lact_stage1.pt \
    --num_scenes 20 --seq_lens 16,32,48,64,96,128 \
    --json_out bench.json
```

Or pass `--data_dir /path/to/extracted` to use a local copy.

## Loading the model

```python
import torch
from model.io_utils import torch_load_checkpoint
from model.vggt_ttt import VGGT_TTT

model = VGGT_TTT.from_pretrained(share_frame_blocks=True, chunk_size=16).cuda().eval()
state = torch_load_checkpoint("vggt_ttt_lact_stage1.pt", map_location="cuda")
model.load_lact_state_dict(state, strict=True)

with torch.no_grad(), torch.amp.autocast(device_type="cuda", dtype=torch.bfloat16):
    model.reset_memory()              # clear fast weights at scene boundary
    out = model(images)               # (B, V, 3, H, W) → camera/depth/point dict
```

Notes:
- Prefer **`torch_load_checkpoint`** from [`model/io_utils.py`](model/io_utils.py) over raw `torch.load` so checkpoints use **`weights_only=True`** when the PyTorch version supports it.
- `share_frame_blocks=True` reuses the teacher's frozen frame blocks (no duplicate weights, ~2 GB saved).
- `reset_memory()` clears the LaCT fast-weight state between scenes; the model carries scene state across chunks within a single forward.
- Heads run in fp32; the rest runs in bf16 under autocast.
- `--grad_ckpt` only needed beyond ~128 frames at 518².

## License

This project is released under the **Apache License, Version 2.0**. The full text is in [`LICENSE`](LICENSE); copyright and third-party notes are in [`NOTICE`](NOTICE). Some modules cite upstream code (e.g. loss helpers adapted from [VGGT](https://github.com/facebookresearch/vggt), Apache-2.0).

## References

- VGGT — https://github.com/facebookresearch/vggt
- LaCT — https://github.com/a1600012888/LaCT
- tttLRM — https://cwchenwang.github.io/tttLRM/
- DL3DV-Evaluation — https://huggingface.co/datasets/DL3DV/DL3DV-Evaluation
- DL3DV-ALL-480P (streaming train) — https://huggingface.co/datasets/DL3DV/DL3DV-ALL-480P
- LaCT stage-1 weights (this repo) — https://huggingface.co/akrao9/VGGT-LACT
