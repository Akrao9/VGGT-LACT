# VGGT-TTT

**VGGT** with **tttLRM-style LaCT aggregator** plus a **virtual-token Gaussian decoder** for novel-view synthesis on long videos. Replaces VGGT's quadratic global attention with `FastWeightGluMLPMultihead` (per-head GLU fast weights) so memory is linear in the frame count and scene state persists across chunks.

## What changed vs vanilla VGGT

- Global attention → LaCT block (frame-wise attention is unchanged).
- 2D RoPE preserved on the LaCT path (matches VGGT global attention).
- Virtual query tokens (Plücker-encoded target rays) are routed through both frame-wise and LaCT layers — the trunk does the mixing, a thin MLP head reads them out as Gaussians.
- Slim LaCT-only checkpoints (~200 MB, Drive-friendly).

## Architecture

```
images ── DINOv2 patch embed (frozen) ──┐
                                        │
target poses → Plücker rays → virtual tokens
                                        │
                ┌───────────────────────┘
                ▼
   ┌─ frame-wise attn (frozen, per-frame, RoPE) ─┐
   │                                              │
   │   ┌─ LaCT (cross-frame, RoPE on q,k) ─┐    │  ×24
   │   │   fast weights w0,w1,w2 persist   │    │
   │   └────────────────────────────────────┘    │
   └──────────────────────────────────────────────┘
                ▼
   camera/depth/point heads (frozen)  &  TTTLRMDecoder → Gaussians → gsplat render
```

## Project layout

| Path | Role |
|---|---|
| `model/lact_ttt_glu.py` | tttLRM `FastWeightGluMLPMultihead` with optional 2D RoPE on q,k |
| `model/lact_block.py` | LayerNorm + GLU TTT + residual; padding-aware mini-batch; optional TBPTT |
| `model/ttt_aggregator.py` | VGGT aggregator with LaCT instead of global; routes `virtual_tokens` |
| `model/vggt_ttt.py` | Top-level model + slim `lact_state_dict()` save/load |
| `model/tttlrm_decoder.py` | Plücker → virtual token seed + thin MLP → Gaussians |
| `model/losses.py` | Distillation + consistency losses (unchanged) |
| `pipeline/input_pipeline.py` | Image/video preprocessing |
| `scripts/script_common.py` | Blackwell-friendly defaults (TF32, fused AdamW, compile guard) |
| `scripts/finetune.py` | Stage 1 distillation / Stage 2 consistency; slim ckpts to Drive |
| `scripts/train_decoder.py` | Stage 3: train `TTTLRMDecoder` + gsplat render |
| `scripts/run_inference.py` | Inference CLI |
| `scripts/evaluate_stage1.py` | Compare student vs frozen VGGT |
| `scripts/dl3dv_streaming.py` | DL3DV-ALL HF download + prefetch queue |

## Quick start

```bash
pip install -e .
# inference
python scripts/run_inference.py --input my_video.mp4 --fps 2 --out ./out
```

## Training (Colab, RTX Pro 6000 96GB)

```bash
# stage 1: LaCT distillation, slim ckpts to Drive
python scripts/finetune.py --stream_dl3dv --stage 1 --epochs 5 --seq_len 32 \
    --save_dir /content/drive/MyDrive/vggt_ttt_ckpts \
    --precision bf16 --compile

# stage 2: long-sequence consistency
python scripts/finetune.py --stream_dl3dv --stage 2 --epochs 10 --seq_len 96 \
    --save_dir /content/drive/MyDrive/vggt_ttt_ckpts

# stage 3: tttLRM Gaussian decoder
python scripts/train_decoder.py --data_dir /path/to/scenes \
    --lact_ckpt /content/drive/MyDrive/vggt_ttt_ckpts/vggt_ttt_lact_stage2.pt \
    --epochs 10 --virtual_grid 24
```

Notes
- `--share_frame_blocks` is on by default — student reuses teacher's frozen frame blocks (no duplicate weights, ~2 GB saved).
- Checkpoints contain only LaCT params (`lact_state_dict()`); reload with `model.load_lact_state_dict()`.
- `reset_memory()` clears fast weights; called automatically per scene boundary in streaming.
- `--tbptt_window N>0` keeps gradients flowing across N chunks.
- `--grad_ckpt` only needed beyond ~128 frames at 518².
- `--grad_accum_steps` > 1 averages gradients over N dataloader steps before `clip_grad_norm_` + `step`; it does **not** reduce peak VRAM per step (each step still runs a full student forward). For VRAM, use `--target_size`, shorter `--seq_len`, `--teacher_device cpu`, and LaCT `--lact_ttt_*_cap` flags. `configure_cuda_backend()` (called by `finetune.py`) sets allocator defaults and prefers flash / memory-efficient SDPA for VGGT frame attention.

## References

- tttLRM: https://cwchenwang.github.io/tttLRM/
- LaCT: https://github.com/a1600012888/LaCT
- VGGT: https://github.com/facebookresearch/vggt
