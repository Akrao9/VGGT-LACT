"""
Loss functions for VGGT-TTT training.

Core regression, gradient, and normal losses adapted from
facebook/vggt  training/loss.py  (Apache-2.0).
Distillation and consistency losses are VGGT-TTT-specific.
"""

import logging
from math import ceil, floor

import torch
import torch.nn as nn
import torch.nn.functional as F


# ──────────────────────────────────────────────
# Numerical helpers  (from train_utils/general.py)
# ──────────────────────────────────────────────

def check_and_fix_inf_nan(
    input_tensor: torch.Tensor,
    name: str = "tensor",
    hard_max: float = 100.0,
) -> torch.Tensor:
    """Replace inf/nan with zeros, clamp to [-hard_max, hard_max]."""
    if input_tensor is None:
        return input_tensor
    has_bad = torch.isnan(input_tensor).any() or torch.isinf(input_tensor).any()
    if has_bad:
        logging.warning(f"{name} contains inf/nan — replacing with zeros.")
        input_tensor = torch.where(
            torch.isnan(input_tensor) | torch.isinf(input_tensor),
            torch.zeros_like(input_tensor),
            input_tensor,
        )
    if hard_max is not None:
        input_tensor = input_tensor.clamp(-hard_max, hard_max)
    return input_tensor


def torch_quantile(input, q, dim=None, keepdim=False, interpolation="nearest"):
    """Fast scalar-quantile via torch.kthvalue (no 2**24 limit)."""
    q = float(q)
    assert 0 <= q <= 1
    if (dim_was_none := dim is None):
        dim = 0
        input = input.reshape((-1,) + (1,) * (input.ndim - 1))
    inter = {"nearest": round, "lower": floor, "higher": ceil}[interpolation]
    k = inter(q * (input.shape[dim] - 1)) + 1
    out = torch.kthvalue(input, k, dim, keepdim=True)[0]
    if keepdim:
        return out
    return out.squeeze() if dim_was_none else out.squeeze(dim)


def filter_by_quantile(loss_tensor, valid_range, min_elements=1000, hard_max=100):
    """Keep only values below the *valid_range* quantile."""
    if loss_tensor.numel() <= min_elements:
        return loss_tensor
    if loss_tensor.numel() > 100_000_000:
        idx = torch.randperm(loss_tensor.numel(), device=loss_tensor.device)[:1_000_000]
        loss_tensor = loss_tensor.view(-1)[idx]
    loss_tensor = loss_tensor.clamp(max=hard_max)
    thresh = min(torch_quantile(loss_tensor.detach(), valid_range), hard_max)
    mask = loss_tensor < thresh
    return loss_tensor[mask] if mask.sum() > min_elements else loss_tensor


# ──────────────────────────────────────────────
# Core regression loss  (from VGGT loss.py)
# ──────────────────────────────────────────────

def regression_loss(
    pred, gt, mask,
    conf=None,
    gradient_loss_fn=None,
    gamma=1.0, alpha=0.2,
    valid_range=-1,
):
    """
    Confidence-weighted regression with optional gradient loss.

    loss_conf = γ · ‖pred − gt‖ · conf − α · log(conf)
    Encourages high confidence on easy pixels, low on hard ones.

    Args:
        pred:  (B, S, H, W, C)
        gt:    (B, S, H, W, C)
        mask:  (B, S, H, W)  bool
        conf:  (B, S, H, W)  optional confidence
        gradient_loss_fn: "grad" | "normal" | None
        gamma, alpha: conf loss weights
        valid_range: quantile for outlier filtering, -1 to disable
    """
    bb, ss, hh, ww, nc = pred.shape

    loss_reg = torch.norm(gt[mask] - pred[mask], dim=-1)
    loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg")

    if conf is not None:
        loss_conf = gamma * loss_reg * conf[mask] - alpha * torch.log(conf[mask])
        loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf")
    else:
        loss_conf = loss_reg  # fall back to plain L2

    # Gradient / normal loss
    loss_grad = torch.tensor(0.0, device=pred.device)
    if gradient_loss_fn is not None:
        to_feed_conf = conf.reshape(bb * ss, hh, ww) if (conf is not None and "conf" in gradient_loss_fn) else None
        if "normal" in gradient_loss_fn:
            loss_grad = gradient_loss_multi_scale(
                pred.reshape(bb * ss, hh, ww, nc),
                gt.reshape(bb * ss, hh, ww, nc),
                mask.reshape(bb * ss, hh, ww),
                gradient_loss_fn=normal_loss, scales=3, conf=to_feed_conf,
            )
        elif "grad" in gradient_loss_fn:
            loss_grad = gradient_loss_multi_scale(
                pred.reshape(bb * ss, hh, ww, nc),
                gt.reshape(bb * ss, hh, ww, nc),
                mask.reshape(bb * ss, hh, ww),
                gradient_loss_fn=gradient_loss, conf=to_feed_conf,
            )

    # Aggregate
    if loss_conf.numel() > 0:
        if valid_range > 0:
            loss_conf = filter_by_quantile(loss_conf, valid_range)
        loss_conf = check_and_fix_inf_nan(loss_conf, "loss_conf_agg").mean()
    else:
        loss_conf = (0.0 * pred).mean()

    if loss_reg.numel() > 0:
        if valid_range > 0:
            loss_reg = filter_by_quantile(loss_reg, valid_range)
        loss_reg = check_and_fix_inf_nan(loss_reg, "loss_reg_agg").mean()
    else:
        loss_reg = (0.0 * pred).mean()

    return loss_conf, loss_grad, loss_reg


# ──────────────────────────────────────────────
# Gradient loss  (spatial smoothness)
# ──────────────────────────────────────────────

def gradient_loss_multi_scale(prediction, target, mask, gradient_loss_fn, scales=4, conf=None):
    """Apply gradient_loss_fn at 2^0 … 2^(scales-1) subsampling."""
    total = 0
    for s in range(scales):
        step = 2 ** s
        total += gradient_loss_fn(
            prediction[:, ::step, ::step],
            target[:, ::step, ::step],
            mask[:, ::step, ::step],
            conf=conf[:, ::step, ::step] if conf is not None else None,
        )
    return total / scales


def gradient_loss(prediction, target, mask, conf=None, gamma=1.0, alpha=0.2):
    """L1 gradient loss in x and y directions."""
    mask = mask[..., None].expand(-1, -1, -1, prediction.shape[-1])
    diff = (prediction - target) * mask

    grad_x = torch.abs(diff[:, :, 1:] - diff[:, :, :-1])
    mask_x = mask[:, :, 1:] * mask[:, :, :-1]
    grad_x = (grad_x * mask_x).clamp(max=100)

    grad_y = torch.abs(diff[:, 1:, :] - diff[:, :-1, :])
    mask_y = mask[:, 1:, :] * mask[:, :-1, :]
    grad_y = (grad_y * mask_y).clamp(max=100)

    if conf is not None:
        conf = conf[..., None].expand(-1, -1, -1, prediction.shape[-1])
        grad_x = gamma * grad_x * conf[:, :, 1:] - alpha * torch.log(conf[:, :, 1:])
        grad_y = gamma * grad_y * conf[:, 1:, :] - alpha * torch.log(conf[:, 1:, :])

    M = mask.sum((1, 2, 3))
    divisor = M.sum()
    if divisor == 0:
        return torch.tensor(0.0, device=prediction.device)
    return (grad_x.sum((1, 2, 3)) + grad_y.sum((1, 2, 3))).sum() / divisor


# ──────────────────────────────────────────────
# Surface normal loss  (geometric consistency)
# ──────────────────────────────────────────────

def point_map_to_normal(point_map, mask, eps=1e-6):
    """Convert (B,H,W,3) point map → (4,B,H,W,3) normals via cross products."""
    with torch.amp.autocast(device_type=point_map.device.type, enabled=False):
        padded_mask = F.pad(mask, (1, 1, 1, 1), value=0)
        pts = F.pad(point_map.permute(0, 3, 1, 2), (1, 1, 1, 1), value=0).permute(0, 2, 3, 1)
        c = pts[:, 1:-1, 1:-1, :]
        u, l, d, r = pts[:, :-2, 1:-1, :], pts[:, 1:-1, :-2, :], pts[:, 2:, 1:-1, :], pts[:, 1:-1, 2:, :]
        n1 = torch.cross(u - c, l - c, dim=-1)
        n2 = torch.cross(l - c, d - c, dim=-1)
        n3 = torch.cross(d - c, r - c, dim=-1)
        n4 = torch.cross(r - c, u - c, dim=-1)
        pm = padded_mask
        cc = pm[:, 1:-1, 1:-1]
        v1 = pm[:, :-2, 1:-1] & cc & pm[:, 1:-1, :-2]
        v2 = pm[:, 1:-1, :-2] & cc & pm[:, 2:, 1:-1]
        v3 = pm[:, 2:, 1:-1] & cc & pm[:, 1:-1, 2:]
        v4 = pm[:, 1:-1, 2:] & cc & pm[:, :-2, 1:-1]
        normals = F.normalize(torch.stack([n1, n2, n3, n4]), p=2, dim=-1, eps=eps)
        valids = torch.stack([v1, v2, v3, v4])
    return normals, valids


def normal_loss(prediction, target, mask, cos_eps=1e-8, conf=None, gamma=1.0, alpha=0.2):
    """Cosine-distance loss between surface normals derived from point maps."""
    pred_n, pred_v = point_map_to_normal(prediction, mask, eps=cos_eps)
    gt_n, gt_v = point_map_to_normal(target, mask, eps=cos_eps)
    valid = pred_v & gt_v
    if valid.sum() < 10:
        return torch.tensor(0.0, device=prediction.device)
    dot = (pred_n[valid] * gt_n[valid]).sum(-1).clamp(-1 + cos_eps, 1 - cos_eps)
    loss = 1 - dot
    if loss.numel() < 10:
        return torch.tensor(0.0, device=prediction.device)
    loss = check_and_fix_inf_nan(loss, "normal_loss")
    if conf is not None:
        c = conf[None].expand(4, -1, -1, -1)[valid]
        loss = gamma * loss * c - alpha * torch.log(c)
    return loss.mean()


def local_normal_consistency_loss(point_map, mask, cos_eps=1e-6):
    """
    Encourage neighboring normal estimates from the same point map to agree.

    This is a true smoothness term. It avoids the old self-comparison pattern
    of `pm` vs `pm.detach()`, which produces a near-zero normal loss.
    """
    normals, valids = point_map_to_normal(point_map, mask, eps=cos_eps)
    valid_count = valids.float().sum(dim=0, keepdim=True)
    enough_views = valid_count.expand_as(valids) > 1

    denom = valid_count.clamp_min(1.0)[..., None]
    mean_normal = (normals * valids[..., None]).sum(dim=0, keepdim=True) / denom
    mean_normal = F.normalize(mean_normal, p=2, dim=-1, eps=cos_eps)

    valid = valids & enough_views
    if valid.sum() < 10:
        return torch.tensor(0.0, device=point_map.device)

    target_normal = mean_normal.expand_as(normals)
    dot = (normals[valid] * target_normal[valid]).sum(-1).clamp(-1 + cos_eps, 1 - cos_eps)
    loss = check_and_fix_inf_nan(1 - dot, "local_normal_consistency")
    return loss.mean() if loss.numel() else torch.tensor(0.0, device=point_map.device)


# ──────────────────────────────────────────────
# Camera pose loss  (translation + rotation + focal)
# ──────────────────────────────────────────────

def camera_loss(pred_pose_enc, gt_pose_enc, loss_type="l1"):
    """
    Decomposed camera loss on pose encodings.

    Pose encoding layout (absT_quaR_FoV):
        [:3]  translation
        [3:7] quaternion rotation
        [7:]  focal length / intrinsics

    Returns (loss_T, loss_R, loss_FL) — all scalar.
    """
    if loss_type == "l1":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).abs()
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).abs()
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).abs()
    elif loss_type == "l2":
        loss_T = (pred_pose_enc[..., :3] - gt_pose_enc[..., :3]).norm(dim=-1, keepdim=True)
        loss_R = (pred_pose_enc[..., 3:7] - gt_pose_enc[..., 3:7]).norm(dim=-1)
        loss_FL = (pred_pose_enc[..., 7:] - gt_pose_enc[..., 7:]).norm(dim=-1)
    else:
        raise ValueError(f"Unknown loss type: {loss_type}")

    loss_T = check_and_fix_inf_nan(loss_T, "loss_T").clamp(max=100).mean()
    loss_R = check_and_fix_inf_nan(loss_R, "loss_R").mean()
    loss_FL = check_and_fix_inf_nan(loss_FL, "loss_FL").mean()
    return loss_T, loss_R, loss_FL


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# VGGT-TTT specific losses
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# Both VGGT teacher and VGGT-TTT student now use the same keys:
#   pose_enc, depth, depth_conf, world_points, world_points_conf


def finite_fraction(value: torch.Tensor) -> torch.Tensor:
    """Return fraction of finite elements as a detached scalar tensor."""
    if value.numel() == 0:
        return value.new_tensor(1.0)
    return torch.isfinite(value).float().mean().detach()


def _zero_loss_like(value: torch.Tensor) -> torch.Tensor:
    """A zero scalar connected to value's graph, even when value is nonfinite."""
    return torch.nan_to_num(value, nan=0.0, posinf=0.0, neginf=0.0).sum() * 0.0


def finite_masked_l1_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    nonfinite_penalty: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    L1 over finite teacher/student entries plus a student nonfinite penalty.

    VGGT's supervised loss helpers replace NaNs with zeros after applying
    dataset masks. For distillation, that behavior can make a broken student
    look perfect, so we expose finite rates and penalize nonfinite student
    outputs instead of silently zeroing the scalar loss.
    """
    target = target.detach()
    pred_finite = torch.isfinite(pred)
    target_finite = torch.isfinite(target)
    valid = pred_finite & target_finite

    if valid.any():
        base = F.l1_loss(pred[valid], target[valid])
    else:
        base = _zero_loss_like(pred)

    pred_rate = pred_finite.float().mean()
    target_rate = target_finite.float().mean()
    penalty = pred.new_tensor(nonfinite_penalty) * (1.0 - pred_rate)
    return base + penalty, pred_rate.detach(), target_rate.detach()


def finite_camera_distillation_loss(
    pred_pose_enc: torch.Tensor,
    gt_pose_enc: torch.Tensor,
    nonfinite_penalty: float = 100.0,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Decomposed pose loss with one finite-rate penalty for the student pose."""
    gt_pose_enc = gt_pose_enc.detach()

    def component_loss(pred_part, target_part):
        valid = torch.isfinite(pred_part) & torch.isfinite(target_part)
        if valid.any():
            return F.l1_loss(pred_part[valid], target_part[valid])
        return _zero_loss_like(pred_part)

    loss_T = component_loss(pred_pose_enc[..., :3], gt_pose_enc[..., :3])
    loss_R = component_loss(pred_pose_enc[..., 3:7], gt_pose_enc[..., 3:7])
    loss_FL = component_loss(pred_pose_enc[..., 7:], gt_pose_enc[..., 7:])

    pred_rate = finite_fraction(pred_pose_enc)
    target_rate = finite_fraction(gt_pose_enc)
    pose_loss = loss_T + loss_R + 0.5 * loss_FL
    pose_loss = pose_loss + pred_pose_enc.new_tensor(nonfinite_penalty) * (1.0 - pred_rate)
    return pose_loss, loss_T, loss_R, loss_FL, target_rate


def distillation_loss(
    student_outputs: dict,
    teacher_outputs: dict,
    weight_pose: float = 1.0,
    weight_depth: float = 1.0,
    weight_point: float = 0.5,
    nonfinite_penalty: float = 100.0,
) -> dict:
    """
    Stage 1 distillation: match student predictions to frozen teacher.

    Uses decomposed camera loss + confidence-weighted depth/point regression
    when confidence maps are available, otherwise falls back to L1.

    Returns dict with individual + total loss for logging.
    """
    device = next(
        (v.device for v in student_outputs.values() if isinstance(v, torch.Tensor)),
        torch.device("cpu"),
    )
    losses = {}
    total = torch.tensor(0.0, device=device)

    # ── Pose encoding ───────────────────────────
    s_pose = student_outputs.get("pose_enc")
    t_pose = teacher_outputs.get("pose_enc")
    if s_pose is not None and t_pose is not None:
        pose_loss, loss_T, loss_R, loss_FL, teacher_pose_finite = finite_camera_distillation_loss(
            s_pose,
            t_pose,
            nonfinite_penalty=nonfinite_penalty,
        )
        losses["loss_pose"] = pose_loss
        losses["loss_pose_T"] = loss_T
        losses["loss_pose_R"] = loss_R
        losses["loss_pose_FL"] = loss_FL
        losses["finite_student_pose"] = finite_fraction(s_pose)
        losses["finite_teacher_pose"] = teacher_pose_finite
        total = total + weight_pose * pose_loss

    # ── Depth ───────────────────────────────────
    s_depth = student_outputs.get("depth")
    t_depth = teacher_outputs.get("depth")
    if s_depth is not None and t_depth is not None:
        depth_loss, student_depth_finite, teacher_depth_finite = finite_masked_l1_loss(
            s_depth,
            t_depth,
            nonfinite_penalty=nonfinite_penalty,
        )
        losses["loss_depth"] = depth_loss
        losses["finite_student_depth"] = student_depth_finite
        losses["finite_teacher_depth"] = teacher_depth_finite
        total = total + weight_depth * depth_loss

    # ── World points ────────────────────────────
    s_pts = student_outputs.get("world_points")
    t_pts = teacher_outputs.get("world_points")
    if s_pts is not None and t_pts is not None:
        point_loss, student_point_finite, teacher_point_finite = finite_masked_l1_loss(
            s_pts,
            t_pts,
            nonfinite_penalty=nonfinite_penalty,
        )
        losses["loss_point"] = point_loss
        losses["finite_student_point"] = student_point_finite
        losses["finite_teacher_point"] = teacher_point_finite
        total = total + weight_point * point_loss

    losses["loss_total"] = total
    return losses


def consistency_loss(
    outputs: dict,
    images: torch.Tensor,
    weight_depth_smooth: float = 0.1,
    weight_camera_smooth: float = 0.01,
    weight_normal: float = 0.05,
) -> dict:
    """
    Stage 2 self-supervised consistency loss for long sequences.

    - Adjacent-frame depth smoothness
    - Camera trajectory smoothness
    - Surface normal consistency (if point maps available)
    """
    device = images.device
    losses = {}
    total = torch.tensor(0.0, device=device)

    depth = outputs.get("depth")              # (B, N, H, W, 1)
    extrinsic = outputs.get("extrinsic")      # (B, N, 4, 4)
    world_points = outputs.get("world_points")  # (B, N, H, W, 3)

    if depth is not None and depth.shape[1] >= 2:
        # Depth smoothness between adjacent frames
        # depth may be (B,N,H,W,1) — squeeze trailing dim for L1
        d = depth.squeeze(-1) if depth.ndim == 5 else depth
        d_loss = F.l1_loss(d[:, :-1], d[:, 1:])
        d_loss = check_and_fix_inf_nan(d_loss, "depth_smooth")
        losses["loss_depth_smooth"] = d_loss
        total = total + weight_depth_smooth * d_loss

    if extrinsic is not None and extrinsic.shape[1] >= 2:
        # Camera translation smoothness
        t1 = extrinsic[:, :-1, :3, 3]
        t2 = extrinsic[:, 1:, :3, 3]
        cam_loss = ((t2 - t1) ** 2).mean()
        cam_loss = check_and_fix_inf_nan(cam_loss, "cam_smooth")
        losses["loss_cam_smooth"] = cam_loss
        total = total + weight_camera_smooth * cam_loss

    if world_points is not None and world_points.shape[1] >= 1:
        # Per-frame normal consistency on world points.
        B, N, H, W, C = world_points.shape
        pm = world_points.reshape(B * N, H, W, C)
        mask = torch.ones(B * N, H, W, dtype=torch.bool, device=device)
        n_loss = local_normal_consistency_loss(pm, mask)
        # Normal self-consistency should be near 0 for clean surfaces
        if isinstance(n_loss, torch.Tensor):
            losses["loss_normal"] = n_loss
            total = total + weight_normal * n_loss

    losses["loss_total"] = total
    return losses
