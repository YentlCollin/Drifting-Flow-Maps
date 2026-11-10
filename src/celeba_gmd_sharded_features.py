"""Experimental feature sharding; never installed into the active trainer.

Encode each generated image and each fixed support image on one rank, then
gather complete feature supports before the unchanged GMD loss. This preserves
support size/order and the differentiable all-gather + DDP scaling convention.
Production GPU parity and throughput must be measured before use in training.
"""
from collections import OrderedDict
from collections.abc import Callable, Mapping

import torch
import torch.distributed as dist
from torch import Tensor
from torch.distributed.nn.functional import all_gather as differentiable_all_gather


def world():
    return dist.get_world_size() if dist.is_initialized() else 1


def gather_features(features: Mapping[str, Tensor], *, differentiable: bool):
    """Pack families by dtype; gather in rank order without casting features."""
    if not features:
        raise ValueError("empty feature dictionary")
    batches = {value.shape[0] for value in features.values()}
    if len(batches) != 1 or any(value.ndim != 3 for value in features.values()):
        raise ValueError("features must share a batch and have [B,tokens,channels] shape")
    if world() == 1:
        return dict(features) if differentiable else {k: v.detach() for k, v in features.items()}
    metadata = [(name, tuple(value.shape), str(value.dtype)) for name, value in features.items()]
    peers = [None] * world()
    dist.all_gather_object(peers, metadata)
    if any(peer != metadata for peer in peers):
        raise ValueError("feature order, local shapes and dtypes must match on all ranks")
    buckets = OrderedDict()
    for name, value in features.items():
        buckets.setdefault(value.dtype, []).append(name)
    output = {}
    for names in buckets.values():
        packed = torch.cat([features[name].flatten(1) for name in names], dim=1).contiguous()
        if differentiable:
            full = torch.cat(differentiable_all_gather(packed), dim=0)
        else:
            shards = [torch.empty_like(packed) for _ in range(world())]
            dist.all_gather(shards, packed.detach())
            full = torch.cat(shards, dim=0)
        offset = 0
        for name in names:
            shape = features[name].shape
            size = shape[1] * shape[2]
            output[name] = full[:, offset:offset + size].reshape(-1, shape[1], shape[2])
            offset += size
    return {name: output[name] for name in features}


def sharded_supports(encode: Callable, generated: Tensor, positive: Tensor, negative: Tensor):
    """Positive/negative inputs are identical complete banks on every rank."""
    size = world()
    rank = dist.get_rank() if dist.is_initialized() else 0
    if positive.shape[0] % size or negative.shape[0] % size:
        raise ValueError("fixed support counts must be divisible by world size")
    def local(images):
        count = images.shape[0] // size
        if count < 1:
            raise ValueError("each rank must receive a nonempty support")
        return images[rank * count:(rank + 1) * count].detach()
    g, p, n = encode(generated, local(positive), local(negative))
    return (gather_features(g, differentiable=True),
            gather_features(p, differentiable=False),
            gather_features(n, differentiable=False))


def global_sharded_mae_convnext_loss(encoder, generated, positive, negative, *, temperatures):
    """Prototype adapter for the existing frozen MAE82 + ConvNeXt13."""
    from src.celeba_pixel_gmd_convnext import extract_convnext_supports, merge_feature_families
    from src.celeba_pixel_gmd_mae_convnext import _extract_mae_supports, drifting_loss_from_complete_features
    encoder.assert_frozen_eval()
    encoder.retain_generated_gradient(generated)
    def encode(g, p, n):
        mae = _extract_mae_supports(encoder.mae, g, p, n)
        convnext = extract_convnext_supports(encoder.convnext, g, p, n,
            generated_chunk_size=8, fixed_chunk_size=32, checkpoint_generated=True)
        return tuple(merge_feature_families(m, c) for m, c in zip(mae, convnext))
    g, p, n = sharded_supports(encode, generated, positive, negative)
    loss, statistics = drifting_loss_from_complete_features(g, p, n, temperatures=temperatures)
    for key, count in (("generated_global", generated.shape[0] * world()),
                       ("positive", positive.shape[0]), ("fixed_negative", negative.shape[0])):
        statistics[f"support/{key}"] = torch.tensor(count, device=loss.device, dtype=torch.float32)
    return loss, statistics
