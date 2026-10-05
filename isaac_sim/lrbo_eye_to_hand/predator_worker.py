"""Isolated PREDATOR inference worker for the Isaac Sim LRBO frontend."""

from __future__ import annotations

import argparse
import sys
import random
import time
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from torch.utils.data import Dataset
import open3d as o3d


def _load_predator(repo_root, config_path, checkpoint_path, device="cuda"):
    repo_root = Path(repo_root).resolve()
    sys.path.insert(0, str(repo_root))

    from datasets.dataloader import get_dataloader
    from lib.utils import load_config
    from models.architectures import KPFCNN

    config = EasyDict(load_config(str(config_path)))
    config.architecture = ["simple", "resnetb"]
    for _ in range(config.num_layers - 1):
        config.architecture.extend(["resnetb_strided", "resnetb", "resnetb"])
    for _ in range(config.num_layers - 2):
        config.architecture.extend(["nearest_upsample", "unary"])
    config.architecture.extend(["nearest_upsample", "last_unary"])
    config.n_points = int(config.n_points)
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Check GPU access in the launch terminal, "
                           "or set LRBO_DEVICE=cpu for slower diagnostic inference.")
    config.device = torch.device(device)

    model = KPFCNN(config).to(config.device)
    checkpoint = torch.load(
        str(checkpoint_path),
        map_location="cpu",
        weights_only=True,
    )
    state_dict = checkpoint.get("state_dict", checkpoint)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    return config, model, get_dataloader


class _PairDataset(Dataset):
    def __init__(self, config, source, target):
        self.config = config
        self.source = np.asarray(source, dtype=np.float32)
        self.target = np.asarray(target, dtype=np.float32)

    def __len__(self):
        return 1

    def __getitem__(self, _index):
        source_features = np.ones((len(self.source), 1), dtype=np.float32)
        target_features = np.ones((len(self.target), 1), dtype=np.float32)
        identity = np.eye(3, dtype=np.float32)
        translation = np.zeros((3, 1), dtype=np.float32)
        correspondences = torch.zeros((1, 2), dtype=torch.long)
        return (
            self.source,
            self.target,
            source_features,
            target_features,
            identity,
            translation,
            correspondences,
            self.source,
            self.target,
            torch.ones(1),
        )


def _sample_by_score(points, features, scores, count):
    if points.shape[0] <= count:
        return points, features
    probabilities = np.asarray(scores, dtype=np.float64).reshape(-1)
    probabilities = np.clip(probabilities, 0.0, None)
    total = probabilities.sum()
    if not np.isfinite(total) or total <= 0.0:
        indices = np.random.choice(points.shape[0], size=count, replace=False)
    else:
        # Tiny positive floor allows sampling when fewer than count scores are nonzero.
        probabilities = np.maximum(probabilities, np.finfo(np.float64).eps)
        probabilities /= probabilities.sum()
        indices = np.random.choice(
            points.shape[0],
            size=count,
            replace=False,
            p=probabilities,
        )
    return points[indices], features[indices]


def _register_pair(config, model, get_dataloader, source, target):
    from lib.benchmark_utils import to_o3d_pcd

    dataset = _PairDataset(config, source, target)
    loader, _neighborhood_limits = get_dataloader(
        dataset=dataset,
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )
    inputs = next(iter(loader))
    for key, value in tuple(inputs.items()):
        if isinstance(value, list):
            inputs[key] = [item.to(config.device) for item in value]
        elif isinstance(value, torch.Tensor):
            inputs[key] = value.to(config.device)

    with torch.inference_mode():
        features, overlap_scores, saliency_scores = model(inputs)

    source_length = int(inputs["stack_lengths"][0][0].item())
    points = inputs["points"][0]
    source_points = points[:source_length]
    target_points = points[source_length:]
    source_features = features[:source_length].detach().cpu().numpy()
    target_features = features[source_length:].detach().cpu().numpy()
    source_scores = (
        overlap_scores[:source_length] * saliency_scores[:source_length]
    ).detach().cpu().numpy()
    target_scores = (
        overlap_scores[source_length:] * saliency_scores[source_length:]
    ).detach().cpu().numpy()

    source_points, source_features = _sample_by_score(
        source_points.detach().cpu().numpy(),
        source_features,
        source_scores,
        config.n_points,
    )
    target_points, target_features = _sample_by_score(
        target_points.detach().cpu().numpy(),
        target_features,
        target_scores,
        config.n_points,
    )

    registration = o3d.pipelines.registration
    source_cloud = to_o3d_pcd(source_points)
    target_cloud = to_o3d_pcd(target_points)
    source_descriptor = registration.Feature()
    target_descriptor = registration.Feature()
    source_descriptor.data = np.asarray(source_features, dtype=np.float64).T
    target_descriptor.data = np.asarray(target_features, dtype=np.float64).T

    initial = registration.registration_ransac_based_on_feature_matching(
        source_cloud,
        target_cloud,
        source_descriptor,
        target_descriptor,
        False,
        0.05,
        registration.TransformationEstimationPointToPoint(False),
        3,
        [
            registration.CorrespondenceCheckerBasedOnEdgeLength(0.9),
            registration.CorrespondenceCheckerBasedOnDistance(0.05),
        ],
        registration.RANSACConvergenceCriteria(50000, 0.999),
    )
    return np.asarray(initial.transformation, dtype=np.float64)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--predator-root", required=True, type=Path)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--scans", nargs="+", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--device", choices=("cuda", "cpu"), default="cuda")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"PREDATOR checkpoint not found: {args.checkpoint}")
    if not args.model.is_file():
        raise FileNotFoundError(f"LRBO base model not found: {args.model}")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    o3d.utility.random.seed(args.seed)
    torch.set_num_threads(min(8, torch.get_num_threads()))
    started = time.monotonic()
    config, model, get_dataloader = _load_predator(
        args.predator_root,
        args.config,
        args.checkpoint,
        args.device,
    )
    target = np.loadtxt(args.model, dtype=np.float32)[:, :3]
    from lrbo_core import run_lrbo_registration

    matrices = []
    model_matrices = []
    fitness = []
    inlier_rmse = []
    source_counts = []
    target_counts = []
    previous_transform = None
    for scan_path in args.scans:
        with np.load(scan_path) as scan:
            source = scan["points_camera_m"]
            pair_target = (scan["model_points_base_m"] * 5.0
                           if "model_points_base_m" in scan else target)
        result = run_lrbo_registration(
            source,
            pair_target,
            lambda scaled_source, scaled_target: _register_pair(
                config,
                model,
                get_dataloader,
                scaled_source,
                scaled_target,
            ),
            initial_candidates=(() if previous_transform is None else (previous_transform,)),
        )
        if result["fitness"] >= 0.9 and result["inlier_rmse_m"] <= 0.005:
            previous_transform = result["T_model_from_source_scaled"].copy()
        matrices.append(result["T_base_camera"])
        model_matrices.append(result["T_model_from_source_scaled"])
        fitness.append(result["fitness"])
        inlier_rmse.append(result["inlier_rmse"])
        source_counts.append(result["source_points"])
        target_counts.append(result["target_points"])
        print(
            f"registered {scan_path.name}: fitness={result['fitness']:.4f}, "
            f"rmse={result['inlier_rmse']:.5f}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        T_base_camera=np.stack(matrices),
        T_model_from_source_scaled=np.stack(model_matrices),
        fitness=np.asarray(fitness, dtype=np.float64),
        inlier_rmse=np.asarray(inlier_rmse, dtype=np.float64),
        source_points=np.asarray(source_counts, dtype=np.int32),
        target_points=np.asarray(target_counts, dtype=np.int32),
    )
    print(f"saved {len(matrices)} registrations to {args.output}; "
          f"elapsed={time.monotonic() - started:.2f}s; device={args.device}")


if __name__ == "__main__":
    main()
