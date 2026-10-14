"""Offline LRBO solve using the existing isolated PREDATOR worker."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile
import numpy as np
from core import validate_transform


def accept_results(matrices, fitness, rmse):
    matrices = np.array([validate_transform(t) for t in matrices])
    fitness, rmse = np.asarray(fitness), np.asarray(rmse)
    if not len(matrices) or fitness.shape != (len(matrices),) or rmse.shape != fitness.shape:
        raise ValueError('Invalid worker output')
    if not np.isfinite(fitness).all() or not np.isfinite(rmse).all() or np.any(fitness < .9) or np.any(rmse > .025) or np.any(rmse < 0):
        raise ValueError('Quality gate failed: require fitness >=0.9, RMSE <=5 mm (after /5 scaling)')
    for i, x in enumerate(matrices):
        for y in matrices[i+1:]:
            angle = np.rad2deg(np.arccos(np.clip((np.trace(x[:3, :3] @ y[:3, :3].T)-1)/2, -1, 1)))
            if np.linalg.norm(x[:3, 3]-y[:3, 3]) > .05 or angle > 5:
                raise ValueError('Scans disagree by >50 mm or >5 degrees')
    u, _, vt = np.linalg.svd(matrices[:, :3, :3].mean(axis=0))
    result = np.eye(4)
    result[:3, :3] = u @ np.diag([1, 1, np.linalg.det(u@vt)]) @ vt
    result[:3, 3] = matrices[:, :3, 3].mean(axis=0)
    return validate_transform(result)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('session', type=Path)
    p.add_argument('--root', type=Path, default=Path(os.environ.get('LRBO_ROOT', Path.home()/'LRBO')))
    p.add_argument('--python', default=os.environ.get('LRBO_PYTHON', str(Path.home()/'miniforge3/envs/lrbo-predator-gpu/bin/python')))
    p.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    a = p.parse_args()
    scans = sorted(a.session.glob('scan_*.npz'))
    if not scans:
        p.error('Session contains no scans')
    frames = set()
    for path in scans:
        with np.load(path, allow_pickle=False) as data:
            frames.add((str(data['base_frame']), str(data['camera_frame'])))
            if 'model_points_base_m' not in data:
                p.error('Each scan must contain pose-matched CAD points')
    if len(frames) != 1:
        p.error('Scan frames differ')
    for name in ('T_base_camera.npy', 'result.json'):
        (a.session/name).unlink(missing_ok=True)
    root = a.root.resolve()
    worker = Path(__file__).resolve().with_name('predator_worker.py')
    with tempfile.TemporaryDirectory(prefix='real_lrbo_') as tmp:
        # Worker requires a legacy model argument even when each scan embeds its CAD.
        model = Path(tmp)/'model.txt'
        with np.load(scans[0]) as data:
            np.savetxt(model, data['model_points_base_m']*5)
        raw = Path(tmp)/'worker.npz'
        command = [a.python, str(worker), '--predator-root', str(root/'third_party/OverlapPredator'),
                   '--config', str(root/'implementation/registration/test/indoor.yaml'),
                   '--checkpoint', os.environ.get('LRBO_CHECKPOINT', str(root/'weights/registration/model_best_recall.pth')),
                   '--model', str(model), '--scans', *[str(s.resolve()) for s in scans],
                   '--device', a.device, '--output', str(raw)]
        env = dict(os.environ)
        for name in ('PYTHONPATH', 'PYTHONHOME', 'LD_LIBRARY_PATH', 'LD_PRELOAD'):
            env.pop(name, None)
        subprocess.run(command, env=env, check=True, timeout=900)
        with np.load(raw) as data:
            output = {key: data[key].copy() for key in data.files}
        np.savez_compressed(a.session/'lrbo_diagnostics.npz', **output)
        result = accept_results(output['T_base_camera'], output['fitness'], output['inlier_rmse'])
        base, camera = next(iter(frames))
        report = dict(method='LRBO', base_frame=base, camera_frame=camera,
                      T_base_camera=result.tolist(), scans=[s.name for s in scans],
                      fitness=output['fitness'].tolist(), rmse_m=(output['inlier_rmse']/5).tolist())
        np.save(a.session/'T_base_camera.npy', result)
        (a.session/'result.json').write_text(json.dumps(report, indent=2))
        print(result)


if __name__ == '__main__':
    main()
