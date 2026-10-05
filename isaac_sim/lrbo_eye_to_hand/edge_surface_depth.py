"""Estimate depth at edge pixel rays from a measured one-sided planar strip.

For a pinhole camera viewing a plane, inverse optical depth is affine in (u,v).
No cube dimensions, fixed Z, extrinsics or simulator GT are used here.
"""
import numpy as np

SURFACE_STRIP_WIDTH_PX = 10
SURFACE_DEPTH_INLIER_M = .003
SURFACE_MIN_INLIER_RATIO = .75
SURFACE_MIN_POINTS = 20
SURFACE_RANSAC_TRIALS = 120


def fit_edge_surface_depth(depth, valid_mask, first, second):
    first, second = np.asarray(first,dtype=float), np.asarray(second,dtype=float)
    delta = second-first
    length = np.linalg.norm(delta)
    if length < 5:
        raise ValueError('Edge too short to fit a local depth surface')
    x0,y0 = np.maximum(0,np.floor(np.minimum(first,second)-SURFACE_STRIP_WIDTH_PX-1)).astype(int)
    x1,y1 = np.minimum([depth.shape[1],depth.shape[0]],
                       np.ceil(np.maximum(first,second)+SURFACE_STRIP_WIDTH_PX+2)).astype(int)
    valid_mask = np.asarray(valid_mask, dtype=bool) & np.isfinite(depth) & (depth > 0)
    ys,xs = np.nonzero(valid_mask[y0:y1,x0:x1])
    uv = np.column_stack((xs+x0,ys+y0)).astype(float)
    z = depth[ys+y0,xs+x0].astype(float)
    tangent = delta/length
    normal = np.array([-tangent[1],tangent[0]])
    along = (uv-first)@tangent/length
    across = (uv-first)@normal
    solutions=[]
    rng=np.random.default_rng(0)
    origin=(first+second)/2
    scale=max(length,1.)
    for side in (-1,1):
        use=(side*across>=1)&(side*across<=SURFACE_STRIP_WIDTH_PX)&(along>=-.08)&(along<=1.08)
        points,values=uv[use],z[use]
        if len(values)<SURFACE_MIN_POINTS:
            continue
        design=np.column_stack(((points-origin)/scale,np.ones(len(points))))
        inverse=1/values
        best=None
        for _ in range(SURFACE_RANSAC_TRIALS):
            indices=rng.choice(len(values),3,replace=False)
            if np.linalg.matrix_rank(design[indices])<3:
                continue
            coeff=np.linalg.solve(design[indices],inverse[indices])
            predicted_inv=design@coeff
            prediction=np.full(len(values),np.inf)
            positive=predicted_inv>0
            prediction[positive]=1/predicted_inv[positive]
            residual=np.abs(prediction-values)
            inliers=residual<=SURFACE_DEPTH_INLIER_M
            score=(int(np.sum(inliers)),-float(np.median(residual[inliers])) if np.any(inliers) else -np.inf)
            if best is None or score>best[0]: best=(score,inliers)
        if best is None:
            continue
        inliers=best[1]
        if np.mean(inliers)<SURFACE_MIN_INLIER_RATIO or np.sum(inliers)<SURFACE_MIN_POINTS:
            continue
        # Refine only measured inlier depths.
        coeff=np.linalg.lstsq(design[inliers],inverse[inliers],rcond=None)[0]
        residual=np.abs(1/(design@coeff)-values)
        inliers=residual<=SURFACE_DEPTH_INLIER_M
        if np.mean(inliers)<SURFACE_MIN_INLIER_RATIO:
            continue
        if any(np.min(np.linalg.norm(points[inliers]-p,axis=1))>SURFACE_STRIP_WIDTH_PX+2 for p in (first,second)):
            continue  # do not extrapolate from an unrelated middle patch
        coverage=np.histogram(np.clip(((points[inliers]-first)@tangent)/length,0,1),bins=8,range=(0,1))[0]
        if np.count_nonzero(coverage)<7:
            continue
        rms=float(np.sqrt(np.mean(residual[inliers]**2)))
        solutions.append((float(np.mean(inliers)), -rms, coeff, side, int(np.sum(inliers)), points[inliers]))
    if not solutions:
        raise ValueError('No reliable one-sided depth surface near edge; reposition/re-detect')
    ratio,neg_rms,coeff,side,count,points=max(solutions,key=lambda item:(item[0],item[1],item[4]))
    def ray_depth(pixels):
        pixels=np.asarray(pixels,dtype=float)
        inverse=np.column_stack(((pixels-origin)/scale,np.ones(len(pixels))))@coeff
        if np.any(~np.isfinite(inverse)) or np.any(inverse<=0):
            raise ValueError('Invalid edge-ray depth from measured surface')
        return 1/inverse
    return ray_depth, {'method':'measured one-sided plane / original pixel ray',
                       'surface_inlier_ratio':ratio,'surface_rms_m':-neg_rms,
                       'surface_inlier_count':count,'surface_side':side,
                       'inverse_depth_coefficients':coeff.tolist(),
                       'inverse_depth_pixel_origin':origin.tolist(),
                       'inverse_depth_pixel_scale':scale}
