"""Independent simulator cube-edge reference; never corrects estimated paths."""
from __future__ import annotations

import itertools
import numpy as np

from edge_tip_path import finite, transform_points, tool_orientation, unit

MAX_MATCH_PIXEL_ERROR = 8.0
MAX_MATCH_ANGLE_DEG = 15.0


def project(points, camera_from_base, k):
    camera = transform_points(camera_from_base, points)
    if np.any(camera[:,2] <= 0):
        return None, camera
    uv = camera[:,:2] / camera[:,2,None]
    return uv * np.array([k[0,0],k[1,1]]) + np.array([k[0,2],k[1,2]]), camera


def point_segment_distances(points, first, second):
    delta = second-first
    denom = np.dot(delta,delta)
    if denom < 1e-12:
        return np.full(len(points),np.inf), np.zeros(len(points))
    t = np.clip((points-first) @ delta / denom,0,1)
    return np.linalg.norm(points-(first+t[:,None]*delta),axis=1),t


def cube_edges(cube_from_local, base_from_world, camera_world, local_cube_size):
    """USD matrix already includes cube scale: local size must NOT be meters again."""
    h = local_cube_size/2
    vertices = np.array(list(itertools.product((-h,h),repeat=3)))
    world = transform_points(cube_from_local,vertices)
    base = transform_points(base_from_world,world)
    camera_local = transform_points(np.linalg.inv(cube_from_local),camera_world)
    edges=[]
    for i,j in itertools.combinations(range(8),2):
        difference = np.abs(vertices[i]-vertices[j]) > 1e-9
        if np.count_nonzero(difference) != 1:
            continue
        varying = int(np.flatnonzero(difference)[0])
        fixed=[axis for axis in range(3) if axis!=varying]
        # Convex cube edge is visible if either adjacent face faces camera.
        visible=any(np.sign(vertices[i,axis])*camera_local[axis] > h for axis in fixed)
        edges.append({'id':f'edge_{i}_{j}', 'axis':varying, 'base':base[[i,j]],
                      'world':world[[i,j]], 'visible':visible})
    return edges


def evaluate_path(plan, cube_world, local_cube_size, camera_gt, intrinsics,
                  tool_axis, roll_axis, normal, clearance, tilt_deg):
    """Match in image space using independent camera GT, report geometric errors.

    Compare the same detected subsegment, not Hough endpoints against full cube
    vertices. Perspective-correct correspondence is fixed by pixel projection;
    observed depth is never used to choose ideal point positions.
    """
    world_base=plan['world_base']
    world_camera=world_base @ camera_gt
    edges=cube_edges(cube_world,np.linalg.inv(world_base),world_camera[:3,3],local_cube_size)
    observed_uv=np.asarray(plan['pixels'],dtype=float)
    observed_dir=unit(observed_uv[1]-observed_uv[0])
    candidates=[]
    camera_from_base=np.linalg.inv(camera_gt)
    for edge in edges:
        uv, camera=project(edge['base'],camera_from_base,intrinsics)
        if uv is None or not edge['visible'] or np.linalg.norm(uv[1]-uv[0])<2:
            continue
        angle=float(np.degrees(np.arccos(np.clip(abs(np.dot(unit(uv[1]-uv[0]),observed_dir)),0,1))))
        distances,s=point_segment_distances(observed_uv,uv[0],uv[1])
        score=float(np.sqrt(np.mean(distances**2))) + .15*angle
        candidates.append((score,edge,uv,camera,s,distances,angle))
    if not candidates:
        return {'matched':False,'reason':'No visible reference cube edge projects into camera'}
    _,edge,uv,camera,s,distances,angle=min(candidates,key=lambda c:c[0])
    pixel_max=float(np.max(distances))
    if pixel_max>MAX_MATCH_PIXEL_ERROR or angle>MAX_MATCH_ANGLE_DEG:
        return {'matched':False,'reason':'RGB candidate does not reliably match a visible physical cube edge',
                'nearest_edge_id':edge['id'],'pixel_max_error_px':pixel_max,'pixel_angle_deg':angle}
    z1,z2=camera[:,2]
    t=s*z1/(z2*(1-s)+s*z1)
    corresponding=edge['base'][0] + t[:,None]*(edge['base'][1]-edge['base'][0])
    if np.linalg.norm(corresponding[1]-corresponding[0])<.005:
        return {'matched':False,'reason':'Matched reference subsegment is too short'}
    steps=np.linspace(0,1,len(plan['tip']))[:,None]
    ideal_edge=corresponding[0]+steps*(corresponding[1]-corresponding[0])
    ideal_tip=ideal_edge+clearance*unit(normal)
    midpoint=np.mean(corresponding,axis=0)
    cube_center=transform_points(np.linalg.inv(world_base),cube_world[:3,3])
    ideal_rotation,ideal_direction,ideal_tool=tool_orientation(
        corresponding[1]-corresponding[0],normal,tool_axis,roll_axis,tilt_deg,cube_center-midpoint)
    ideal_tcp=ideal_tip-ideal_rotation@plan['offset']
    estimated_edge=plan['base'][0]+steps*(plan['base'][1]-plan['base'][0])
    errors=np.linalg.norm(estimated_edge-ideal_edge,axis=1)*1000
    tcp_errors=np.linalg.norm(plan['tcp']-ideal_tcp,axis=1)*1000
    direction_error=float(np.degrees(np.arccos(np.clip(np.dot(plan['edge_direction'],ideal_direction),-1,1))))
    rotation_error=float(np.degrees(np.arccos(np.clip((np.trace(plan['rotation'].T@ideal_rotation)-1)/2,-1,1))))
    # Isolate offset/pose amplification by also applying estimated orientation to ideal tip.
    ideal_tcp_same_rotation=ideal_tip-plan['rotation']@plan['offset']
    return {'matched':True, 'edge_id':edge['id'], 'pixel_max_error_px':pixel_max,
            'pixel_angle_deg':angle, 'coverage_percent':float(abs(t[1]-t[0])*100),
            'edge_direction_error_deg':direction_error,'tool_orientation_error_deg':rotation_error,
            'endpoint_error_mm':errors[[0,-1]].tolist(), 'path_rms_error_mm':float(np.sqrt(np.mean(errors**2))),
            'path_max_error_mm':float(np.max(errors)),
            'endpoint_delta_est_minus_ideal_mm':((plan['base']-corresponding)*1000).tolist(),
            'estimated_length_mm':float(np.linalg.norm(plan['base'][1]-plan['base'][0])*1000),
            'ideal_subsegment_length_mm':float(np.linalg.norm(corresponding[1]-corresponding[0])*1000),
            'ideal_full_edge_length_mm':float(np.linalg.norm(edge['base'][1]-edge['base'][0])*1000),
            'tcp_rms_error_mm':float(np.sqrt(np.mean(tcp_errors**2))),
            'tcp_same_orientation_rms_mm':float(np.sqrt(np.mean(np.sum((plan['tcp']-ideal_tcp_same_rotation)**2,axis=1)))*1000),
            'ideal_full_edge_base':edge['base'].tolist(), 'ideal_subsegment_base':corresponding.tolist(),
            'ideal_tip_base':ideal_tip.tolist(), 'ideal_tcp_base':ideal_tcp.tolist(),
            'ideal_projected_uv':uv.tolist(), 'ideal_rotation':ideal_rotation.tolist(),
            'reference':'Isaac cube geometry + camera pose GT; used for evaluation only',
            'calibration_source':plan['calibration_source'],
            'calibration_file':plan['calibration_file'], 'pixels':np.asarray(plan['pixels']).tolist(),
            'estimated_endpoints_base':plan['base'].tolist(),
            'estimated_tip_base':plan['tip'].tolist(), 'estimated_tcp_base':plan['tcp'].tolist(),
            'estimated_rotation':plan['rotation'].tolist(),
            'T_world_base':world_base.tolist(), 'T_base_camera_GT':camera_gt.tolist(),
            'intrinsics':np.asarray(intrinsics).tolist(), 'cube_world_matrix':cube_world.tolist(),
            'edge_clearance_m':clearance, 'tool_tilt_deg':tilt_deg,
            'TCP_to_tip_offset_m':plan['offset'].tolist()}


def evaluation_text(report):
    if not report['matched']:
        return 'Ideal-path comparison unavailable: '+report['reason']
    return (f"Matched {report['edge_id']}; visible edge coverage: {report['coverage_percent']:.1f}%\n"
            f"P1/P2 error: {report['endpoint_error_mm'][0]:.2f} / {report['endpoint_error_mm'][1]:.2f} mm\n"
            f"Edge/tip path RMS / max: {report['path_rms_error_mm']:.2f} / {report['path_max_error_mm']:.2f} mm\n"
            f"Edge direction / tool orientation: {report['edge_direction_error_deg']:.3f} / {report['tool_orientation_error_deg']:.3f} deg\n"
            f"TCP RMS (ideal orientation): {report['tcp_rms_error_mm']:.2f} mm; "
            f"same orientation: {report['tcp_same_orientation_rms_mm']:.2f} mm\n"
            f"Length estimated / ideal segment / full edge: {report['estimated_length_mm']:.1f} / "
            f"{report['ideal_subsegment_length_mm']:.1f} / {report['ideal_full_edge_length_mm']:.1f} mm\n"
            "Geometry comparison only; robot tracking is not measured.")


def draw_ideal_reference(stage, plan, report):
    if not report['matched']:
        return
    from pxr import UsdGeom,Gf
    for name,key,color,width in (
            ('IdealFullEdge','ideal_full_edge_base',(0.8,0.8,0.8),.0015),
            ('IdealMatchedEdge','ideal_subsegment_base',(1,0,1),.002),
            ('IdealTipPath','ideal_tip_base',(.2,.4,1),.002),
            ('IdealTCPPath','ideal_tcp_base',(.3,1,.2),.002)):
        points=transform_points(plan['world_base'],np.asarray(report[key]))
        curve=UsdGeom.BasisCurves.Define(stage,'/CubeEdgePathPreview/'+name)
        curve.CreateTypeAttr('linear'); curve.CreateWrapAttr('nonperiodic')
        curve.CreateCurveVertexCountsAttr([len(points)])
        curve.CreatePointsAttr([Gf.Vec3f(*map(float,p)) for p in points])
        curve.CreateWidthsAttr([width]); curve.SetWidthsInterpolation('constant')
        curve.CreateDisplayColorAttr([Gf.Vec3f(*color)])
