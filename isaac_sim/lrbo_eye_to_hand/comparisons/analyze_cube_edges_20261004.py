import sys,json,collections
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parent.parent))
from calibration_comparison import load_calibration
root=Path(__file__).resolve().parent.parent
groups=collections.defaultdict(dict)
for p in sorted((root/'comparisons/edge_paths').glob('edge_path_20261004*.json')):
 d=json.loads(p.read_text())
 if d.get('matched') and d.get('calibration_source') in ('OpenCV','LRBO'):
  groups[json.dumps([d['pixels'],d['cube_world_matrix']])][d['calibration_source']]=(p,d)
rows=[]
pair_camera_deltas=[]
def transform(t,p):return p@t[:3,:3].T+t[:3,3]
def rms(p):return float(np.sqrt(np.mean(np.sum(p*p,axis=1)))*1000)
for pair in groups.values():
 if not all(x in pair for x in ('OpenCV','LRBO')):continue
 i=len(rows)//2+1
 pair_cameras=[]
 for source in ('OpenCV','LRBO'):
  p,d=pair[source]; est=np.array(d['estimated_endpoints_base']);ideal=np.array(d['ideal_subsegment_base']);gt=np.array(d['T_base_camera_GT']);saved=load_calibration((root.parent/'calibration' if d['calibration_source']=='OpenCV' else root/'results')/Path(d['calibration_file']).name)['matrix'];cam=transform(np.linalg.inv(saved),est);idealcam=transform(np.linalg.inv(gt),ideal);uv=np.array(d['pixels']);k=np.array(d['intrinsics']);rays=np.column_stack(((uv[:,0]-k[0,2])/k[0,0],(uv[:,1]-k[1,2])/k[1,1],np.ones(2)))
  pair_cameras.append(cam)
  rgb_only=rays*idealcam[:,2,None];rgb_vec=rgb_only-idealcam;depth_vec=cam-rgb_only
  a,b=np.array(d['ideal_full_edge_base']);direction=b-a;points=est[0]+np.linspace(0,1,101)[:,None]*(est[1]-est[0]);t=np.clip((points-a)@direction/(direction@direction),0,1);near=a+t[:,None]*direction
  rows.append(dict(group=i,source=source,file=p.name,edge_id=d['edge_id'],pixel_max=d['pixel_max_error_px'],pixel_angle=d['pixel_angle_deg'],coverage=d['coverage_percent'],edge_rms=d['path_rms_error_mm'],p_errors=d['endpoint_error_mm'],nearest_edge_rms=rms(points-near),perception_endpoint_rms=rms(cam-idealcam),rgb_endpoint_rms=rms(rgb_vec),depth_endpoint_rms=rms(depth_vec),depth_bias_mm=((cam[:,2]-idealcam[:,2])*1000).tolist(),calibration_endpoint_rms=rms(est-transform(gt,cam)),length_bias=d['estimated_length_mm']-d['ideal_subsegment_length_mm']))
 pair_camera_deltas.append(float(np.max(np.linalg.norm(pair_cameras[0]-pair_cameras[1],axis=1))*1000))
print(json.dumps(rows,indent=2))
(root/'comparisons/edge_error_analysis_20261004.json').write_text(json.dumps(rows,indent=2)+'\n')
lines=['# 偵測方塊邊緣與實際稜邊誤差分析（2026-10-04）','','參考為 Isaac Sim cube 真實幾何與 camera pose GT；僅用於分析，不改動控制路徑。配對具有相同 pixel endpoints 與方塊姿態；不一定是完全相同 snapshot。','','## 影像與 3D 幾何比較','','| 組 | 方法 | 影像最大偏差 px | 影像角度差 ° | 對應稜邊 RMS mm | 到實際稜邊最近距離 RMS mm | P1 / P2 誤差 mm |','|---|---|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['group']} | {r['source']} | {r['pixel_max']:.3f} | {r['pixel_angle']:.3f} | {r['edge_rms']:.3f} | {r['nearest_edge_rms']:.3f} | {r['p_errors'][0]:.3f} / {r['p_errors'][1]:.3f} |")
lines+=['','對應稜邊 RMS：比較同一段可見稜邊上的對應點，包括沿稜邊方向偏差。最近距離 RMS：計算每點到真實完整線段的距離，较直接表示是否偏離實際稜邊；不衡量沿邊端點位置是否正確。','','## 誤差來源分解','','| 組 | 方法 | RGB ray 端點 RMS mm | 深度增量端點 RMS mm | 感知總端點 RMS mm | 外參增量端點 RMS mm | Z bias P1 / P2 mm |','|---|---|---:|---:|---:|---:|---|']
for r in rows:lines.append(f"| {r['group']} | {r['source']} | {r['rgb_endpoint_rms']:.3f} | {r['depth_endpoint_rms']:.3f} | {r['perception_endpoint_rms']:.3f} | {r['calibration_endpoint_rms']:.3f} | {r['depth_bias_mm'][0]:+.3f} / {r['depth_bias_mm'][1]:+.3f} |")
lines+=['','RGB ray：原偵測 pixel ray 使用對應理想稜邊的真實 Z，隔離影像端點偏差。深度增量：將同一 ray 的真實 Z 換成偵測估計 Z。感知總量：重建 camera 3D endpoints 與真實 camera endpoints 的差。外參增量：同一 camera 3D 點分別使用已存外參和 GT 轉換的差。各欄是向量分解後的 RMS，不能把標量直接相加；端點 RMS 也不同於整段 RMS。','','## 來源報告','']
for r in rows:lines.append(f"- 組 {r['group']} {r['source']}: {r['file']}")
lines += ['', '## 更新統計', '',
          f'共 {len(rows)//2} 組配對；camera endpoint 配對差異最大 {max(pair_camera_deltas,default=0):.3f} mm。', '',
          '| 指標（每組等權重） | OpenCV | LRBO |', '|---|---:|---:|']
for metric,label in [('nearest_edge_rms','到實際稜邊 RMS 中位數 mm'),('nearest_edge_rms','到實際稜邊 RMS 平均 mm'),('edge_rms','對應稜邊 RMS 中位數 mm'),('pixel_max','影像最大偏差中位數 px'),('depth_endpoint_rms','深度端點增量 RMS 中位數 mm')]:
    values=[]
    for source in ('OpenCV','LRBO'):
        data=[r[metric] for r in rows if r['source']==source]
        values.append(float(np.mean(data) if '平均' in label else np.median(data)))
    lines.append(f'| {label} | {values[0]:.3f} | {values[1]:.3f} |')
for metric,label in [('nearest_edge_rms','到稜邊最近距離'),('edge_rms','對應位置')]:
    wins=sum(rows[i][metric]<rows[i+1][metric] for i in range(0,len(rows),2))
    lines.append(f'\n{label}：OpenCV 較小 {wins} 組；LRBO 較小 {len(rows)//2-wins} 組（含數值相同情況）。')
lines += ['', '大誤差組（任一方法到稜邊 RMS > 10 mm，此門檻僅用於描述異常）：']
for i in range(0,len(rows),2):
    a,b=rows[i:i+2]
    if max(a['nearest_edge_rms'],b['nearest_edge_rms'])>10:
        lines.append(f"- 組 {a['group']}：OpenCV / LRBO 最近距離 RMS {a['nearest_edge_rms']:.2f} / {b['nearest_edge_rms']:.2f} mm；影像偏差 {a['pixel_max']:.2f} px；深度增量端點 RMS {a['depth_endpoint_rms']:.2f} / {b['depth_endpoint_rms']:.2f} mm。")
lines += ['', '所有樣本均保留於統計，沒有刪掉大誤差組。方法只替換外參，感知數據基本共用，因此共同異常需要先查選邊與深度。這些為規劃幾何誤差，不包含實際手臂追蹤。']
(root/'comparisons/edge_error_analysis_20261004.md').write_text('\n'.join(lines)+'\n')
