# 方塊單次偵測與校正比較

沿邊完成後追加固定 tip 的姿態旋轉：工具由對桌面 45° 轉為垂直向下，tip 保持在沿邊終點（包含 20 mm clearance）。TCP 每一步使用 `tip - R @ d_tcp_tip` 補償，預覽的藍色弧線為最後旋轉的 TCP 路徑。旋轉使用 edge 的桌面 XY 投影方向作軸，避免測量 edge 的微小 Z 誤差讓最終工具偏離垂直；原本偵測 edge 與路徑座標不會被修正。整段旋轉也先檢查 IK、插值 tip 偏差、workspace 與 collision geometry；失敗時不開始手臂路徑。

執行：

```bash
/home/neaf6000pro/isaacsim/python.sh \
  /home/n/ur5e_isaac_calibration_ws/isaac_sim/lrbo_eye_to_hand/run_lrbo_cube_edge.py
```

方塊為 12 cm，啟動時隨機 XY 與 Z 軸旋轉。Home、手動關節與夾爪沿用原場景；此入口已移除錄製介面並停用錄製寫入。

## 單次偵測

在 **Cube detection / OpenCV vs LRBO** 視窗按 **DETECT ONCE**，取得當下同一相機 frame 的 RGB 與 Depth，偵測一次並凍結 RGB overlay、Green Mask、端點與深度。再次按下會更新結果；失敗時會清除先前的邊，不保留過期結果。**CLEAR / LIVE PREVIEW** 清除結果並恢復未偵測的 RGB 預覽。原本 **D435 RGB** viewport 一直保持即時。

凍結畫面的邊與深度只屬於擷取時刻；手臂移動後需要重新按 DETECT ONCE。深度單位為公尺，相機 optical Z。RGB/Hough 候選線仍可能受陰影影響，不等同已證明的物理稜邊。

## 使用已儲存參數

此入口不建立 OpenCV/LRBO 校正器，不擷取校正資料、不求解、不啟動 PREDATOR worker，也不建立校正視窗。只讀取先前已儲存的兩份 `T_base_camera` 外參，保存在 detector 的 `opencv_calibration`、`lrbo_calibration` 中，原始參數檔不會被改寫。比較報告另存到 `comparisons/`。

## 比較

預設選取檔名時間戳最新的 OpenCV YAML 與 LRBO JSON，完整路徑顯示在可編輯欄位。

- **LOAD / COMPARE SAVED**：載入並比較兩份之前儲存的外參。
- **SAVE COMPARISON**：將差值、來源檔、兩份矩陣與可用的 GT 誤差另存至 `comparisons/`。

顯示 camera origin 的平移差、LRBO − OpenCV 的 xyz 差與相對旋轉角；兩份 `T_base_camera` 矩陣輸出至 terminal。平移 m → mm，旋轉以度顯示。兩者的方向皆為 camera optical frame → robot base。

OpenCV 匯出將 parent 寫成 `base`，LRBO 使用機器人 prim 路徑。只有儲存的 capture GT 一致時才承認此別名；GT 是比較條件與誤差評估的證據，不用來修正/擬合任何估計。如果相機或底座姿態不同就拒絕比較。若 parent 名稱相同但缺少 GT，只能做數值比較，畫面會標示無法驗證歷史姿態。兩種估計的差值本身不代表精度；可用時另列各自相對 simulation GT 的誤差。

可在啟動前以 `OPENCV_CALIBRATION_FILE`、`LRBO_CALIBRATION_FILE` 指定檔案，也可修改畫面的路徑欄位。支援本專案 YAML/JSON，以及帶有配套 metadata 的 `*_T_base_camera.npy`；不接受無座標系 metadata 的裸矩陣。

離線測試：

```bash
python3 -m unittest discover \
  -s /home/n/ur5e_isaac_calibration_ws/isaac_sim/lrbo_eye_to_hand \
  -p test_cube_edge_comparison.py -v
```

## Edge → base 3D tip / TCP 路徑（預設不移動）

`run_lrbo_cube_edge.py` 保留原場景、單次邊緣偵測與校正檔比較，幾何與選用執行器放在 `edge_tip_path.py`。

1. 按 **DETECT ONCE**，同時凍結該 frame 的 pixel endpoints、測量深度、相機 intrinsics 與座標系證據。
2. 自動產生預覽；也可按 **BUILD / PREVIEW 3D PATH** 重建。獨立讀取 OpenCV 欄位指定的校正檔，不要求 LRBO 比較成功。
3. 世界場景中 **紅色線／球**是 camera → base 後的 detected edge 與 P1/P2，**青色**是有 clearance 的 desired tip path，**橘色**是補償 offset 後的 TCP path。所有 base 點經 `T_world_base` 才畫入 USD，避免混用 robot-root 與世界座標。
4. terminal 列出 pixel、camera / base 3D endpoints、edge/tool direction、實際 local 工具軸、完整 TCP→tip vector、R 與 wxyz quaternion。

主要設定在入口檔的 `EXECUTE_ROBOT` 附近：

```python
EXECUTE_ROBOT = True        # 仍需按 EXECUTE PATH 才執行
TOOL_TILT_DEG = 45.0
EDGE_CLEARANCE = 0.020       # m，沿桌面 +normal 向上留空
PRE_START_CLEARANCE = 0.080
WAYPOINT_SPACING = 0.002
TCP_TO_TIP_OFFSET = None    # 自動從既有 URDF 的閉合指墊求出
TCP_TIP_AXIS = None         # 自動從 tool0→gripper-base 安裝轉換求出
TIP_SPEED_M_S = 0.030
JOINT_SPEED_RAD_S = 0.30
JOINT_ACCEL_RAD_S2 = 1.80
MIN_WAYPOINT_DURATION_S = 0.100
```

**Tip 定義**：Robotiq 2F-140 關閉（main joint=0.7 rad）時，兩個 finger pad 遠端面中心的中點。不是 tool0 原點，也不是單一左／右指。現有 URDF 算出的 `d_tcp_tip ≈ [0,0,0.23316] m`；實際工具軸經安裝轉換確認為 tool0 的 +Z。若更換實際工具，應重新定義 offset/axis；不能把夾爪開啟時的 tip 當成同一個固定 offset。

姿態使 tool axis 對桌面 XY 成 45° 並指向下方，其 XY 投影與 edge XY 投影垂直；優先取向 cube 內側，使工具本體位於另一側。沿 edge orientation 固定。補償採 `tcp = desired_tip - R_base_tcp @ d_tcp_tip`，不只是 Z 高度。

使用 OpenCV `T_base_camera` 將 ROS optical frame 的點轉到當時校正使用的機器人 root frame。這不是 URDF 另一個可能含 180° 固定旋轉的 `base` link；需要靠儲存 GT 與目前相機/root 姿態一致來驗證舊檔能用在此場景。GT 只作驗證，不用於修正 OpenCV 估計。

Depth 為 camera optical Z。`edge_surface_depth.py` 分別從邊緣兩側的遮罩內部取樣，用 RANSAC 擬合局部平面的 inverse depth，再計算原始 `(u,v)` ray 與該表面的交點深度。避免把往內偏移的 patch median 直接套到原始端點而產生偏差；原 median 另存為診斷值。這是局部平面假設，不設定固定 Z，不使用理想 cube／GT／校正外參修正深度，仍可能受錯誤選邊影響。

參數在 `edge_surface_depth.py`：`SURFACE_STRIP_WIDTH_PX=10`、`SURFACE_DEPTH_INLIER_M=0.003`、`SURFACE_MIN_INLIER_RATIO=0.75`。必須有足夠 inlier、涵蓋整段邊與支援兩端，否則拒絕，沒有強制回退。overlay 顯示支援比例與表面 RMS；它們是擬合品質，不代表手眼校正準確度。仍要求至少 80% edge 深度有效，並保留直線 3D edge 一致性、NaN、長度及 workspace 檢查。

## 選用執行（請先確認預覽）

預設 False 下即使按 **EXECUTE PATH**，也不送手臂／夾爪動作。確認青色 tip 與橘色 TCP 路徑後，將檔案中的 `EXECUTE_ROBOT=True` 並重啟；重新 DETECT / PREVIEW，按 **EXECUTE PATH** 才動作。

流程：保持目前手臂 → 關夾爪 → 實際閉合 tip offset 驗證 → 全路徑 preflight → 先上抬目前 TCP → 平滑過渡到 pre-start 與目標姿態 → edge start → 固定姿態沿 edge waypoints → edge end 保持。**STOP** 以目前關節位置為保持目標。

Lula IK 使用從現有 URDF 抽出的 world→tool0 純運動鏈，不更改原 URDF 或場景。執行前比較 Lula FK 與實際 USD tool0 frame；若 frame/joint convention 不一致則拒絕。全路徑先解 IK，檢查關節限位、分支跳變、Cartesian waypoint jump，並在插值點檢查 tip 路徑偏差與工具／手臂 URDF collision geometry 是否侵入桌面／方塊。這個幾何篩檢不是完整自碰撞／三角形碰撞規劃器。

關節指令送至既有 `ArticulationController.apply_action`，用 simulation time 與五次平滑曲線限制速度／加速度；不 teleport joint、不改場景。運動期間手動 Home／夾爪指令被丟棄，避免覆蓋軌跡；停止後恢復。執行時若夾爪開合改變、base/cube 改變、追蹤偏差過大、simulation reset 或 IK 失敗，會停止。相機姿態在開始執行前亦會重查。請先用預覽核對 edge 是否真的是預期的物理稜邊。

Tip world XY 範圍是目前桌面內縮 5 mm；Z 為桌面上方 2 mm 到 0.80 m。TCP world XY 允許比桌面各擴 0.30 m、Z 允許桌面上方 2 mm 到 1.20 m，因 45° 工具會使 TCP 位於 tip 側上方。調整此範圍不會改變桌面配置。

數學／motion guard 測試：

```bash
python3 -m unittest discover \
  -s /home/n/ur5e_isaac_calibration_ws/isaac_sim/lrbo_eye_to_hand \
  -p test_edge_tip_path.py -v
```

## 相機路徑 vs 理想路徑

按 DETECT ONCE / BUILD PATH 後，除了原本的相機估計路徑，新增獨立 Isaac 幾何參考。依 Cube 的實際 USD local size 與完整世界矩陣（包含 scale）建立 12 條物理稜邊；以當次相機 pose GT 投影到影像，找與 RGB 偵測線最接近的可見稜邊。配對須通過 pixel distance / angle 門檻，否則顯示無可靠配對，避免將陰影線硬當成物理稜邊。

Hough 常只找到完整稜邊的一部分，所以參考依 pixel endpoints 作 perspective-correct 對應，比較同一段可見稜邊，另列完整稜邊涵蓋率。不用估計深度來修正理想參考的位置。

原線：紅＝相機估計 edge，青＝估計 tip，橘＝估計 TCP。

新增：灰＝完整理想稜邊，紫紅＝配對的理想子線段，藍＝理想 tip，綠＝理想 TCP。

看 **青 vs 藍** 可比較預期 tip 路徑定位差；看 **橘 vs 綠** 可比較含姿態／offset 的 TCP 路徑差。不要將橘 TCP 與方塊 edge 的 23.3 cm 工具 offset 當成相機定位誤差。

顯示：

- P1/P2 3D 端點誤差與 terminal xyz 分量（估計 − 理想，mm）。
- 同一子線段的 edge / tip RMS 與最大誤差（mm）。相同 clearance 在兩邊同時加，因此這兩項定位誤差相同。
- edge direction 與 tool orientation 差（deg）。
- TCP RMS：理想端點+理想方向建立的完整 TCP 參考；另列沿用估計 orientation 的 TCP 誤差，幫助區分位置誤差與 R@offset 的角度放大。
- 估計線段長度、理想子線段長度、完整實體稜邊長度與涵蓋率。

每次 preview 自動將端點、兩組路徑、誤差與來源參數另存 `comparisons/edge_paths/edge_path_*.json`。理想幾何與 camera GT 只用於比較，不修正相機估計、不覆蓋校正檔、不改变實際控制路徑。這項比較不量測機器人實際走出的 tracking trajectory；沒有執行機器人也能評估。

純數學測試：`python3 -m unittest discover -s /home/n/ur5e_isaac_calibration_ws/isaac_sim/lrbo_eye_to_hand -p test_edge_path_evaluation.py -v`

碰撞頂點直接讀取原 URDF 的 collision STL／box、origin 與 scale，再使用同一套 FK 轉換；不從 USD 視覺 mesh 回推 link 幾何，避免匯入／姿態快取造成錯位。接近路徑先嘗試 80 mm，再嘗試 160／240 mm；所有候選仍檢查 workspace、IK、幾何。沿邊 clearance 保持 20 mm。

偵測前可按 USE OpenCV／USE LRBO 選擇既有校正參數（預設 OpenCV），目前來源显示於按鈕下方與 Preview。所選檔案用於 pixel/depth→base 路徑與執行，沒有重新校正或失敗時偷偷回退。切換會清除舊 3D 路徑，保留同一份 RGB/depth snapshot 供公平比較，必須重新 BUILD；執行中禁止切換。報告附 calibration_source／calibration_file。

此版本不建立 RGB／Depth 錄製 writer，不處理錄製命令，也不產生錄製 heatmap。比較 JSON／執行診斷紀錄仍正常儲存。原始場景程式的錄製功能未修改。
