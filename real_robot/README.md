# UR5e + D435 實機 eye-to-hand

另一台沒有 Isaac 的電腦可只搬 `real_robot/`，詳見 [搬移與環境準備](MIGRATION.md)。
LRBO worker 與計算核心已包含在此目錄，無需 `isaac_sim/`；PREDATOR 原始碼與權重仍需另外攜帶。

實機入口獨立於 Isaac Sim，採用 ROS 2 Humble、UR 官方 driver 的 TF 與 D435 RGB-D topics。
採樣程式只讀取資料；手臂由示教器移動到姿態後停止，再按 c 擷取。
相機與機器人底座全程固定。ChArUco 板須剛性固定於末端；LRBO 則觀察機器人底座／肩部表面。

參考控制架構：[SO101_based_UR5e_tp](https://github.com/sam610510/SO101_based_UR5e_tp)，
此次讀取版本 `84a2865c9003ed3c0823bba6cdeebf2a38c57105`。
使用相同 UR ROS2 driver 介面，未複製它的 SO101 遙操作、啟動歸位或場地專用 FK／安全常數。

## 環境與連線

採樣使用 `/usr/bin/python3`，先 source ROS 與你實際的 UR driver workspace。
需要 rclpy、sensor_msgs、tf2_ros、cv_bridge、message_filters、NumPy、OpenCV aruco。
CAD 取樣另需 trimesh，DAE mesh 可能需要 pycollada；PREDATOR 沿用既有獨立環境。
本機已驗證 OpenCV 4.5.4 與離線測試，尚未驗證實機 topics、TF、GUI 與配準品質。

```bash
source /opt/ros/humble/setup.bash
# source /你的UR工作區/install/setup.bash
cd /home/n/eye-to-hand
```

UR driver 依你現有工作區啟動，指定 UR5e IP 與該台機器的 `kinematics_params_file`。
請依[官方 driver 文件](https://github.com/UniversalRobots/Universal_Robots_ROS2_Driver)完成機器人校正參數與連線。
採樣不需要啟動 SO101 遙操作；避免它啟動後的自動歸位。
手臂移動時依現場既有安全程序操作，停止後才採樣。

D435 範例（參數依已安裝 wrapper 版本確認）：

```bash
ros2 launch realsense2_camera rs_launch.py enable_color:=true enable_depth:=true \
  align_depth.enable:=true enable_sync:=true
```

參數與 optical 座標定義見 [RealSense ROS wrapper](https://github.com/realsenseai/realsense-ros)。
先用 `ros2 topic list` 確認 topics；預設為 `/camera/camera/color/image_raw`、
`/camera/camera/color/camera_info`、`/camera/camera/aligned_depth_to_color/image_raw`。
若命名不同，以 `--image`、`--info`、`--depth` 覆寫。
程式取影像 timestamp 對應的 TF，TF buffer 未取得該時刻時拒絕採樣；不改用最新姿態。
RGB-D 時間差最多 30 ms，因此必須在機器人靜止時採樣。
CameraInfo 必須和彩色影像同 frame／解析度，depth 必須對齊彩色 optical frame。
使用實際內參與畸變，未沿用模擬相機的零畸變。

## ChArUco

```bash
/usr/bin/python3 real_robot/capture.py --method charuco --base base --tool tool0
```

預設板為 DICT_5X5_250、7×5 格、方格 40 mm、marker 30 mm，和現有模擬一致。
印製後量測實際格長，透過 `--square`／`--marker` 指定公尺；列印縮放會直接影響外參尺度。
可用 `--squares-x`／`--squares-y` 調整板規格。

按 c 收集至少 10 組，建議 20–30 組不同位置與繞不同軸的旋轉；板面保持清楚可見。
按 s 用 ANDREFF 求解；按 q 離開。角點不足、重投影誤差 >1.5 px、退化旋轉會拒絕。
結果寫入該 session 的 `T_base_camera.npy`／`result.json`，配對姿態存於 `samples.npz`。
新增樣本會撤除先前解，以免將舊解當作全部樣本的新結果。
closure RMS 是內部一致性指標，不代表實機絕對精度；保留額外姿態作獨立驗證。

## LRBO

LRBO 使用既有 `predator_worker.py` 的 PREDATOR + ICP。
模擬 instance segmentation 改成人工多邊形 mask，需排除桌面、背景、夾爪等非 CAD 區域。
CAD 模型必須是 Nx3 `.npy`，單位公尺，以採樣的 `--base` frame 表示，
只包含所選底座／肩部表面，且肩部姿態與擷取時一致。

可從正確的 visual mesh 與實機 TF 產生模型：

```bash
/usr/bin/python3 real_robot/build_model.py \
  --base base \
  --mesh base_link_inertia /實際模型/base.dae \
  --mesh shoulder_link /實際模型/shoulder.dae \
  --output real_robot/models/base_shoulder.npy
```

這些 mesh 路徑需換成你 URDF 使用的檔案。工具假設 mesh 原點／方向／尺度等同 link frame，
若 URDF visual 有 origin 或 scale，先套用那些變換再使用；不要直接拿 mm STL 當公尺。
使用 driver 載入的該台機器校正 TF，確認 `base`、`base_link` 等 frame 的方向差異，
不要沿用 Isaac robot root 的座標修正。產生模型後保持肩部姿態不變。

```bash
/usr/bin/python3 real_robot/capture.py --method lrbo --base base --tool tool0 \
  --model real_robot/models/base_shoulder.npy
```

按 c，滑鼠點選包住機器人表面的多邊形，Enter 完成、Esc 取消。
選取期间保持手臂與相機固定；保存的是按 c 當下的 RGB-D。
可擷取多筆，按 q 離開，再離線執行：

```bash
/usr/bin/python3 real_robot/solve_lrbo.py real_robot/data/你的session
```

預設使用 `/home/n/LRBO`、`lrbo-predator-gpu` Python 和既有權重。
可覆寫 `LRBO_ROOT`、`LRBO_PYTHON`、`LRBO_CHECKPOINT`，或加 `--device cpu`。
結果要求每筆 fitness >=0.9、真實尺度 RMSE <=5 mm，任兩筆差異 <=50 mm／5°，
否則保留診斷但不產生新校正解。worker 的兩點雲放大 5 倍，輸出平移縮回 1/5，旋轉不變。
對稱底座仍可能配準錯誤，即使通過門檻也需檢查覆疊與獨立量測。

## 結果與驗證

兩者 `T_base_camera` 均將 D435 color optical frame 點映射到指定 base frame：
`p_base = T_base_camera @ p_camera`，平移為公尺。結果 JSON 明確記錄 frame。
比較兩種方法時必須使用相同 base／camera frame 與固定相機安裝。
實機資料存於 `real_robot/data/`，模型存於 `real_robot/models/`，不自動加入 Git。

```bash
/usr/bin/python3 -m unittest discover -s real_robot -p 'test_*.py' -v
```

測試涵蓋合成 eye-to-hand 真值的方向與尺度、退化姿態拒絕、LRBO 品質與多筆一致性。
尚待實機驗證：UR driver／D435 連線、CAD mesh frame、深度表面品質、真實配準與外參精度。
目前提供手動採樣與離線求解；自動移動採樣及校正後路徑執行尚未移植。
