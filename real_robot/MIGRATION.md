# 搬到沒有 Isaac 的電腦

只需複製 `real_robot/` 即可取得實機採樣、CAD 取樣與兩種求解程式。
此目錄已內含 `predator_worker.py`、`lrbo_core.py`，執行時不讀取 `isaac_sim/`。
以下指令在複製後的 `real_robot/` 目錄執行，不依賴 `/home/n`。

## 採樣環境

以 Ubuntu 22.04 + ROS 2 Humble 作為目前基準。
依 README 連結的官方文件安裝 ROS、UR driver 與 RealSense ROS wrapper。
需要 Python 套件／ROS 模組：NumPy、OpenCV（含 aruco 和 GUI）、rclpy、sensor_msgs、
cv_bridge、message_filters、tf2_ros。CAD 取樣另需 trimesh、DAE importer。
使用 ROS 的系統 Python，避免用 PREDATOR 的 Conda Python 執行採樣。

```bash
source /opt/ros/humble/setup.bash
# source /實機UR工作區/install/setup.bash
/usr/bin/python3 -c 'import cv2, numpy, rclpy, sensor_msgs, cv_bridge, message_filters, tf2_ros; assert hasattr(cv2, "aruco"); print("capture dependencies OK")'
/usr/bin/python3 -m unittest discover -s . -p 'test_*.py' -v
/usr/bin/python3 capture.py --method charuco --output ./data
```

另開終端啟動 UR／D435 driver，並確認 topics、TF frame 名稱。
需攜帶該台 UR5e 的校正 YAML、使用的 URDF／mesh；原電腦的歷史模擬外參不適用新安裝。

## LRBO 額外依賴

`real_robot/` 不包含 PREDATOR 原始碼、模型權重或完整 Python 環境。
還需攜帶原電腦 `/home/n/LRBO` 下的：

- `third_party/OverlapPredator/`（包含本機修改、kernel 與 cpp_wrappers 原始碼）
- `implementation/registration/test/indoor.yaml`
- `weights/registration/model_best_recall.pth`

保留上述相對目錄結構，例如搬到 `~/LRBO`。
在目標機重新建立獨立 PREDATOR 環境，安裝 PyTorch、Open3D、easydict 等依賴，
並依 PREDATOR 原始碼說明重新編譯 C++ extensions；不能假設來源機的二進位可直接搬用。
`lrbo_environment_reference.json` 是来源機套件版本清單，供排查用，並非可直接安裝的 lockfile。
PyTorch/CUDA 版本須和目標機 GPU、driver 相容；無 GPU 可用 `--device cpu`，速度較慢。

```bash
export LRBO_ROOT="$HOME/LRBO"
export LRBO_PYTHON="/目標機的PREDATOR環境/bin/python"
"$LRBO_PYTHON" -c 'import torch, numpy, open3d, easydict; print(torch.__version__, torch.cuda.is_available())'
/usr/bin/python3 capture.py --method lrbo --model ./models/base_shoulder.npy --output ./data
/usr/bin/python3 solve_lrbo.py ./data/你的session
```

CAD 模型用 README 的 `build_model.py` 在現場生成，將檔名改成相對路徑即可。
不要直接沿用模擬 robot root 的點雲當成實機 base frame。

## 驗證範圍

已驗證實機目錄移到獨立位置後可執行離線測試，不依賴 Isaac 目錄。
尚未在目標機驗證 ROS／相機連線、PREDATOR 重建、GPU 相容性與實機校正精度。
