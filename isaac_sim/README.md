# UR5e + Robotiq 2F-140 + D435 模擬環境

本目錄以 [`ur5e_robotiq_d435_live_v16.py`](ur5e_robotiq_d435_live_v16.py) 為執行入口，建立 Isaac Sim 場景，提供手臂與夾爪即時控制，以及模擬相機 RGB／深度影像。

> 本文件依照此 Python 檔的實作整理。程式檔頭標示 Isaac Sim 6.1、v18，並保留舊的 `ur5e_camera_scene.py` 執行範例；以下使用實際的 `ur5e_robotiq_d435_live_v16.py` 檔名。

## 功能

- 固定基座的 UR5e 與 Robotiq 2F-140，從已展開的 URDF 匯入。
- 0.70 × 0.70 m 工作台、中心標記與 4 × 4 m 地板。
- 主視窗顯示整體場景，第二個 Viewport 顯示 D435 RGB 視角。
- Isaac Sim 原生 `omni.ui` 控制面板，可即時調整六軸角度及夾爪閉合百分比。
- 可選的 OpenCV 深度視窗，顯示彩色深度圖與中心像素深度（公尺）。

此腳本使用模擬相機，沒有建立 ROS 2 topic、啟動實體 RealSense 或連線實體機器人。U-Arm 目前僅預留命令佇列介面，尚未實作通訊。

## 執行環境與模型

使用提供腳本所用 API 的 Isaac Sim 環境（程式標示為 6.1），透過安裝目錄內的 `python.sh` 啟動。程式設定 `headless=False`，需要可顯示 Isaac Sim GUI 的圖形環境。

模型檔案需保留以下結構：

```text
ur5e_camera_ws/
├── isaac_sim/
│   ├── ur5e_robotiq_d435_live_v16.py
│   ├── ur5e_robotiq_2f140.urdf
│   └── ur5e_robotiq_2f140.xacro
└── src/
    ├── universal_robot/
    └── robotiq_ros/grippers/robotiq_description/
```

啟動時直接使用現有的 `.urdf`，不會自動執行 Xacro。若修改 `.xacro`，需先重新產生對應 URDF。已展開的 URDF 與 mesh 齊備時，此腳本的執行流程不需要啟動 ROS 2。

`cv2` 為選用依賴；Isaac Sim 的 Python 環境若無法匯入 OpenCV，會略過深度視窗。若需要該視窗，OpenCV 必須支援 GUI 顯示。

## 啟動

此機器的 Isaac Sim 安裝目錄為 `/home/neaf6000pro/isaacsim`：

```bash
cd /home/neaf6000pro/isaacsim
./python.sh /home/n/ur5e_isaac_calibration_ws/isaac_sim/ur5e_robotiq_d435_live_v16.py
```

首次匯入與相機初始化需要等待模型及渲染資源載入。完成後可看到整體場景、D435 相機視窗，以及 `UR5e + Robotiq 2F-140 LIVE Control` 控制面板。關閉 Isaac Sim 主程式即可結束，程式會清理 OpenCV 視窗。

搬移專案時，除了 `WORKSPACE_ROOT`，也要修改 `import_ur5e()` 內寫死的 URDF、mesh、解析後 URDF 與 USD 輸出絕對路徑。腳本目前未提供命令列參數。

## 控制面板

| 控制項 | 操作方式 |
| --- | --- |
| 六軸數值欄／滑桿 | 修改即送出目標角度，單位為度，GUI 範圍為 −180°～180° |
| `current` | 顯示模擬中的目前角度或夾爪閉合百分比 |
| `Home` | 送出 `[0, -90, 90, -90, -90, 0]` 度的六軸目標 |
| `Read Current` | 將角度輸入欄同步成目前姿態，不送出移動命令 |
| 夾爪滑桿 | 0% 為開啟，100% 為閉合，主關節對應 0～0.70 rad |
| `OPEN`／`CLOSE` | 將夾爪目標設為 0%／100% |

六軸順序為 `shoulder_pan_joint`、`shoulder_lift_joint`、`elbow_joint`、`wrist_1_joint`、`wrist_2_joint`、`wrist_3_joint`。GUI 初始欄位使用 Home 數值，但啟動時不會自動送出 Home 命令；需要時按下 `Home`。

手臂命令由主迴圈轉換成弧度，透過 `ArticulationController.apply_action()` 交給 PhysX 位置驅動。GUI 範圍不代表模型的實際關節限制。每次處理佇列時只採用最新命令，狀態標籤每三個模擬迴圈更新一次。

夾爪匯入前會從衍生 URDF 移除自閉合的 `<mimic .../>` 標籤，再依 `ROBOTIQ_JOINT_MULTIPLIERS` 的正負倍率同步控制六個連桿關節；來源 URDF 保留原樣。

## 場景參數

參數目前在 Python 檔中設定，修改後需重新啟動。座標採世界／UR5e 基座座標，Z 軸朝上，長度單位為公尺。

| 參數 | 目前值 | 用途 |
| --- | --- | --- |
| `TABLE_WIDTH` / `TABLE_DEPTH` | `0.70` / `0.70` | 工作台寬、深 |
| `TABLE_THICKNESS` | `0.03` | 工作台厚度 |
| `TABLE_HEIGHT` | `0.0` | 目前建模方式下的工作台底面 Z |
| `WORKSPACE_CENTER` | `[0.45, 0.0, 0.0]` | 工作區參考中心 |
| `GROUND_SIZE_X` / `GROUND_SIZE_Y` | `4.0` / `4.0` | 地板尺寸 |
| `GROUND_TOP_Z` | `0.0` | 地板頂面 Z |
| `CAMERA_POSITION` | `[0.80, 0.00, 1.5]` | 相機位置 |
| `CAMERA_TARGET` | `[0.45, 0.00, 0.00]` | 相機朝向目標 |
| `CAMERA_RESOLUTION` | `(640, 480)` | 影像寬、高 |
| `CAMERA_FREQUENCY` | `30` | 相機設定頻率（Hz），實際速度取決於模擬與渲染 |
| `DEPTH_MIN_M` / `DEPTH_MAX_M` | `0.20` / `3.00` | 深度視窗色彩映射範圍 |

`create_workspace()` 將工作台幾何中心設為 `TABLE_HEIGHT + TABLE_THICKNESS / 2`，因此目前桌面頂部是 Z = 0.03 m。`CAMERA_TARGET` 是獨立設定，修改工作區位置時也需同步調整瞄準位置。相機實際位於 Y = 0，應以參數值為準，而非程式中舊的「−Y 側」註解。

## RGB 與深度

相機 Prim 為 `/D435/camera_color_optical_frame`。相機外殼與內參都是近似設定；`create_d435()` 使用焦距 `1.88`、水平 aperture `3.67`，垂直 aperture 依影像比例計算。

深度使用 `distance_to_image_plane` annotator，表示到成像平面的深度。OpenCV 視窗名稱為 `D435 Depth (meters)`，以 TURBO 色盤顯示；非有限值或非正值像素標黑，中心十字標示取樣位置。0.20～3.00 m 只限制顯示色彩，並非裁切原始深度資料或宣告實體 D435 的量測範圍。

主迴圈每 300 次記錄 RGB／深度陣列形狀與有效深度範圍。程式目前沒有儲存影像、錄影或匯出點雲。

## 匯入產物

實際啟動流程由 `main()` 呼叫 `import_ur5e()`，會產生：

- `isaac_sim/ur5e_robotiq_2f140_isaac_resolved.urdf`：將 UR 與 Robotiq 的 package URI 換成絕對路徑，並移除 mimic 標籤。
- `isaac_sim/generated_ur5e_robotiq_2f140_nomimic_v17/`：URDF importer 產生的 USD 資產；實際入口檔路徑見啟動日誌。

匯入後直接開啟產生的 USD stage，再加入地板、工作台與相機。檔案前段的 `/tmp` 路徑與 `prepare_urdf()` 不屬於目前主程式的匯入流程。

## 問題排查

| 現象 | 檢查方式 |
| --- | --- |
| 找不到 `isaacsim`、`omni` 或相關 API | 確認使用 Isaac Sim 的 `python.sh`，以及環境是否提供腳本使用的 API |
| `Expanded URDF not found` 或 description 目錄不存在 | 檢查 `import_ur5e()` 中的絕對路徑與模型資產 |
| 控制面板數值與機器人姿態不同 | 按 `Read Current` 讀回實際姿態；按 `Home` 才會送出 Home 目標 |
| 缺少 UR5e／Robotiq DOF | 檢查日誌中的匯入 DOF 名稱，以及來源 URDF 與夾爪關節名稱 |
| 沒有深度視窗 | 確認 Isaac Sim Python 可匯入支援 GUI 的 `cv2`，並等待深度資料完成初始化 |
| 深度影像黑色或沒有資料 | 查看深度警告，確認相機朝向、可見幾何與渲染初始化狀態 |

ROS 2／RViz2 工作區的使用方式請見[專案 README](../README.md)。
