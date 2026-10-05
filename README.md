# UR5e Isaac 校正與邊緣路徑

獨立備份 repository，來源 `/home/n/ur5e_camera_ws`；原專案未修改。包含 OpenCV eye-to-hand、LRBO eye-to-hand、校正後 cube edge 路徑執行。

| 用途 | 入口 |
|---|---|
| OpenCV 校正 | `isaac_sim/ur5e_camera_scene_charuco_calibration.py` |
| LRBO 校正 | `isaac_sim/lrbo_eye_to_hand/run_lrbo_isaac.py` |
| 邊緣偵測與路徑執行 | `isaac_sim/lrbo_eye_to_hand/run_lrbo_cube_edge.py` |

## 啟動指令

以下三個入口已在目前環境測試正常。先進入專案，每次只啟動一個：

```bash
cd /home/n/eye-to-hand
```

OpenCV eye-to-hand 校正：

```bash
./run_isaac.sh opencv
```

LRBO eye-to-hand 校正：

```bash
./run_isaac.sh lrbo
```

邊緣偵測與路徑執行（可選已儲存的 OpenCV／LRBO 校正參數）：

```bash
./run_isaac.sh edge
```

Launcher 先切換至 Isaac 安裝目錄，清除外部 Python/native 路徑，並跳過未使用的 ROS 初始化。完整啟動紀錄保存在 `logs/`，不加入 Git。預設 Isaac 位於 `/home/neaf6000pro/isaacsim`，其他安裝位置可用 `ISAAC_PATH` 指定。

共用場景使用本 repository 的模型、校正資料與輸出資料夾，不再從原工作區載入。校正檔案內記錄的歷史路徑與證據維持原樣。

操作詳見 [cube edge 說明](isaac_sim/lrbo_eye_to_hand/CUBE_EDGE.md) 與 [LRBO 說明](isaac_sim/lrbo_eye_to_hand/README.md)。`src/` 保留共用 UR／Robotiq 模型與其授權，沒有複製巢狀 Git metadata。

`calibration/`、LRBO `captures/`／`results/`、比較報告與執行診斷均保留；generated USD、resolved URDF、Python cache 與錄影排除。

Isaac Sim 安裝、`/home/n/LRBO` 的客製內容與 PREDATOR 權重、Conda 環境屬外部依賴，尚未包含在此 repo；清單見 `backup/external_dependencies.txt` 與 `backup/lrbo_python_packages.json`。原始檔案校驗清單見 `backup/source_copy_manifest.json`。

## Git 備份

遠端 repository：<https://github.com/NN92002/eye-to-hand>，分支為 `main`。

```bash
git push origin main
```
