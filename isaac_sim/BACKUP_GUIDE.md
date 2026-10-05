# 獨立 repository 更新

此文件下方為原工作區整理紀錄。現在獨立 repository 位於 `/home/n/ur5e_isaac_calibration_ws`，沒有設定 origin；原遠端只屬原工作區。最新備份與推送步驟以根目錄 README 為準。

# Isaac Sim 專案備份整理

本工作區已有 Git，分支 `main`，origin 為 `https://github.com/NN92002/ur5e_camera_ws`。此次整理不移動程式、不改原场景、不建立 commit、不推送。

## 三個版本入口

| 用途 | 入口 | 說明與資料 |
|---|---|---|
| OpenCV eye-to-hand 校正 | `ur5e_camera_scene_charuco_calibration.py` | ChArUco 校正；`calibration/` 保留 YAML、矩陣 NPY、GT 與樣本 NPZ |
| LRBO eye-to-hand 校正 | `lrbo_eye_to_hand/run_lrbo_isaac.py` | `lrbo_eye_to_hand/README.md`；`captures/`、`results/`、worker、kernels 保留 |
| 校正後部署／邊緣路徑執行 | `lrbo_eye_to_hand/run_lrbo_cube_edge.py` | `lrbo_eye_to_hand/CUBE_EDGE.md`；可選 OpenCV／LRBO 外參、20 mm clearance、固定 tip 最後轉垂直；不錄製 |

三個入口均使用既有 UR5e、桌面、環境相機配置。部署版本依賴 LRBO 入口的 scene loader，因此兩者須一起備份。共用 URDF／Xacro、底層場景程式、`charuco_board.png` 與 `src/` 模型 mesh 也需保留。

## 備份範圍

保留原始碼、README、測試、OpenCV calibration 全資料、LRBO captures/results、比較 Markdown/JSON 與執行 JSONL。樣本和診斷資料是此次備份的一部分。

`.gitignore` 排除 ROS build/install/log、Python cache、Isaac generated USD 資產與帶本機絕對路徑的 resolved URDF；這些可由原始模型重建。不刪除任何本機檔案。

## 啟動

在工作區根目錄：

```bash
# 按實際 Isaac Sim 安裝位置調整
ISAAC_PYTHON=/home/neaf6000pro/isaacsim/python.sh
"$ISAAC_PYTHON" isaac_sim/ur5e_camera_scene_charuco_calibration.py
"$ISAAC_PYTHON" isaac_sim/lrbo_eye_to_hand/run_lrbo_isaac.py
"$ISAAC_PYTHON" isaac_sim/lrbo_eye_to_hand/run_lrbo_cube_edge.py
```

每次只啟動一個入口。歷史外參與 launcher 中的本機路徑搬機後需要更新，不能把環境快照當成一鍵安裝器。

## 工作區外的依賴

- Isaac Sim 安裝：`/home/neaf6000pro/isaacsim`（目前 6.1）。不加入此 Git repo。
- LRBO 上游：`/home/n/LRBO`，origin `https://github.com/leihui6/LRBO`。
- PREDATOR：`/home/n/LRBO/third_party/OverlapPredator`。
- 權重：`/home/n/LRBO/weights/registration/model_best_recall.pth`。
- LRBO 模型／設定：`dataset/UR5e_base_m_x5.txt`、`implementation/registration/test/indoor.yaml`。
- Python 環境：`/home/n/miniforge3/envs/lrbo-predator-gpu`；版本清單見 `backup/lrbo_python_packages.json`。

以上工作區外的權重、客製上游內容須另備份；本 repo 沒有包含它們。可用 `LRBO_ROOT`、`LRBO_PYTHON`、`LRBO_CHECKPOINT` 指定部署路徑。上游 commit 與本機修改狀態見 `backup/external_dependencies.txt`。

## 下一步 Git 備份

先確認差異，尤其既有場景檔有較大修改，整理時沒有覆蓋或還原它們：

```bash
git status --short
git diff --stat
git add README.md .gitignore isaac_sim backup
git diff --cached --stat
git commit -m "Back up Isaac OpenCV and LRBO calibration and cube edge execution"
git push origin main
```

這些是後續手動步驟，尚未執行。執行前可查看 `backup/pending_isaac_files.json`，它記錄此次準備備份的 Isaac 檔案大小與 SHA-256。
