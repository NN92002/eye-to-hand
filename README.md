# UR5e Isaac 校正與邊緣路徑

獨立備份 repository，來源 `/home/n/ur5e_camera_ws`；原專案未修改。包含 OpenCV eye-to-hand、LRBO eye-to-hand、校正後 cube edge 路徑執行。

| 用途 | 入口 |
|---|---|
| OpenCV 校正 | `isaac_sim/ur5e_camera_scene_charuco_calibration.py` |
| LRBO 校正 | `isaac_sim/lrbo_eye_to_hand/run_lrbo_isaac.py` |
| 邊緣偵測與路徑執行 | `isaac_sim/lrbo_eye_to_hand/run_lrbo_cube_edge.py` |

在本 repository 根目錄執行（選一個入口）：

```bash
/home/neaf6000pro/isaacsim/python.sh isaac_sim/lrbo_eye_to_hand/run_lrbo_cube_edge.py
```

共用場景使用本 repository 的模型、校正資料與輸出資料夾，不再從原工作區載入。校正檔案內記錄的歷史路徑與證據維持原樣。

操作詳見 [cube edge 說明](isaac_sim/lrbo_eye_to_hand/CUBE_EDGE.md) 與 [LRBO 說明](isaac_sim/lrbo_eye_to_hand/README.md)。`src/` 保留共用 UR／Robotiq 模型與其授權，沒有複製巢狀 Git metadata。

`calibration/`、LRBO `captures/`／`results/`、比較報告與執行診斷均保留；generated USD、resolved URDF、Python cache 與錄影排除。

Isaac Sim 安裝、`/home/n/LRBO` 的客製內容與 PREDATOR 權重、Conda 環境屬外部依賴，尚未包含在此 repo；清單見 `backup/external_dependencies.txt` 與 `backup/lrbo_python_packages.json`。原始檔案校驗清單見 `backup/source_copy_manifest.json`。

本機 Git 已初始化；尚未設定遠端。建立新的空 GitHub repository 後可執行：

```bash
git remote add origin <新 repository URL>
git push -u origin main
```
