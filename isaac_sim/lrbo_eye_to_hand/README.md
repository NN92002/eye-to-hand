# LRBO Eye-to-Hand in Isaac Sim

## 啟動與操作

```bash
/home/neaf6000pro/isaacsim/python.sh \
  /home/n/eye-to-hand/isaac_sim/lrbo_eye_to_hand/run_lrbo_isaac.py
```

更新後請關閉舊程式並重啟。固定相機、機器人底座和手臂姿態，讓底座與肩部
清楚可見。先按 CAPTURE BASE ROI 擷取一筆，再按 RUN LRBO，通過品質檢查後
按 SAVE RESULT。舊擷取缺少當下 CAD 模型，前端要求重新擷取。

多筆資料會各自配準，任意兩筆差異超過 50 mm 或 5° 時不產生平均外參。
相機 GT 僅供誤差評估與儲存檢查，不參與模型建立或配準初始化。

## Runtime

使用獨立 PREDATOR Python worker，並隔離 Isaac Sim 的 Python 與 native library
環境。預設使用 CUDA；CPU 診斷可在啟動指令前加 `LRBO_DEVICE=cpu`。
可用 `LRBO_ROOT`、`LRBO_PYTHON`、`LRBO_CHECKPOINT` 修改既有 runtime 路徑。
`LRBO_RUNTIME_ROOT`、`LRBO_PREDATOR_ENTRYPOINT` 已不使用。

新擷取包含 `model_points_base_m`：由當下底座／肩部 visual mesh 按面積取樣，
以 imported robot root 為座標系。兩點雲各乘 5 配準，輸出旋轉不變、平移除以 5，
不取逆矩陣。輸出的 `T_base_camera` 直接映射相機 optical 座標至該 robot root；
因此不需猜測外部 LRBO 模型與 URDF base_link 的 180° 修正。

擷取存於 `captures/`，通過檢查的校正结果存於 `results/`。幾何近似對稱或遮擋
仍可能導致配準錯誤；本次修改未放寬儲存門檻，也不保證每個視角皆可校正。

## 離線測試

```bash
/home/n/miniforge3/envs/lrbo-predator-gpu/bin/python -m unittest discover \
  -s /home/n/eye-to-hand/isaac_sim/lrbo_eye_to_hand -p test_lrbo.py -v
```

## Version 3: pose-aware robot CAD target

New captures sample the selected base/shoulder visual mesh surfaces in the
imported robot root frame. Each scan stores `model_points_base_m`, so shoulder
motion updates the matching target and its frame is explicitly the same robot
root used for evaluation. The fixed LRBO target is retained only for legacy
worker replay; the frontend asks users to recapture old scans. This uses known
robot geometry and forward kinematics, never camera GT. This is a simulator CAD
variant of LRBO registration, not reproduction of the original fixed-model demo.

The five scans are registered independently. If any pair differs by more than
50 mm or 5 degrees, solving fails before averaging. Capture with one stable arm
pose and no occlusion first. The quality checks still apply; no save threshold
was relaxed. After upgrading, restart Isaac Sim and acquire fresh scans.

## 固定相機的連續掃描初始化

每筆仍執行 PREDATOR。當前一筆 fitness >= 0.9、RMSE <= 5 mm，後一筆也會
用前一筆點雲外參初始化一次 ICP，再以點雲覆蓋率和殘差選擇結果。這不使用
相機 GT；它利用 eye-to-hand 相機／base 固定的條件。請保持相機和底座固定，
CLEAR LIST 後先擷取一筆底座清楚可見的影像。多筆一致性與儲存檢查仍保留。
