# OpenCV / LRBO 分組比較（2026-10-04）

指標為偵測 3D 線段到實際完整稜邊的最近距離 RMS，不包含手臂追蹤誤差。組 1–4 沿用相同 endpoints／cube 姿態配對；新增資料按相同 cube 姿態與實際 edge_id 分組，多次偵測以中位數彙整。

| 組別 | OpenCV RMS mm | LRBO RMS mm | 比較條件 |
|---|---:|---:|---|
| 1 | 2.24 | 1.88 | 相同 pixel endpoints／方塊姿態 |
| 2 | 4.24 | 5.41 | 相同 pixel endpoints／方塊姿態 |
| 3 | 32.68 | 33.64 | 相同 pixel endpoints／方塊姿態 |
| 4 | 2.01 | 1.70 | 相同 pixel endpoints／方塊姿態 |
| 5 | 2.91（3 筆） | 3.43（6 筆） | 同方塊、edge_5_7；端點不同 |
| 6 | — | 1.34（1 筆） | 同方塊、edge_1_3；端點不同 |

新增組別有不同 endpoints／depth，不能把方法間差距全部歸因於外參；不把不同實際稜邊硬配成同一組。

## 新增組別逐筆資料

| 組別 | 方法 | 報告 | 最近距離 RMS mm |
|---|---|---|---:|
| 5 | OpenCV | edge_path_20261004_211035_834128.json | 2.908 |
| 5 | OpenCV | edge_path_20261004_211048_520578.json | 2.908 |
| 5 | OpenCV | edge_path_20261004_211236_191718.json | 1.026 |
| 5 | LRBO | edge_path_20261004_211250_949956.json | 3.587 |
| 5 | LRBO | edge_path_20261004_211302_826720.json | 3.274 |
| 5 | LRBO | edge_path_20261004_211359_474382.json | 5.070 |
| 5 | LRBO | edge_path_20261004_211400_615862.json | 3.941 |
| 5 | LRBO | edge_path_20261004_211409_289386.json | 1.552 |
| 5 | LRBO | edge_path_20261004_211430_691599.json | 1.552 |
| 6 | LRBO | edge_path_20261004_211249_898075.json | 1.343 |
