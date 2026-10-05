#!/usr/bin/env bash
set -euo pipefail
project_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
isaac_root="${ISAAC_PATH:-/home/neaf6000pro/isaacsim}"
case "${1:-edge}" in
    edge) entry="$project_root/isaac_sim/lrbo_eye_to_hand/run_lrbo_cube_edge.py" ;;
    opencv) entry="$project_root/isaac_sim/ur5e_camera_scene_charuco_calibration.py" ;;
    lrbo) entry="$project_root/isaac_sim/lrbo_eye_to_hand/run_lrbo_isaac.py" ;;
    *) echo "Usage: $0 [edge|opencv|lrbo]" >&2; exit 2 ;;
esac
if [[ ! -x "$isaac_root/python.sh" ]]; then
    echo "Isaac python.sh not found: $isaac_root; set ISAAC_PATH" >&2
    exit 1
fi
mkdir -p "$project_root/logs"
run_log="$project_root/logs/isaac_$(date +%Y%m%d_%H%M%S).log"
echo "Entry: $entry"
echo "Log: $run_log"
# Let Isaac's own setup scripts build its native/Python paths. No ROS is used.
cd "$isaac_root"
env -u PYTHONPATH -u PYTHONHOME -u PYTHONEXE -u LD_LIBRARY_PATH -u LD_PRELOAD \
    -u CONDA_PREFIX -u CONDA_DEFAULT_ENV -u CONDA_SHLVL \
    "$isaac_root/python.sh" --no-ros-env "$entry" 2>&1 | tee "$run_log"
