#!/usr/bin/env bash
# DEPRECATED — kept for backward compatibility.
# Please switch to:  ./run.sh sensor
#
# The sensor and grippers now build from a single image (docker/Dockerfile),
# driven by docker/run.sh.
#
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "[deprecated] build_launch_docker_ros2.sh → forwarding to ./run.sh sensor" >&2
exec "${SCRIPT_DIR}/run.sh" sensor "$@"
