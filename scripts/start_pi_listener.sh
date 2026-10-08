#!/usr/bin/env bash
#
# Start the Pi listener after releasing GPIOs and port 8765 from stale
# pi.app TCP listeners. Only processes whose command line contains both
# "pi.app" and "--tcp-listen" are targeted.

set -euo pipefail

ROOT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="$ROOT_DIR/.venv/bin/python"
TERM_WAIT_SECONDS=5

if [[ ! -x "$PYTHON" ]]; then
    echo "Missing Pi virtual environment: $PYTHON" >&2
    exit 1
fi

listener_pids() {
    local proc pid command comm
    for proc in /proc/[0-9]*; do
        pid="${proc##*/}"
        [[ "$pid" == "$$" ]] && continue
        [[ -r "$proc/cmdline" ]] || continue
        comm="$(< "$proc/comm")"
        [[ "$comm" == python* ]] || continue
        command="$(tr '\0' ' ' < "$proc/cmdline")"
        if [[ "$command" == *"pi.app"* && "$command" == *"--tcp-listen"* ]]; then
            printf '%s\n' "$pid"
        fi
    done
}

stale_pids="$(listener_pids || true)"
if [[ -n "$stale_pids" ]]; then
    echo "Stopping stale Pi listeners: $stale_pids"
    while read -r pid; do
        [[ -n "$pid" ]] && kill -TERM "$pid" 2>/dev/null || true
    done <<< "$stale_pids"

    for _ in $(seq 1 "$TERM_WAIT_SECONDS"); do
        remaining="$(listener_pids || true)"
        [[ -z "$remaining" ]] && break
        sleep 1
    done

    remaining="$(listener_pids || true)"
    if [[ -n "$remaining" ]]; then
        echo "Force-stopping stale Pi listeners: $remaining" >&2
        while read -r pid; do
            [[ -n "$pid" ]] && kill -KILL "$pid" 2>/dev/null || true
        done <<< "$remaining"
    fi
fi

if [[ "$#" -eq 0 ]]; then
    set -- \
        --tcp-listen 0.0.0.0:8765 \
        --realtime \
        --lcd \
        --status-interval 0.5
fi

exec "$PYTHON" -m pi.app "$@"
