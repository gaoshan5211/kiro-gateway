#!/usr/bin/env bash
set -euo pipefail

LABEL="${KIRO_GATEWAY_LAUNCH_AGENT:-com.kiro-gateway}"
DOMAIN="gui/$(id -u)"
SERVICE="${DOMAIN}/${LABEL}"
PLIST="${KIRO_GATEWAY_PLIST:-${HOME}/Library/LaunchAgents/${LABEL}.plist}"
HEALTH_URL="${KIRO_GATEWAY_HEALTH_URL:-http://127.0.0.1:6009/health}"
HEALTH_TIMEOUT_SECONDS="${KIRO_GATEWAY_HEALTH_TIMEOUT_SECONDS:-20}"

usage() {
  cat <<EOF
Usage:
  ./restart.sh          Restart the local Kiro Gateway LaunchAgent and check health
  ./restart.sh status   Show LaunchAgent status and health
  ./restart.sh help     Show this help

Environment overrides:
  KIRO_GATEWAY_LAUNCH_AGENT=${LABEL}
  KIRO_GATEWAY_PLIST=${PLIST}
  KIRO_GATEWAY_HEALTH_URL=${HEALTH_URL}
EOF
}

require_plist() {
  if [[ ! -f "${PLIST}" ]]; then
    echo "LaunchAgent plist not found: ${PLIST}" >&2
    echo "Check whether the local Kiro Gateway service has been installed." >&2
    exit 1
  fi
}

ensure_loaded() {
  if launchctl print "${SERVICE}" >/dev/null 2>&1; then
    return
  fi

  echo "LaunchAgent is not loaded. Loading: ${PLIST}"
  launchctl bootstrap "${DOMAIN}" "${PLIST}"
}

print_status() {
  launchctl print "${SERVICE}" 2>/dev/null | awk '
    /^[[:space:]]*state = / { print }
    /^[[:space:]]*pid = / { print }
    /^[[:space:]]*runs = / { print }
    /^[[:space:]]*last exit code = / { print }
    /^[[:space:]]*last terminating signal = / { print }
  '
}

check_health() {
  local deadline
  local response

  deadline=$((SECONDS + HEALTH_TIMEOUT_SECONDS))
  while (( SECONDS < deadline )); do
    response="$(curl -fsS --max-time 3 "${HEALTH_URL}" 2>/dev/null || true)"
    if [[ -n "${response}" ]]; then
      echo "Health OK: ${response}"
      return 0
    fi
    sleep 1
  done

  echo "Health check failed after ${HEALTH_TIMEOUT_SECONDS}s: ${HEALTH_URL}" >&2
  echo "Check logs/stdout.log and logs/stderr.log for details." >&2
  return 1
}

restart_service() {
  require_plist
  ensure_loaded

  echo "Restarting ${SERVICE}"
  launchctl kickstart -k "${SERVICE}"
  check_health
  print_status
}

case "${1:-restart}" in
  restart)
    restart_service
    ;;
  status)
    require_plist
    ensure_loaded
    print_status
    check_health
    ;;
  help|--help|-h)
    usage
    ;;
  *)
    usage >&2
    exit 2
    ;;
esac
