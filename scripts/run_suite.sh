#!/usr/bin/env bash
# Run model-router tests in a fresh, sanitized Hermes home.
set -u

repo_dir=$(cd -- "$(dirname -- "$0")/.." && pwd -P)
mode=${1:---full}

if [[ -n ${ROUTER_TEST_PYTHON:-} ]]; then
    test_python=$ROUTER_TEST_PYTHON
else
    hermes_agent=${ROUTER_HERMES_AGENT:-"$HOME/.hermes/hermes-agent"}
    test_python="$hermes_agent/venv/bin/python"
fi

if [[ ! -x $test_python ]]; then
    printf 'ROUTER_TEST_PYTHON must name an executable Python interpreter: %s\n' "$test_python" >&2
    exit 2
fi

if [[ -n ${ROUTER_HERMES_AGENT:-} ]]; then
    hermes_agent=$ROUTER_HERMES_AGENT
else
    hermes_agent=$(cd -- "$(dirname -- "$test_python")/../.." 2>/dev/null && pwd -P || true)
fi

run_dir=$(mktemp -d)
mkdir -p "$run_dir/home/.hermes"
cat > "$run_dir/home/.hermes/config.yaml" <<EOF
# Sanitized offline test topology. No live credentials or provider settings.
delegation:
  max_spawn_depth: ${ROUTER_TEST_MAX_SPAWN_DEPTH:-1}
  max_concurrent_children: ${ROUTER_TEST_MAX_CONCURRENT_CHILDREN:-3}
  max_iterations: ${ROUTER_TEST_MAX_ITERATIONS:-250}
  orchestrator_enabled: true
EOF
ln -s "$repo_dir" "$run_dir/model_router"

export HOME="$run_dir/home"
export HERMES_HOME="$run_dir/home/.hermes"
export PYTHONPATH="$run_dir:$run_dir/model_router${PYTHONPATH:+:$PYTHONPATH}${hermes_agent:+:$hermes_agent}"
export PYTHONDONTWRITEBYTECODE=1
if [[ -d $hermes_agent/node_modules ]]; then
    export NODE_PATH="$hermes_agent/node_modules${NODE_PATH:+:$NODE_PATH}"
fi

cd "$run_dir" || exit 2
case "$mode" in
    --focused)
        exec "$test_python" -B -m unittest \
            model_router.test_claude_delegation \
            model_router.test_claude_opus_bridge \
            model_router.test_bridge_policy \
            model_router.test_worker_admission \
            model_router.test_external_orchestrator \
            model_router.test_conductor_route \
            model_router.test_quota_redispatch \
            model_router.test_balance \
            model_router.test_any_parent_orchestrates \
            model_router.test_workflow_switch
        ;;
    --full)
        exec "$test_python" -B -m unittest discover -s model_router -t . -p 'test_*.py'
        ;;
    *)
        printf 'usage: %s [--focused|--full]\n' "$0" >&2
        exit 2
        ;;
esac
