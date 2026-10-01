import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from claude_opus_bridge import (CANONICAL_OPUS_MODEL, ClaudeBridgeFailure, classify_coding_dispatch,
                                classify_review_dispatch, dispatch)


class ClaudeOpusBridgeTests(unittest.TestCase):
    def setUp(self):
        from model_router import execution_adapters
        owner = patch.object(execution_adapters, "RESERVATIONS", execution_adapters.ReservationBook())
        owner.start()
        self.addCleanup(owner.stop)

    @patch("claude_opus_bridge.subprocess.run")
    def test_invalid_limits_never_launch_a_process(self, run):
        with tempfile.TemporaryDirectory() as directory:
            for turns in (0, -1, 0.5, 1.5, True, "2"):
                with self.subTest(turns=turns), self.assertRaisesRegex(ValueError, "positive integer"):
                    dispatch("[opus-review] Review parser", Path(directory), review=True, max_turns=turns)
            for budget in (0, -1, 0.001, float("nan"), float("inf")):
                with self.subTest(budget=budget), self.assertRaisesRegex(ValueError, "at least 0.01"):
                    dispatch("[opus-review] Review parser", Path(directory), review=True, max_budget_usd=budget)
        run.assert_not_called()

    @patch("claude_opus_bridge.subprocess.run", side_effect=OSError("CLI missing"))
    def test_adapter_records_parent_identity_and_terminal_launch_failure(self, run):
        import sqlite3
        from model_router import _run_opus5_bridge
        from agent_activity import load_agent_activity
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            lifecycle = root / "bridge.jsonl"
            db = root / "state.db"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE async_delegations (delegation_id TEXT, origin_session TEXT, parent_session_id TEXT, state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL, task_json TEXT, result_json TEXT)")
                conn.execute("CREATE TABLE sessions (id TEXT, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
                conn.execute("INSERT INTO sessions VALUES ('parent', NULL, 1, NULL, 'gpt-terra')")
                conn.execute("INSERT INTO sessions VALUES ('child', 'parent', 2, NULL, 'gpt-terra')")
            with self.assertRaisesRegex(OSError, "CLI missing"):
                _run_opus5_bridge(repo=directory, task="[opus-review] Review parser", write=False,
                    review=True, cfg={"coding_agent": {"lifecycle_path": str(lifecycle)}},
                    parent_session_id="parent", session_id="child", parent_turn_id="parent:turn",
                    turn_id="child:turn")
            events = [json.loads(line) for line in lifecycle.read_text().splitlines()]
            activity = load_agent_activity(db, bridge_lifecycle_path=lifecycle)
        self.assertEqual([e["state"] for e in events], ["running", "error"])
        self.assertEqual({e["parent_session_id"] for e in events}, {"parent"})
        self.assertEqual({e["parent_turn_id"] for e in events}, {"parent:turn"})
        bridge = next(child for parent in activity["parents"] for child in parent["children"]
                      if child.get("id") == events[0]["bridge_run_id"])
        self.assertEqual(bridge["state"], "error")
        run.assert_called_once()

    @patch("model_router.claude_opus_bridge.dispatch")
    def test_delegated_reviews_use_their_own_configured_turn_limit(self, bridge):
        from model_router import _run_opus5_bridge

        bridge.return_value = {"result": "reviewed"}
        _run_opus5_bridge(
            repo="/tmp", task="[sonnet-review] Review parser", write=False, review=True,
            cfg={"coding_agent": {"max_turns": 3, "delegated_review": {"max_turns": 17}}},
        )
        self.assertEqual(bridge.call_args.kwargs["max_turns"], 17)

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_lifecycle_records_started_then_one_terminal_with_parent_and_precedence(self, run, log):
        run.return_value.returncode = 1
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({
            "subtype": "error_max_turns", "is_error": True,
            "modelUsage": {"claude-sonnet-4": {}}, "num_turns": 8,
        })
        with tempfile.TemporaryDirectory() as directory:
            lifecycle = Path(directory) / "bridge.jsonl"
            with self.assertRaisesRegex(RuntimeError, "max turns"):
                dispatch(
                    "[opus-review] Review only", Path(directory), review=True,
                    parent_session_id="parent-session", parent_turn_id="parent-turn",
                    lifecycle_path=lifecycle, max_turns=12, max_budget_usd=2.5,
                )
            events = [json.loads(line) for line in lifecycle.read_text().splitlines()]
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--max-turns") + 1], "12")
        self.assertEqual(command[command.index("--max-budget-usd") + 1], "2.50")
        self.assertEqual([event["event"] for event in events], ["started", "terminal"])
        self.assertEqual({event["bridge_run_id"] for event in events}, {events[0]["bridge_run_id"]})
        self.assertEqual(events[0]["parent_session_id"], "parent-session")
        self.assertEqual(events[0]["parent_turn_id"], "parent-turn")
        self.assertEqual(events[1]["state"], "max-turn")
        self.assertNotIn("task", events[0])
        self.assertNotIn("prompt", json.dumps(events))

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_minimum_supported_budget_is_preserved_in_argv(self, run, logged):
        run.return_value.returncode = 0
        run.return_value.stdout = json.dumps({"modelUsage": {CANONICAL_OPUS_MODEL: {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            dispatch("[opus-review] Review parser", Path(directory), review=True,
                     max_turns=1, max_budget_usd=0.01)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--max-turns") + 1], "1")
        self.assertEqual(command[command.index("--max-budget-usd") + 1], "0.01")

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run", side_effect=__import__("subprocess").TimeoutExpired("claude", 1))
    def test_timeout_is_terminal_and_has_highest_precedence(self, run, log):
        with tempfile.TemporaryDirectory() as directory:
            lifecycle = Path(directory) / "bridge.jsonl"
            with self.assertRaises(ClaudeBridgeFailure) as raised:
                dispatch("[opus-review] Review only", Path(directory), review=True, timeout=1, lifecycle_path=lifecycle)
            self.assertEqual(raised.exception.failure_kind, "timeout")
            events = lifecycle.read_text().splitlines()
            terminal = json.loads(events[-1])
        self.assertEqual(terminal["state"], "timeout")
        self.assertEqual(len(events), 2)

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_parseable_non_object_output_records_terminal_error_before_raising(self, run, log):
        for stdout in ("[]", "null", '"not a result object"'):
            with self.subTest(stdout=stdout), tempfile.TemporaryDirectory() as directory:
                run.return_value.returncode = 1
                run.return_value.stderr = "CLI error"
                run.return_value.stdout = stdout
                lifecycle = Path(directory) / "bridge.jsonl"
                with self.assertRaises(ClaudeBridgeFailure) as raised:
                    dispatch("[opus-review] Review parser", Path(directory), review=True, lifecycle_path=lifecycle)
                self.assertEqual(raised.exception.failure_kind, "malformed-json")
                events = [json.loads(line) for line in lifecycle.read_text().splitlines()]
            self.assertEqual([event["event"] for event in events], ["started", "terminal"])
            self.assertEqual(events[-1]["state"], "error")
            self.assertTrue(events[-1]["malformed"])
            self.assertEqual(events[-1]["returncode"], 1)
        log.assert_not_called()

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_cli_failures_are_typed_and_keep_one_terminal_lifecycle_record(self, run, log):
        cases = (
            ("max-turn", {"subtype": "error_max_turns", "is_error": True,
                           "modelUsage": {CANONICAL_OPUS_MODEL: {}}}, 0),
            ("budget", {"subtype": "error_budget", "is_error": True,
                        "modelUsage": {CANONICAL_OPUS_MODEL: {}}}, 0),
            ("nonzero-exit", {"modelUsage": {CANONICAL_OPUS_MODEL: {}}}, 2),
            ("model-mismatch", {"modelUsage": {"claude-sonnet-5": {}}, "result": "no"}, 0),
        )
        with tempfile.TemporaryDirectory() as directory:
            for expected, payload, returncode in cases:
                with self.subTest(expected=expected):
                    lifecycle = Path(directory) / f"{expected}.jsonl"
                    run.return_value.returncode = returncode
                    run.return_value.stderr = "CLI error"
                    run.return_value.stdout = json.dumps(payload)
                    with self.assertRaises(ClaudeBridgeFailure) as raised:
                        dispatch("[opus-review] Review parser", Path(directory), review=True,
                                 lifecycle_path=lifecycle)
                    self.assertEqual(raised.exception.failure_kind, expected)
                    events = [json.loads(line) for line in lifecycle.read_text().splitlines()]
                    self.assertEqual([event["event"] for event in events], ["started", "terminal"])
        log.assert_not_called()

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_valid_object_with_nonzero_exit_records_terminal_error(self, run, log):
        run.return_value.returncode = 1
        run.return_value.stderr = "CLI error"
        run.return_value.stdout = json.dumps({"modelUsage": {CANONICAL_OPUS_MODEL: {}}, "result": "nope"})
        with tempfile.TemporaryDirectory() as directory:
            lifecycle = Path(directory) / "bridge.jsonl"
            with self.assertRaisesRegex(RuntimeError, "failed with exit 1"):
                dispatch("[opus-review] Review parser", Path(directory), review=True, lifecycle_path=lifecycle)
            events = [json.loads(line) for line in lifecycle.read_text().splitlines()]
        self.assertEqual([event["state"] for event in events], ["running", "error"])
        self.assertFalse(events[-1].get("malformed", False))
        log.assert_not_called()

    def test_classifier_is_conservative_and_manual_override_is_available(self):
        self.assertEqual(classify_coding_dispatch("[opus5] Implement the parser")[0], True)
        self.assertEqual(classify_coding_dispatch("[opus5] Implement this bounded CSS label fix")[0], True)
        self.assertEqual(classify_coding_dispatch("Debug the backend parser")[0], True)
        self.assertEqual(classify_coding_dispatch("Write product UI CSS")[0], False)
        self.assertEqual(classify_coding_dispatch("Say hello")[0], False)

    def test_design_veto_applies_to_accented_hungarian(self):
        """The Hungarian design terms are spelled unaccented, so matching raw
        text let real Hungarian design work past the Sol-only veto."""
        for task in (
            "Refaktoráld a wireframe komponens tipográfiáját.",
            "Igazítsd a felület színpalettáját.",
        ):
            with self.subTest(task=task):
                eligible, reason = classify_coding_dispatch(task)
                self.assertFalse(eligible, task)
                self.assertIn("Sol-only", reason)

    def test_explicit_review_accepts_non_coding_and_design_work_read_only(self):
        for task in (
            "[opus-review] Vizsgáld felül ezt a jogosultsági tervet; ne módosíts fájlt.",
            "[opus-review] Review the UI hierarchy and accessibility risks; do not edit files.",
        ):
            with self.subTest(task=task):
                eligible, reason = classify_review_dispatch(task)
                self.assertTrue(eligible, task)
                self.assertIn("review", reason.casefold())
        self.assertFalse(classify_review_dispatch("Review this plan")[0])

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_dispatch_logs_only_the_actual_canonical_opus_model(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {CANONICAL_OPUS_MODEL: {}}, "session_id": "verified", "num_turns": 1, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            result = dispatch("[opus] Implement a parser test", Path(directory))
        self.assertEqual(result["effective_model"], CANONICAL_OPUS_MODEL)
        command = run.call_args.args[0]
        self.assertIn("--model", command)
        self.assertIn("opus", command)
        # Opus results are accepted only when Claude reports canonical Opus, so
        # do not ask the CLI to spend a rejected Sonnet fallback.
        self.assertNotIn("--fallback-model", command)
        self.assertIn("--tools", command)
        self.assertEqual(command[command.index("--tools") + 1], "Read")
        self.assertIn("--disallowedTools", command)
        self.assertNotIn("--dangerously-skip-permissions", command)
        logged = log.call_args.args[0]
        self.assertEqual((logged.tier, logged.model, logged.effort), ("opus5", CANONICAL_OPUS_MODEL, "external"))

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_review_keeps_prompt_off_argv_and_enforces_600_second_floor(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {CANONICAL_OPUS_MODEL: {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            dispatch("[opus-review] Review only", Path(directory), review=True, timeout=300)
        command = run.call_args.args[0]
        self.assertNotIn("[opus-review] Review only", command)
        self.assertEqual(run.call_args.kwargs["input"].split("\n", 1)[0], "[opus-review] Review only")
        self.assertEqual(run.call_args.kwargs["timeout"], 600)

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_preferred_step_down_validates_the_admitted_tier_and_keeps_requested_identity(self, run, log):
        from model_router.target_identity import resolve_target

        cfg = {"models": {"opus5": "claude-opus-5-5", "sonnet5": "claude-sonnet-5-5"}}
        identity = resolve_target("opus", transport="claude_cli", cfg=cfg).identity
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {"claude-sonnet-5-5": {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            result = dispatch("[opus-review] Review parser", Path(directory), review=True,
                              model="sonnet", requested_alias="opus", identity=identity)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], "sonnet")
        self.assertEqual(result["effective_model"], "claude-sonnet-5-5")
        self.assertEqual(result["identity"]["requested"]["value"], "claude-opus-5-5")
        self.assertEqual(result["substitution"]["observed"], "claude-sonnet-5-5")

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_conflicting_top_level_model_cannot_override_model_usage_evidence(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({
            "model": CANONICAL_OPUS_MODEL,
            "modelUsage": {"claude-sonnet-5-5": {}},
            "result": "wrong tier",
        })
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ClaudeBridgeFailure) as raised:
                dispatch("[opus-review] Review parser", Path(directory), review=True)
        self.assertEqual(raised.exception.failure_kind, "model-mismatch")
        log.assert_not_called()

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_exact_canonical_review_requests_and_accepts_the_canonical_model(self, run, log):
        from model_router.target_identity import resolve_target
        from model_router import runtime_capabilities as runtime

        caps = (runtime.Capability("exact_model", "supported", reason="fixture capability evidence"),)
        snapshot = runtime.RuntimeSnapshot(
            1, "fixture", runtime.Fact("unknown"), runtime.Fact("unknown"),
            runtime.Fact("unknown"), (), (runtime.AdapterCapabilities("claude_cli", caps),), (),
        )
        cfg = {"models": {"opus5": CANONICAL_OPUS_MODEL}}
        identity = resolve_target("opus", transport="claude_cli", selection_mode="exact",
                                  requested_model=CANONICAL_OPUS_MODEL, cfg=cfg, snapshot=snapshot).identity
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {CANONICAL_OPUS_MODEL: {}}, "result": "ok"})
        # F03: the bridge re-checks exact capability itself; the fixture evidence
        # is injected at that seam too (the installed CLI reports it unknown).
        with tempfile.TemporaryDirectory() as directory, \
                patch("model_router.target_identity._cli_exact_capability",
                      return_value=("supported", "fixture capability evidence")):
            result = dispatch("[opus-review] Review parser", Path(directory), review=True, identity=identity)
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], CANONICAL_OPUS_MODEL)
        self.assertEqual(result["effective_model"], CANONICAL_OPUS_MODEL)

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_the_review_label_selects_the_claude_tier(self, run, log):
        """Two tiers exist so routine review can spend the cheaper one; a single
        tier would burn the separate quota that is the reason to reach Claude."""
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {"claude-sonnet-5-5": {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            out = dispatch("[sonnet-review] Review only", Path(directory), review=True,
                           lifecycle_path=Path(directory) / "bridge.jsonl")
        command = run.call_args.args[0]
        self.assertEqual(command[command.index("--model") + 1], "sonnet")
        # Sonnet has no lower tier worth accepting: the result is trusted on the
        # strength of the model that produced it, so a silent drop must not happen.
        self.assertNotIn("--fallback-model", command)
        self.assertEqual(out["effective_model"], "claude-sonnet-5-5")
        self.assertEqual((log.call_args.args[0].tier, log.call_args.args[0].model),
                         ("sonnet5", "claude-sonnet-5-5"))

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_a_review_that_served_another_tier_is_rejected(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {"claude-sonnet-5-5": {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(RuntimeError):
                dispatch("[opus-review] Review only", Path(directory), review=True)

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_explicit_opus_review_accepts_canonical_opus_with_internal_subtask_usage(self, run, log):
        """The requested alias is satisfied by the canonical primary route, not a pure usage map."""
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({
            "modelUsage": {
                CANONICAL_OPUS_MODEL: {"inputTokens": 100, "outputTokens": 50},
                "claude-haiku-4-5": {"inputTokens": 10, "outputTokens": 5},
            },
            "result": "review complete",
        })
        with tempfile.TemporaryDirectory() as directory:
            result = dispatch("[opus-review] Review only", Path(directory), review=True)
        self.assertEqual(result["effective_model"], CANONICAL_OPUS_MODEL)
        self.assertEqual(run.call_args.args[0][run.call_args.args[0].index("--model") + 1], "opus")
        logged = log.call_args.args[0]
        self.assertEqual((logged.tier, logged.model), ("opus5", CANONICAL_OPUS_MODEL))

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_dispatch_rejects_alias_or_fallback_as_effective_route(self, run, log):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {"claude-sonnet-4": {}}, "result": "ok"})
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(RuntimeError, "did not serve"):
                dispatch("[opus] Implement a parser test", Path(directory))
        log.assert_not_called()

    def test_a05_public_cli_dispatch_crosses_adapter_boundary(self):
        from model_router import claude_opus_bridge as cli
        from model_router import execution_adapters as adapters

        book = adapters.ReservationBook()
        payload = {'subtype': 'success', 'result': 'verdict',
                   'modelUsage': {'claude-sonnet-5-5': {}}, 'num_turns': 1}
        with patch.object(adapters, 'RESERVATIONS', book), \
             patch.object(cli, '_load_config', return_value={}), \
             patch.object(cli, '_log_decision'), \
             patch.object(cli.subprocess, 'run', return_value=SimpleNamespace(
                 stdout=json.dumps(payload), stderr='', returncode=0)):
            cli.dispatch('[sonnet-review] Review parser', Path('/tmp'), review=True, model='sonnet')
        self.assertEqual(len(book.records()), 1, 'public CLI dispatch bypassed adapter journal')


_SONNET_SUCCESS = {"subtype": "success", "result": "verdict", "modelUsage": {"claude-sonnet-5-5": {}},
                   "num_turns": 1}
_SONNET_MISMATCH = {"subtype": "success", "result": "verdict", "modelUsage": {"claude-opus-5-5": {}},
                    "num_turns": 1}


class CliEntrypointBoundaryTests(unittest.TestCase):
    """F03 (A05): the standalone ``main()`` (package and script copy), the public
    Python ``dispatch`` and the middleware bridge each cross one normalised CLI
    boundary exactly once: one raw execution, one receipt, one lifecycle pair and
    at most one routed-call record. Never a live CLI: ``subprocess.run`` is mocked."""

    ENTRIES = ("public", "standalone", "script", "middleware")
    TASK = "[sonnet-review] Review parser"

    def setUp(self):
        from model_router import execution_adapters
        self.book = execution_adapters.ReservationBook()
        owner = patch.object(execution_adapters, "RESERVATIONS", self.book)
        owner.start()
        self.addCleanup(owner.stop)

    def _module(self, entry):
        if entry == "script":
            import claude_opus_bridge as script  # the harness's script-mode copy
            return script
        from model_router import claude_opus_bridge as package
        return package

    def _invoke(self, entry, module, repo, lifecycle, *, task=None, identity=None):
        import io
        import sys
        from contextlib import redirect_stdout
        task = task or self.TASK
        if entry == "public":
            return module.dispatch(task, Path(repo), review=True, model="sonnet", identity=identity,
                                   lifecycle_path=lifecycle)
        if entry in ("standalone", "script"):
            argv = ["claude_opus_bridge.py", "--repo", str(repo), "--task", task, "--review",
                    "--lifecycle-path", str(lifecycle)]
            out = io.StringIO()
            with patch.object(sys, "argv", argv), redirect_stdout(out):
                self.assertEqual(module.main(), 0)
            return json.loads(out.getvalue())
        import model_router as router
        return router._run_opus5_bridge(repo=str(repo), task=task, write=False, review=True, model="sonnet",
                                        identity=identity,
                                        cfg={"coding_agent": {"lifecycle_path": str(lifecycle)}})

    def _run(self, entry, *, outcome="success", repo_name="", task=None, identity=None, patches=()):
        from contextlib import ExitStack
        module = self._module(entry)
        if outcome == "timeout":
            effect = {"side_effect": __import__("subprocess").TimeoutExpired("claude", 1)}
        elif outcome == "start-failure":
            effect = {"side_effect": FileNotFoundError("claude")}
        else:
            stdout = {"success": json.dumps(_SONNET_SUCCESS), "model-mismatch": json.dumps(_SONNET_MISMATCH),
                      "malformed": "not json"}[outcome]
            effect = {"return_value": SimpleNamespace(stdout=stdout, stderr="", returncode=0)}
        with tempfile.TemporaryDirectory() as directory:
            routes = Path(directory) / "routes.jsonl"
            lifecycle = Path(directory) / "bridge.jsonl"
            repo = Path(directory) / repo_name if repo_name else Path(directory)
            cfg = {"logging": {"enabled": True, "path": str(routes)}}
            result = error = None
            with ExitStack() as stack:
                stack.enter_context(patch.object(module, "_load_config", return_value=cfg))
                run = stack.enter_context(patch.object(module.subprocess, "run", **effect))
                for item in patches:
                    stack.enter_context(item)
                try:
                    result = self._invoke(entry, module, repo, lifecycle, task=task, identity=identity)
                except Exception as exc:  # the assertions below name the expected kind
                    error = exc
            # Counted inside the scratch directory, before cleanup removes it.
            events = [json.loads(line) for line in lifecycle.read_text().splitlines()] if lifecycle.exists() else []
            route_lines = ([line for line in routes.read_text().splitlines() if line.strip()]
                           if routes.exists() else [])
        return SimpleNamespace(result=result, error=error, run=run, events=events, routes=route_lines,
                               receipts=self.book.records())

    def _assert_one_attempt(self, out, *, status, state, routes):
        self.assertEqual(out.run.call_count, 1, "exactly one raw CLI execution")
        self.assertEqual(len(out.receipts), 1, "exactly one normalised receipt")
        receipt = out.receipts[0]
        self.assertEqual((receipt.transport, receipt.scope), ("claude_cli", "claude_cli:local_credentials"))
        self.assertEqual(receipt.status, status)
        self.assertEqual([e["event"] for e in out.events], ["started", "terminal"], "one lifecycle pair")
        self.assertEqual(out.events[-1]["state"], state)
        self.assertEqual(len(out.routes), routes, "routed-call records")
        return receipt

    def test_success_crosses_the_boundary_once_with_truthful_legacy_identity(self):
        for entry in self.ENTRIES:
            with self.subTest(entry=entry):
                self.book._claims.clear()
                out = self._run(entry)
                self.assertIsNone(out.error)
                receipt = self._assert_one_attempt(out, status="succeeded", state="success", routes=1)
                self.assertEqual(receipt.observed_model, "claude-sonnet-5-5")
                self.assertEqual(receipt.resolved_tier, "sonnet")
                self.assertEqual(receipt.handle, out.result["bridge_run_id"])
                self.assertEqual(receipt.handle, out.events[0]["bridge_run_id"])
                identity = out.result["identity"]
                self.assertIsNotNone(identity, "a legacy caller gets truthful legacy identity facts")
                # Legacy facts come from the explicit tier and actual command; the
                # served model is observed only from validated CLI evidence.
                self.assertEqual(identity["transport"], "claude_cli")
                self.assertEqual(identity["selection_mode"], "profile_preferred")
                self.assertEqual(identity["requested"]["value"], "claude-sonnet-5-5")
                self.assertEqual((identity["resolved"]["value"], identity["resolved"]["canonical"]),
                                 ("sonnet", False))
                self.assertEqual(identity["observed"],
                                 {"value": "claude-sonnet-5-5", "source": "claude_cli.result.modelUsage",
                                  "canonical": True})
                self.assertEqual(identity["account"], "unknown")
                self.assertEqual(identity["effort"]["applied"], "unknown")

    def test_attempted_failures_record_one_terminal_receipt_and_no_route(self):
        cases = (("model-mismatch", "model-mismatch", "failed", "error"),
                 ("timeout", "timeout", "timed_out", "timeout"),
                 ("malformed", "malformed-json", "failed", "error"))
        for entry in self.ENTRIES:
            for outcome, kind, status, state in cases:
                with self.subTest(entry=entry, outcome=outcome):
                    self.book._claims.clear()
                    out = self._run(entry, outcome=outcome)
                    self.assertEqual(type(out.error).__name__, "ClaudeBridgeFailure")
                    self.assertEqual(out.error.failure_kind, kind)
                    self.assertFalse(out.error.refused, "an attempted failure is not a refusal")
                    self._assert_one_attempt(out, status=status, state=state, routes=0)

    def test_process_start_failure_stays_unknown_with_one_terminal_record(self):
        for entry in self.ENTRIES:
            with self.subTest(entry=entry):
                self.book._claims.clear()
                out = self._run(entry, outcome="start-failure")
                self.assertIsInstance(out.error, FileNotFoundError)
                self._assert_one_attempt(out, status="unknown", state="error", routes=0)

    def test_invalid_input_never_launches_and_writes_no_receipt(self):
        for entry in self.ENTRIES:
            for name, options in (("missing repository", {"repo_name": "missing"}),
                                  ("unlabelled review", {"task": "Review parser"})):
                with self.subTest(entry=entry, invalid=name):
                    self.book._claims.clear()
                    out = self._run(entry, **options)
                    self.assertIsInstance(out.error, ValueError)
                    out.run.assert_not_called()
                    self.assertEqual(out.receipts, ())
                    self.assertEqual((out.events, out.routes), ([], []))

    def _exact_identity(self):
        from model_router.execution_contracts import ModelFact, TargetIdentity
        return TargetIdentity("anthropic", "unknown", "claude_cli", "sonnet", "exact",
                              requested=ModelFact("claude-sonnet-5-5", "operator_request"))

    def test_direct_exact_invocation_without_capability_is_refused_before_launch(self):
        for entry in ("public", "middleware"):
            with self.subTest(entry=entry):
                self.book._claims.clear()
                out = self._run(entry, identity=self._exact_identity(), patches=(
                    patch("model_router.target_identity._cli_exact_capability",
                          return_value=("unknown", "fixture: no exact_model evidence")),))
                self.assertEqual(type(out.error).__name__, "ClaudeBridgeFailure")
                self.assertTrue(out.error.refused, "never-launched refusal")
                self.assertEqual(out.error.failure_kind, "capability")
                self.assertIn("exact_model", str(out.error))
                out.run.assert_not_called()
                self.assertEqual(out.receipts, ())
                self.assertEqual((out.events, out.routes), ([], []))

    def test_direct_exact_invocation_with_capability_launches_once_and_validates(self):
        for entry in ("public", "middleware"):
            for outcome, status in (("success", "succeeded"), ("model-mismatch", "failed")):
                with self.subTest(entry=entry, outcome=outcome):
                    self.book._claims.clear()
                    out = self._run(entry, outcome=outcome, identity=self._exact_identity(), patches=(
                        patch("model_router.target_identity._cli_exact_capability",
                              return_value=("supported", "fixture capability evidence")),))
                    command = out.run.call_args.args[0]
                    self.assertEqual(command[command.index("--model") + 1], "claude-sonnet-5-5")
                    self._assert_one_attempt(out, status=status, state="success" if status == "succeeded"
                                             else "error", routes=1 if status == "succeeded" else 0)
                    if outcome == "model-mismatch":
                        self.assertEqual(out.error.failure_kind, "model-mismatch")
                        self.assertFalse(out.error.refused)

    def test_explicit_caller_identity_stays_authoritative(self):
        from model_router.execution_contracts import ModelFact, TargetIdentity
        caller = TargetIdentity("anthropic", "unknown", "claude_cli", "sonnet", "profile_preferred",
                                requested=ModelFact("claude-sonnet-5-5", "caller_fixture"),
                                resolved=ModelFact("sonnet", "caller_fixture_alias", canonical=False))
        out = self._run("public", identity=caller)
        self.assertIsNone(out.error)
        self.assertEqual(out.result["identity"]["requested"]["source"], "caller_fixture")
        self.assertEqual(out.result["identity"]["resolved"]["source"], "caller_fixture_alias")
        self.assertEqual(len(out.receipts), 1)

    def test_preferred_identity_reports_anthropic_transport_owner(self):
        from model_router.execution_contracts import ModelFact, TargetIdentity
        caller = TargetIdentity("openai-codex", "unknown", "claude_cli", "sonnet", "profile_preferred",
                                requested=ModelFact("claude-sonnet-5-5", "caller_fixture"),
                                resolved=ModelFact("sonnet", "caller_fixture_alias", canonical=False))
        out = self._run("public", identity=caller)
        self.assertIsNone(out.error)
        out.run.assert_called_once()
        self.assertEqual(len(out.receipts), 1)
        self.assertEqual(out.result["identity"]["provider"], "anthropic")
        self.assertEqual(out.result["identity"]["requested"]["source"], "caller_fixture")
        self.assertEqual(out.result["identity"]["observed"]["value"], "claude-sonnet-5-5")

    def test_exact_identity_with_wrong_provider_is_refused_before_launch(self):
        identity = self._exact_identity()
        from dataclasses import replace
        wrong_provider = replace(identity, provider="openai-codex")
        out = self._run("public", identity=wrong_provider, patches=(
            patch("model_router.target_identity._cli_exact_capability",
                  return_value=("supported", "fixture capability evidence")),))
        self.assertEqual(type(out.error).__name__, "ClaudeBridgeFailure")
        self.assertTrue(out.error.refused)
        self.assertEqual(out.error.failure_class, "capability")
        out.run.assert_not_called()
        self.assertEqual(out.receipts, ())
        self.assertEqual((out.events, out.routes), ([], []))

    def test_step_down_keeps_requested_admitted_and_observed_distinct(self):
        from model_router import claude_opus_bridge as cli
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cli, "_load_config", return_value={}), \
                patch.object(cli.subprocess, "run", return_value=SimpleNamespace(
                    stdout=json.dumps(_SONNET_SUCCESS), stderr="", returncode=0)):
            result = cli.dispatch("[opus-review] Review parser", Path(directory), review=True, model="sonnet",
                                  requested_alias="opus", adjustment="opus5→sonnet5 (weekly usage 75%)")
        identity = result["identity"]
        self.assertIsNotNone(identity, "a legacy step-down keeps requested/admitted/observed facts")
        self.assertEqual(identity["alias"], "opus")
        self.assertEqual(identity["requested"]["value"], "claude-opus-5-5")
        self.assertEqual((identity["resolved"]["value"], identity["resolved"]["canonical"]), ("sonnet", False))
        self.assertEqual(identity["observed"]["value"], "claude-sonnet-5-5")
        self.assertEqual(result["substitution"]["observed"], "claude-sonnet-5-5")
        receipt, = self.book.records()
        self.assertEqual(receipt.resolved_tier, "sonnet")
        self.assertIn("weekly usage 75%", receipt.adjustment)

    def test_missing_invocation_ids_do_not_deduplicate_later_calls(self):
        first = self._run("public")
        second = self._run("public")
        self.assertIsNone(second.error)
        self.assertEqual(len(second.receipts), 2, "two new calls, two observation-only receipts")
        self.assertNotEqual(first.result["bridge_run_id"], second.result["bridge_run_id"])
        self.assertEqual(len({r.attempt_key for r in second.receipts}), 2)

    def test_script_mode_shares_the_package_boundary_without_duplicates(self):
        import sys
        out = self._run("script")
        self.assertEqual(len(out.receipts), 1, "script-mode receipt reached the package journal")
        self.assertNotIn("execution_adapters", sys.modules, "a second top-level adapter module was imported")

    def test_private_raw_operation_never_enters_the_boundary(self):
        from model_router import claude_opus_bridge as cli
        from model_router import execution_adapters as adapters
        self.assertTrue(hasattr(cli, "_run_cli"))
        with tempfile.TemporaryDirectory() as directory, \
                patch.object(cli, "_load_config", return_value={}), \
                patch.object(adapters, "dispatch_legacy", side_effect=AssertionError("re-entered")), \
                patch.object(cli.subprocess, "run", return_value=SimpleNamespace(
                    stdout=json.dumps(_SONNET_SUCCESS), stderr="", returncode=0)) as run:
            plan = cli._prepare(self.TASK, Path(directory), review=True, model="sonnet")
            cli._run_cli(plan)
        run.assert_called_once()
        self.assertEqual(self.book.records(), ())


if __name__ == "__main__":
    unittest.main()


class AdjustmentEvidenceTests(unittest.TestCase):
    def setUp(self):
        from model_router import execution_adapters
        owner = patch.object(execution_adapters, "RESERVATIONS", execution_adapters.ReservationBook())
        owner.start()
        self.addCleanup(owner.stop)

    @patch("claude_opus_bridge._log_decision")
    @patch("claude_opus_bridge.subprocess.run")
    def test_requested_and_effective_tiers_reach_lifecycle_and_route_log(self, run, logged):
        run.return_value.returncode = 0
        run.return_value.stderr = ""
        run.return_value.stdout = json.dumps({"modelUsage": {"claude-sonnet-5-5": {}},
                                              "result": "reviewed"})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "bridge.jsonl"
            dispatch("[opus-review] Review parser", Path(directory), review=True, model="sonnet",
                     requested_alias="opus", adjustment="opus5→sonnet5 (weekly usage 75%)",
                     parent_session_id="parent", lifecycle_path=path)
            events = [json.loads(line) for line in path.read_text().splitlines()]
        for event in events:
            self.assertEqual(event["requested_tier"], "opus")
            self.assertEqual(event["effective_tier"], "sonnet")
            self.assertIn("weekly usage 75%", event["adjusted"])
        self.assertEqual(events[-1]["canonical_model"], "claude-sonnet-5-5")
        self.assertIn("usage soft limit: opus5→sonnet5", logged.call_args.args[0].reason)


class HostPackageImportTests(unittest.TestCase):
    """Hermes loads the plugin as ``hermes_plugins.model_router`` from a directory
    named ``model-router``, so no top-level ``model_router`` package exists there.
    The bridge must import inside that package, or every caller of it (the
    delegate_task admission guard among them) raises ModuleNotFoundError."""

    def test_bridge_imports_when_the_plugin_is_loaded_under_the_host_package_name(self):
        import os
        import subprocess
        import sys

        plugin_dir = str(Path(__file__).resolve().parent)
        probe = (
            "import importlib, importlib.util, sys, types\n"
            "sys.modules['hermes_plugins'] = types.ModuleType('hermes_plugins')\n"
            "sys.modules['hermes_plugins'].__path__ = []\n"
            f"spec = importlib.util.spec_from_file_location('hermes_plugins.model_router', {plugin_dir + '/__init__.py'!r}, submodule_search_locations=[{plugin_dir!r}])\n"
            "plugin = importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name] = plugin\n"
            "spec.loader.exec_module(plugin)\n"
            "bridge = importlib.import_module('hermes_plugins.model_router.claude_opus_bridge')\n"
            "assert 'model_router' not in sys.modules, 'a second, top-level copy of the router was imported'\n"
            "assert bridge.RouteDecision is plugin.RouteDecision\n"
            "print('OK')\n"
        )
        with tempfile.TemporaryDirectory() as cwd:
            env = {"HOME": cwd, "HERMES_HOME": cwd + "/.hermes", "PATH": os.environ.get("PATH", "")}
            with patch("subprocess.run", wraps=subprocess.run) as run:
                result = subprocess.run(
                    [sys.executable, "-c", probe], cwd=cwd, env=env, text=True,
                    capture_output=True, timeout=60,
                )
        self.assertEqual(run.call_args.kwargs["timeout"], 60)
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip(), "OK")

    def test_a_package_load_leaves_sys_path_unchanged(self):
        """B-M1: the sys.path.insert for script mode must not run when the bridge
        is loaded as part of the plugin package -- it must stay inside the
        `else` branch of `if __package__:`."""
        import os
        import subprocess
        import sys

        plugin_dir = str(Path(__file__).resolve().parent)
        plugin_parent = str(Path(plugin_dir).resolve().parent)
        probe = (
            "import importlib, importlib.util, sys, types\n"
            "sys.modules['hermes_plugins'] = types.ModuleType('hermes_plugins')\n"
            "sys.modules['hermes_plugins'].__path__ = []\n"
            f"spec = importlib.util.spec_from_file_location('hermes_plugins.model_router', {plugin_dir + '/__init__.py'!r}, submodule_search_locations=[{plugin_dir!r}])\n"
            "plugin = importlib.util.module_from_spec(spec)\n"
            "sys.modules[spec.name] = plugin\n"
            "spec.loader.exec_module(plugin)\n"
            "before = list(sys.path)\n"
            "importlib.import_module('hermes_plugins.model_router.claude_opus_bridge')\n"
            f"assert {plugin_parent!r} not in sys.path, 'script-mode sys.path.insert ran for a package load'\n"
            "print('OK')\n"
        )
        with tempfile.TemporaryDirectory() as cwd:
            env = {"HOME": cwd, "HERMES_HOME": cwd + "/.hermes", "PATH": os.environ.get("PATH", "")}
            result = subprocess.run(
                [sys.executable, "-c", probe], cwd=cwd, env=env, text=True,
                capture_output=True, timeout=60,
            )
        self.assertEqual(result.returncode, 0, result.stderr[-2000:])
        self.assertEqual(result.stdout.strip(), "OK")
