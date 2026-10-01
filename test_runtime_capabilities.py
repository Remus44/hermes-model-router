"""Offline compatibility matrix: capability evidence is not dispatch/admission."""
from copy import deepcopy
from dataclasses import FrozenInstanceError
from concurrent.futures import ThreadPoolExecutor
import json
import sys
import unittest
from unittest.mock import patch

import model_router as router
from model_router import claude_delegation
from model_router.host_delegation_fixtures import (
    host_delegation, delegate_task_request, DIRECT_CLAUDE_TOOLS, DEFERRED_CLAUDE_TOOLS,
)
from model_router import runtime_capabilities as runtime


class RuntimeCapabilitiesTests(unittest.TestCase):
    def setUp(self):
        # Import only the safe host modules, never run_agent or a real child.
        import tools.delegate_tool as host
        self.host = host
        self.cfg = {"orchestration": {"enabled": True, "conductor": "sonnet5"},
                    "callable": {"sonnet5": True}, "claude_delegation": {"enabled": True}}
        runtime.clear_cache()
        self.addCleanup(runtime.clear_cache)

    def snapshot(self, request=None):
        return runtime.snapshot(request if request is not None else delegate_task_request(), self.cfg)

    def _shapes(self, name):
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA
        params = DELEGATE_TASK_SCHEMA['parameters']
        return {
            'flat': {'name': name, 'parameters': params},
            'openai': {'type': 'function', 'function': {'name': name, 'parameters': params}},
            'anthropic': {'name': name, 'input_schema': params},
        }

    def test_a09_supported_delegate_task_names_across_shapes(self):
        for name in ('delegate_task', 'mcp__delegate_task'):
            for shape, tool in self._shapes(name).items():
                with self.subTest(name=name, shape=shape):
                    runtime.clear_cache()
                    cap = runtime.snapshot({'tools': [tool]}, {}).adapter('hermes_codex').capability('submission')
                    self.assertEqual(cap.status, 'supported', cap.reason)
                    mp = runtime.snapshot({'tools': [tool]}, {}).adapter('hermes_codex').capability('model_parameter')
                    # The installed host schema is batch-only: no model parameter.
                    self.assertEqual(mp.status, 'unsupported', mp.reason)

    def test_a09_supported_delegate_claude_names_across_shapes(self):
        for name in ('delegate_claude', 'mcp__delegate_claude'):
            for shape, tool in self._shapes(name).items():
                with self.subTest(name=name, shape=shape):
                    runtime.clear_cache()
                    with patch.object(claude_delegation, 'is_active', return_value=True):
                        cap = runtime.snapshot({'tools': [tool]}, self.cfg).adapter('hermes_claude').capability('submission')
                    self.assertEqual(cap.status, 'supported', cap.reason)

    def test_claude_submission_rejects_malformed_or_missing_schema_across_names_and_shapes(self):
        for name in ('delegate_claude', 'mcp__delegate_claude'):
            for shape in ('flat', 'openai', 'anthropic'):
                malformed = self._shapes(name)[shape]
                missing = self._shapes(name)[shape]
                if shape == 'openai':
                    malformed['function']['parameters'] = 'malformed'
                    missing['function'].pop('parameters')
                elif shape == 'anthropic':
                    malformed['input_schema'] = 'malformed'
                    missing.pop('input_schema')
                else:
                    malformed['parameters'] = 'malformed'
                    missing.pop('parameters')
                for label, tool in (('malformed', malformed), ('missing', missing)):
                    with self.subTest(name=name, shape=shape, label=label), host_delegation(depth=2), \
                         patch.object(claude_delegation, 'is_active', return_value=True):
                        runtime.clear_cache()
                        snap = self.snapshot({'tools': [tool]})
                        cap = snap.adapter('hermes_claude').capability('submission')
                        choice = runtime.resolve_topology(snap, 'nested_conductor', transport='hermes_claude')
                        self.assertEqual(cap.status, 'unknown')
                        self.assertEqual(cap.reason, 'delegate_claude schema is missing or malformed')
                        self.assertEqual((choice.status, choice.selected), ('unsupported', 'parent_direct'))

    def test_claude_submission_deferred_tool_is_unknown_and_refuses_nested_topology(self):
        with host_delegation(depth=2), patch.object(claude_delegation, 'is_active', return_value=True):
            snap = self.snapshot({'tools': [{'name': 'terminal'}], 'deferred_tools': ['mcp__delegate_claude']})
            cap = snap.adapter('hermes_claude').capability('submission')
            choice = runtime.resolve_topology(snap, 'nested_conductor', transport='hermes_claude')
        self.assertEqual(cap.status, 'unknown')
        self.assertEqual(cap.reason, 'request does not carry delegate_claude (absent or deferred)')
        self.assertEqual((choice.status, choice.selected), ('unsupported', 'parent_direct'))

    def test_a09_arbitrary_prefix_and_unrelated_tools_do_not_count(self):
        for name in ('other__delegate_task', 'mcp__x__delegate_task', 'MCP__delegate_task',
                     'mcp__delegate_task_extra', 'delegate_claude', 'mcp__delegate_claude'):
            tool = self._shapes(name)['flat']
            runtime.clear_cache()
            cap = runtime.snapshot({'tools': [tool]}, {}).adapter('hermes_codex').capability('submission')
            self.assertEqual(cap.status, 'unknown', f'{name}: {cap.reason}')
        runtime.clear_cache()
        with patch.object(claude_delegation, 'is_active', return_value=True):
            cap = runtime.snapshot({'tools': [self._shapes('other__delegate_claude')['flat']]}, self.cfg
                                   ).adapter('hermes_claude').capability('submission')
        self.assertEqual(cap.status, 'unknown', cap.reason)

    def test_a09_missing_malformed_and_deferred_are_unknown(self):
        malformed = {'tools': [{'name': 'mcp__delegate_task', 'input_schema': 'nope'}]}
        no_props = {'tools': [{'function': {'name': 'mcp__delegate_task', 'parameters': {}}}]}
        for label, req in (('missing', {'tools': []}), ('deferred', {'tools': [{'name': 'terminal'}],
                           'deferred_tools': ['mcp__delegate_task']}), ('malformed', malformed),
                           ('no_props', no_props)):
            runtime.clear_cache()
            cap = runtime.snapshot(req, {}).adapter('hermes_codex').capability('submission')
            self.assertEqual(cap.status, 'unknown', f'{label}: {cap.reason}')
        runtime.clear_cache()
        self.assertIn('absent or deferred',
                      runtime.snapshot({'tools': []}, {}).adapter('hermes_codex').capability('submission').reason)
        self.assertIn('missing or malformed',
                      runtime.snapshot(malformed, {}).adapter('hermes_codex').capability('submission').reason)

    def test_a09_prefixed_name_changes_fingerprint(self):
        runtime.clear_cache()
        a = runtime.snapshot({'tools': [self._shapes('mcp__delegate_task')['flat']]}, {})
        b = runtime.snapshot({'tools': []}, {})
        self.assertNotEqual(a.fingerprint, b.fingerprint)

    def test_a09_runtime_diagnostic_recognizes_supported_prefixed_delegate_task(self):
        from tools.delegate_tool import DELEGATE_TASK_SCHEMA

        def snapshot(name):
            return runtime.snapshot({'tools': [{'name': name,
                'input_schema': DELEGATE_TASK_SCHEMA['parameters']}]}, {})
        plain = snapshot('delegate_task').adapter('hermes_codex').capability('submission')
        prefixed = snapshot('mcp__delegate_task').adapter('hermes_codex').capability('submission')
        self.assertEqual(plain.status, 'supported')
        self.assertEqual(prefixed.status, plain.status, f'prefixed diagnostic: {prefixed.reason}')

    def test_depth_one_refuses_nested_and_defaults_to_parent_direct(self):
        with host_delegation(depth=1):
            snap = self.snapshot()
            self.assertEqual(snap.max_spawn_depth.value, 1)
            choice = runtime.resolve_topology(snap, "nested_conductor")
            self.assertEqual(choice.status, "unsupported")
            self.assertIn("depth", choice.reason)
            self.assertEqual(choice.selected, "parent_direct")
            self.assertEqual(runtime.resolve_topology(snap).selected, "parent_direct")

    def test_depth_two_supports_checked_codex_conductor(self):
        with host_delegation(depth=2):
            choice = runtime.resolve_topology(self.snapshot(), "nested_conductor")
            self.assertEqual((choice.status, choice.selected), ("supported", "nested_conductor"))

    def test_orchestrator_kill_switch_is_not_depth_support(self):
        with host_delegation(depth=2, orchestrator_enabled=False):
            self.assertEqual(runtime.resolve_topology(self.snapshot(), "nested_conductor").status, "unsupported")

    def test_alias_alone_never_selects_external_conductor(self):
        with host_delegation(depth=2), patch.object(claude_delegation, "is_active", return_value=False):
            choice = runtime.resolve_topology(self.snapshot(), "nested_conductor", transport="hermes_claude")
            self.assertEqual(choice.status, "unsupported")
            self.assertEqual(choice.selected, "parent_direct")
            self.assertTrue(choice.reason)

    def test_normal_and_deferred_claude_visibility_are_distinct(self):
        with host_delegation(depth=2), patch.object(claude_delegation, "is_active", return_value=True):
            for tools, expected in ((DIRECT_CLAUDE_TOOLS, "supported"), (DEFERRED_CLAUDE_TOOLS, "unknown")):
                request = delegate_task_request()
                request["tools"].extend(self._shapes(n)['flat'] for n in tools if n != "delegate_task")
                snap = self.snapshot(request)
                self.assertEqual(snap.adapter("hermes_claude").capability("submission").status, expected)
                if expected == "unknown":
                    self.assertEqual(runtime.resolve_topology(snap, "nested_conductor", transport="hermes_claude").status, "unsupported")

    def test_schema_variants_under_both_parent_providers_and_depths(self):
        for wire in ("openai", "anthropic"):
            for depth in (1, 2):
                for schema in ("batch-only", "model-param"):
                    with self.subTest(wire=wire, depth=depth, schema=schema), host_delegation(depth=depth):
                        snap = self.snapshot(delegate_task_request(schema, wire=wire))
                        cap = snap.adapter("hermes_codex").capability("model_parameter")
                        self.assertEqual(cap.status, "supported" if schema == "model-param" else "unsupported")
                        self.assertEqual(snap.adapter("hermes_codex").capability("exact_model").status, "unknown")
                        self.assertEqual(runtime.resolve_topology(snap, "nested_conductor").status,
                                         "supported" if depth == 2 else "unsupported")

    def test_missing_or_false_seams_fail_closed(self):
        for name in ("delegate_task", "_resolve_child_toolsets", "_build_child_agent"):
            with self.subTest(name=name), host_delegation(depth=2), patch.object(self.host, name, False):
                snap = self.snapshot()
                self.assertEqual(runtime.resolve_topology(snap, "nested_conductor").status, "unsupported")

    def test_missing_host_does_not_import_it_or_invent_limits(self):
        with patch.dict(sys.modules, {"tools.delegate_tool": None, "tools.delegate_tool_config": None}):
            snap = self.snapshot()
            self.assertEqual(snap.max_spawn_depth.status, "unknown")
            self.assertEqual(snap.adapter("hermes_codex").capability("submission").status, "unsupported")

    def test_absent_and_malformed_tool_schema_are_explicit(self):
        for request in ({}, {"tools": [{"name": "delegate_task", "parameters": {"properties": []}}]}):
            with self.subTest(request=request):
                self.assertNotEqual(self.snapshot(request).adapter("hermes_codex").capability("submission").status, "supported")

    def test_real_installed_schema_and_control_seams(self):
        request = {"tools": [deepcopy(self.host.DELEGATE_TASK_SCHEMA)]}
        snap = self.snapshot(request)
        adapter = snap.adapter("hermes_codex")
        self.assertEqual(adapter.capability("model_parameter").status, "unsupported")
        self.assertEqual(adapter.capability("cancellation").status, "supported")
        self.assertIn("cooperative", adapter.capability("cancellation").reason)
        self.assertEqual(adapter.capability("async_delivery").status, "unknown")
        self.assertIn("S06", adapter.capability("async_delivery").reason)
        self.assertEqual(adapter.capability("workspace_isolation").status, "unknown")

    def test_matrix_unknown_identity_effort_permissions_and_cli(self):
        for wire in ("openai", "anthropic"):
            with self.subTest(parent=wire):
                snap = self.snapshot(delegate_task_request(wire=wire))
                for transport in ("hermes_codex", "hermes_claude", "claude_cli"):
                    adapter = snap.adapter(transport)
                    for name in ("exact_model", "identity_observability", "permissions", "workspace_isolation"):
                        self.assertEqual(adapter.capability(name).status, "unknown")
                    self.assertIn(adapter.capability("effort_application").status, ("unknown", "unsupported"))
                self.assertEqual(snap.adapter("claude_cli").capability("submission").status, "unknown")

    def test_mixed_batch_and_concurrency_not_slot_reservation(self):
        with host_delegation(depth=2, max_concurrent_children=1):
            snap = self.snapshot()
            self.assertEqual(snap.max_concurrent_children.value, 1)
            self.assertEqual(snap.adapter("hermes_claude").capability("mixed_target_batch").status, "unsupported")
            self.assertTrue(snap.admission_required)

    def test_codex_mixed_batch_is_never_supported_without_split_evidence(self):
        for schema, expected in (("model-param", "unknown"), ("batch-only", "unsupported")):
            with self.subTest(schema=schema):
                cap = self.snapshot(delegate_task_request(schema)).adapter("hermes_codex").capability("mixed_target_batch")
                self.assertEqual(cap.status, expected)
                self.assertTrue(cap.reason)
        cap = self.snapshot({}).adapter("hermes_codex").capability("mixed_target_batch")
        self.assertEqual(cap.status, "unknown")

    def test_fallback_ownership_needs_a_checked_seam_in_both_states(self):
        # present seams: no seam reports ownership, so unknown (never supported)
        for transport, active in (("hermes_codex", True), ("hermes_claude", True)):
            with patch.object(claude_delegation, "is_active", return_value=active):
                cap = self.snapshot().adapter(transport).capability("fallback_ownership")
                self.assertEqual(cap.status, "unknown", transport)
                self.assertTrue(cap.reason)
        # missing host: codex unsupported with reason
        with patch.dict(sys.modules, {"tools.delegate_tool": None, "tools.delegate_tool_config": None}):
            cap = self.snapshot().adapter("hermes_codex").capability("fallback_ownership")
            self.assertEqual((cap.status, bool(cap.reason)), ("unsupported", True))
        # false host seam: codex unsupported
        with patch.object(self.host, "_build_child_agent", False):
            self.assertEqual(self.snapshot().adapter("hermes_codex").capability("fallback_ownership").status, "unsupported")
        # inactive Claude delegation: unsupported, consistent with its submission
        with patch.object(claude_delegation, "is_active", return_value=False):
            adapter = self.snapshot().adapter("hermes_claude")
            self.assertEqual(adapter.capability("submission").status, "unsupported")
            self.assertEqual(adapter.capability("fallback_ownership").status, "unsupported")

    def test_claude_cancellation_and_async_are_unknown_without_a_claude_seam(self):
        with patch.object(claude_delegation, "is_active", return_value=True):
            request = delegate_task_request()
            request["tools"].append(self._shapes("delegate_claude")['flat'])
            adapter = self.snapshot(request).adapter("hermes_claude")
            self.assertEqual(adapter.capability("submission").status, "supported")
            for name in ("cancellation", "async_delivery"):
                cap = adapter.capability(name)
                self.assertEqual(cap.status, "unknown", name)
                self.assertIn("delegate_claude", cap.reason)

    def test_codex_cancellation_present_and_missing_seam(self):
        self.assertEqual(self.snapshot().adapter("hermes_codex").capability("cancellation").status, "supported")
        with patch.object(self.host, "interrupt_subagent", False):
            self.assertEqual(self.snapshot().adapter("hermes_codex").capability("cancellation").status, "unknown")

    def test_codex_async_delivery_needs_getter_and_stop_hook_both_present_and_missing(self):
        cap = self.snapshot().adapter("hermes_codex").capability("async_delivery")
        self.assertEqual(cap.status, "unknown")
        self.assertIn("S06", cap.reason)
        import tools.delegate_tool_config as host_cfg
        with patch.object(host_cfg, "_get_max_async_children", False):
            cap = self.snapshot().adapter("hermes_codex").capability("async_delivery")
            self.assertEqual(cap.status, "unsupported")
            self.assertIn("async child limit", cap.reason)
        with patch.object(router, "on_subagent_stop", None):
            cap = self.snapshot().adapter("hermes_codex").capability("async_delivery")
            self.assertEqual(cap.status, "unsupported")
            self.assertIn("on_subagent_stop", cap.reason)

    def _matrix(self, request=None, *, parallel=False):
        return runtime.compatibility_matrix(self.snapshot(request), parallel=parallel)

    def test_matrix_emits_every_section7_row_with_owner_and_reason(self):
        rows = self._matrix()
        self.assertEqual(len(runtime.MATRIX_CASES), 14)
        self.assertEqual(len(rows), 14 * 3)
        self.assertEqual({r["transport"] for r in rows}, set(runtime.TRANSPORTS))
        self.assertEqual(len({(r["case"], r["transport"]) for r in rows}), 42)
        for row in rows:
            self.assertIn(row["status"], ("supported", "unsupported", "unknown"))
            self.assertTrue(row["owner"].startswith("S"), row)
            if row["status"] != "supported":
                self.assertTrue(row["reason"], row)

    def test_matrix_exact_statuses_for_rows_s01_cannot_evaluate(self):
        table = {r["case"] + "/" + r["transport"]: (r["status"], r["owner"]) for r in self._matrix()}
        for transport in runtime.TRANSPORTS:
            for case, owner in (("usage_admission", "S03"), ("quota_vs_pacing", "S03"),
                                ("restart_duplicate_hook", "S06"), ("parent_stop_amend", "S09"),
                                ("verification_substitution", "S14")):
                self.assertEqual(table[case + "/" + transport], ("unknown", owner))
            self.assertEqual(table["exact_model/" + transport], ("unknown", "S02"))
            self.assertEqual(table["tool_context_permissions/" + transport], ("unknown", "S04"))

    def test_matrix_exact_statuses_for_evaluated_rows(self):
        with host_delegation(depth=2), patch.object(claude_delegation, "is_active", return_value=True):
            codex_parent = {c + "/" + t: s for c, t, s in
                            ((r["case"], r["transport"], r["status"]) for r in self._matrix(delegate_task_request("model-param")))}
            claude_request = delegate_task_request("model-param", wire="anthropic")
            claude_request["tools"].append(self._shapes("delegate_claude")['flat'])
            claude_parent = {c + "/" + t: s for c, t, s in
                             ((r["case"], r["transport"], r["status"]) for r in self._matrix(claude_request))}
        # parent that can reach delegate_claude vs one that cannot
        self.assertEqual(codex_parent["submission/hermes_claude"], "unknown")
        self.assertEqual(claude_parent["submission/hermes_claude"], "supported")
        self.assertEqual(codex_parent["submission/hermes_codex"], "supported")
        self.assertEqual(claude_parent["submission/claude_cli"], "unknown")
        for table in (codex_parent, claude_parent):
            self.assertEqual(table["mixed_target_batch/hermes_codex"], "unknown")
            self.assertEqual(table["mixed_target_batch/hermes_claude"], "unsupported")
            self.assertEqual(table["mixed_target_batch/claude_cli"], "unsupported")
            self.assertEqual(table["native_fallback/hermes_codex"], "unknown")
            self.assertEqual(table["native_fallback/hermes_claude"], "unknown")
            self.assertEqual(table["background_completion/hermes_codex"], "unknown")
            self.assertEqual(table["background_completion/hermes_claude"], "unknown")
            self.assertEqual(table["failure_cancel_timeout/hermes_codex"], "supported")
            self.assertEqual(table["failure_cancel_timeout/hermes_claude"], "unknown")
            self.assertEqual(table["failure_cancel_timeout/claude_cli"], "unknown")
        self.assertEqual(codex_parent["exact_effort/claude_cli"], "unknown")

    def test_matrix_unsupported_host_and_deferred_visibility_cells(self):
        with patch.dict(sys.modules, {"tools.delegate_tool": None, "tools.delegate_tool_config": None}):
            table = {(r["case"], r["transport"]): r["status"] for r in self._matrix()}
        self.assertEqual(table[("submission", "hermes_codex")], "unsupported")
        self.assertEqual(table[("native_fallback", "hermes_codex")], "unsupported")
        self.assertEqual(table[("failure_cancel_timeout", "hermes_codex")], "unknown")
        self.assertEqual(table[("background_completion", "hermes_codex")], "unsupported")
        with host_delegation(depth=2), patch.object(claude_delegation, "is_active", return_value=True):
            request = delegate_task_request()
            request["tools"].extend({"name": n, "parameters": {}} for n in DEFERRED_CLAUDE_TOOLS if n != "delegate_task")
            table = {(r["case"], r["transport"]): r["status"] for r in self._matrix(request)}
        self.assertEqual(table[("submission", "hermes_claude")], "unknown")

    def test_matrix_concurrent_calls_never_derives_status_from_batch_width(self):
        # C01: the host per-call batch width is its own observed fact; it is not a limit on
        # distinct legacy invocations or on the CLI, so the row stays unknown for every width.
        for limit in (1, 3):
            for parallel in (False, True):
                with self.subTest(limit=limit, parallel=parallel), \
                        host_delegation(depth=1, max_concurrent_children=limit):
                    rows = [r for r in self._matrix(parallel=parallel) if r["case"] == "concurrent_calls"]
                    self.assertEqual({r["transport"] for r in rows}, set(runtime.TRANSPORTS))
                    self.assertEqual({r["status"] for r in rows}, {"unknown"})
                    for r in rows:
                        self.assertEqual(r["owner"], "S05")
                        self.assertEqual(r["observed"], {"host_batch_width": limit})
                        self.assertIn("batch width", r["reason"])
                        self.assertIn("not a limit on distinct", r["reason"])
                        self.assertIn("host", r["reason"])
        with patch.dict(sys.modules, {"tools.delegate_tool": None, "tools.delegate_tool_config": None}):
            rows = [r for r in self._matrix(parallel=True) if r["case"] == "concurrent_calls"]
            self.assertEqual({r["status"] for r in rows}, {"unknown"})
            self.assertTrue(all(r["observed"] == {"host_batch_width": "unknown"} and r["reason"]
                                for r in rows))

    def test_matrix_other_rows_carry_no_observed_batch_width(self):
        rows = [r for r in self._matrix() if r["case"] != "concurrent_calls"]
        self.assertTrue(all("observed" not in r for r in rows))

    def test_fallback_ownership_not_reported_from_config(self):
        snap = self.snapshot()
        for transport in ("hermes_codex", "hermes_claude"):
            self.assertIsNone(snap.adapter(transport).capability("fallback_ownership").value)

    def test_fingerprint_changes_on_depth_schema_config_and_seam(self):
        with host_delegation(depth=1):
            first = self.snapshot()
            self.assertIs(self.snapshot(), first)
        with host_delegation(depth=2):
            self.assertNotEqual(self.snapshot().fingerprint, first.fingerprint)
        self.assertNotEqual(self.snapshot(delegate_task_request("model-param")).fingerprint, first.fingerprint)
        self.cfg["callable"]["sonnet5"] = False
        self.assertNotEqual(self.snapshot().fingerprint, first.fingerprint)
        with patch.object(self.host, "delegate_task", False):
            self.assertNotEqual(self.snapshot().fingerprint, first.fingerprint)

    def test_cache_ttl_and_bound(self):
        with patch.object(runtime.time, "monotonic", return_value=100):
            first = self.snapshot()
        with patch.object(runtime.time, "monotonic", return_value=100 + runtime.CACHE_TTL_SECONDS + 1):
            self.assertIsNot(first, self.snapshot())
        for i in range(runtime.CACHE_MAX_ENTRIES + 5):
            self.cfg["orchestration"]["conductor"] = "fixture-" + str(i)
            self.snapshot()
        self.assertLessEqual(runtime.cache_info()["size"], runtime.CACHE_MAX_ENTRIES)

    def test_snapshot_is_immutable_json_diagnostic_is_detached(self):
        snap = self.snapshot()
        with self.assertRaises(FrozenInstanceError):
            snap.admission_required = False
        diagnostic = runtime.diagnostic(snap)
        json.dumps(diagnostic)
        self.assertEqual(diagnostic["configured"]["orchestration_enabled"], True)
        self.assertEqual(diagnostic["topology"]["active"], "unknown")
        self.assertEqual(diagnostic["topology"]["selected"], "parent_direct")
        diagnostic["configured"]["orchestration_enabled"] = False
        self.assertTrue(runtime.diagnostic(snap)["configured"]["orchestration_enabled"])

    def test_discovery_has_no_dispatch_subprocess_network_or_writes(self):
        cfg, request = deepcopy(self.cfg), delegate_task_request()
        before = deepcopy(request)
        with patch.object(self.host, "delegate_task", side_effect=AssertionError("dispatch")), \
             patch("subprocess.Popen", side_effect=AssertionError("subprocess")), \
             patch("socket.socket", side_effect=AssertionError("network")), \
             patch.object(claude_delegation, "install_reasoning_bridge", side_effect=AssertionError("mutation")):
            snap = self.snapshot(request)
            self.assertEqual(snap.schema_version, 1)
        self.assertEqual(self.cfg, cfg)
        self.assertEqual(request, before)

    def test_concurrent_snapshot_reads_do_not_leak_config(self):
        before = deepcopy(self.cfg)
        with ThreadPoolExecutor(max_workers=8) as pool:
            snapshots = list(pool.map(lambda _: self.snapshot(), range(32)))
        self.assertTrue(all(s.fingerprint == snapshots[0].fingerprint for s in snapshots))
        self.assertEqual(self.cfg, before)

    def test_invalid_topology_and_transport_refused(self):
        snap = self.snapshot()
        for topology, transport in (("invented", "hermes_codex"), ("nested_conductor", "invented")):
            choice = runtime.resolve_topology(snap, topology, transport=transport)
            self.assertEqual(choice.status, "unsupported")
            self.assertTrue(choice.reason)

    def test_checked_effort_seam_moved_is_explicit(self):
        with patch.object(self.host, "_resolve_child_runtime", False):
            cap = self.snapshot().adapter("hermes_claude").capability("effort_application")
            self.assertEqual(cap.status, "unsupported")
            self.assertTrue(cap.reason)

    def test_read_only_router_diagnostic_entrypoint(self):
        with host_delegation(depth=1):
            data = router.runtime_diagnostic(delegate_task_request(), self.cfg)
            self.assertEqual(data["topology"]["selected"], "parent_direct")
            self.assertEqual(data["schema_version"], 1)
