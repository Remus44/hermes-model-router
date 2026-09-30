"""S02: resolved target identity and configuration drift diagnostic (I03, I04).

Fixtures are sanitized and reproduce the audited Sonnet disagreement: router
tiers, the CLI alias map and the operator's host config name different Sonnet
models. No test reads the real ~/.hermes/config.yaml or opens a network socket.
"""
import dataclasses
import json
import socket
import subprocess
import tempfile
import time
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import model_router as router
from model_router import claude_opus_bridge as bridge
from model_router import execution_contracts as contracts
from model_router import runtime_capabilities as runtime
from model_router import target_identity as identity

ROUTER_CFG = {
    "provider": "openai-codex",
    "models": {"terra": "gpt-5.6-terra", "sol": "gpt-6.1-sol"},
    "tier_providers": {"terra": "openai-codex", "opus5": "anthropic", "sonnet5": "anthropic",
                       "haiku": "anthropic"},
    "callable": {"sonnet5": True, "opus5": True},
    "claude_delegation": {"tiers": {"haiku": "claude-haiku-4-5-20251001",
                                    "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}},
}
HOST_CFG = {
    "model": {"default": "claude-opus-5-5", "provider": "anthropic"},
    "fallback_providers": [{"provider": "anthropic", "model": "claude-sonnet-5"}],
    "delegation": {
        "model": "gpt-5.6-terra",
        "fallback_providers": [{"provider": "anthropic", "model": "claude-sonnet-5"}],
        "targets": {
            "opus5": {"provider": "anthropic", "model": "claude-opus-5-5"},
            "sonnet5": {"provider": "anthropic", "model": "claude-sonnet-5"},
            "terra": {"provider": "openai-codex", "model": "gpt-5.6-terra"},
        },
    },
}
OBSERVED = {"sonnet": [("claude-sonnet-5", "audit:cli-review-1"),
                       ("claude-sonnet-5", "audit:cli-review-2"),
                       ("claude-sonnet-5", "audit:cli-review-3")]}
HOST_YAML = """model:
  default: claude-opus-5-5
delegation:
  fallback_providers:
  - provider: anthropic
    model: claude-sonnet-5
  targets:
    opus5:
      provider: anthropic
      model: claude-opus-5-5
    sonnet5:
      provider: anthropic
      model: claude-sonnet-5
"""


def cli_exact_snapshot(status):
    """A real snapshot with the CLI exact_model capability replaced by ``status``."""
    snap = runtime.snapshot(None, ROUTER_CFG)
    adapters = []
    for adapter in snap.adapters:
        if adapter.transport == "claude_cli":
            caps = tuple(runtime.Capability("exact_model", status, reason="fixture evidence")
                         if c.name == "exact_model" else c for c in adapter.capabilities)
            adapter = runtime.AdapterCapabilities("claude_cli", caps)
        adapters.append(adapter)
    return dataclasses.replace(snap, adapters=tuple(adapters),
                               fingerprint=snap.fingerprint + ":" + status)


class Base(unittest.TestCase):
    def setUp(self):
        identity.clear_cache()
        runtime.clear_cache()
        self.addCleanup(identity.clear_cache)
        self.addCleanup(runtime.clear_cache)
        self.cfg = deepcopy(ROUTER_CFG)
        self.host = deepcopy(HOST_CFG)

    def resolve(self, alias, transport, **kw):
        kw.setdefault("cfg", self.cfg)
        kw.setdefault("host_cfg", self.host)
        return identity.resolve_target(alias, transport=transport, **kw)


class IdentityRecordTests(Base):
    def test_identity_is_json_compatible_with_explicit_unknown_fields(self):
        res = self.resolve("opus5", "hermes_claude", effort="high")
        data = res.identity.as_dict()
        json.dumps(data)
        self.assertEqual(data["transport"], "hermes_claude")
        self.assertEqual(data["alias"], "opus5")
        self.assertEqual(data["provider"], "anthropic")
        self.assertEqual(data["selection_mode"], "profile_preferred")
        self.assertEqual(data["requested"]["value"], "claude-opus-5-5")
        self.assertEqual(data["requested"]["source"], "router_config:claude_delegation.tiers.opus")
        self.assertEqual(data["observed"], {"value": "unknown", "source": "not_observed",
                                            "canonical": True})
        self.assertEqual(data["effort"]["requested"], "high")
        self.assertEqual(data["effort"]["applied"], "unknown")

    def test_no_effort_requested_is_not_applicable_not_unknown(self):
        data = self.resolve("opus5", "hermes_claude").identity.as_dict()
        self.assertEqual(data["effort"]["requested"], "not_applicable")
        self.assertEqual(data["effort"]["applied"], "not_applicable")

    def test_invalid_transport_mode_and_frozen_record(self):
        with self.assertRaises(ValueError):
            self.resolve("opus5", "carrier_pigeon")
        with self.assertRaises(ValueError):
            self.resolve("opus5", "hermes_claude", selection_mode="whatever")
        ident = self.resolve("opus5", "hermes_claude").identity
        with self.assertRaises(dataclasses.FrozenInstanceError):
            ident.alias = "x"

    def test_unknown_alias_is_unsupported_not_guessed(self):
        res = self.resolve("nonesuch", "hermes_codex")
        self.assertEqual(res.status, "unsupported")
        self.assertEqual(res.failure_class, "capability")
        self.assertEqual(res.identity.resolved.value, "unknown")


class ResolutionTests(Base):
    def test_codex_route_matching_config_resolves(self):
        res = self.resolve("terra", "hermes_codex", selection_mode="exact")
        self.assertEqual(res.status, "resolved")
        self.assertIsNone(res.mismatch)
        self.assertEqual(res.identity.resolved.value, "gpt-5.6-terra")
        self.assertEqual(res.identity.provider, "openai-codex")

    def test_exact_sonnet_disagreement_is_a_typed_mismatch_naming_owners(self):
        res = self.resolve("sonnet5", "hermes_codex", selection_mode="exact")
        self.assertEqual(res.status, "exact_mismatch")
        self.assertEqual(res.failure_class, "exact-route-mismatch")
        mismatch = res.mismatch
        self.assertIsInstance(mismatch, contracts.ExactRouteMismatch)
        self.assertEqual(mismatch.failure_class, "exact-route-mismatch")
        self.assertEqual(mismatch.stage, "resolved")
        self.assertEqual((mismatch.requested, mismatch.actual), ("claude-sonnet-5-5", "claude-sonnet-5"))
        self.assertIn("host_config:delegation.targets.sonnet5.model", mismatch.actual_source)
        self.assertIn("claude_delegation.tiers.sonnet", mismatch.requested_source)
        json.dumps(mismatch.as_dict())

    def test_preferred_substitution_is_recorded_not_silent(self):
        res = self.resolve("sonnet5", "hermes_codex", selection_mode="profile_preferred")
        self.assertEqual(res.status, "resolved")
        self.assertIsNone(res.mismatch)
        self.assertEqual(res.substitution["requested"], "claude-sonnet-5-5")
        self.assertEqual(res.substitution["resolved"], "claude-sonnet-5")
        self.assertEqual(res.substitution["policy"], "profile_preferred")

    def test_exact_explicit_model_different_from_route_is_refused(self):
        res = self.resolve("opus5", "hermes_codex", selection_mode="exact",
                           requested_model="claude-sonnet-5-5")
        self.assertEqual(res.status, "exact_mismatch")
        self.assertEqual(res.identity.requested.source, "operator_request")

    def test_exact_explicit_model_matching_host_target_resolves(self):
        res = self.resolve("sonnet5", "hermes_codex", selection_mode="exact",
                           requested_model="claude-sonnet-5")
        self.assertEqual(res.status, "resolved")

    def test_hermes_claude_uses_tier_model_and_agrees_with_config(self):
        res = self.resolve("sonnet5", "hermes_claude", selection_mode="exact")
        self.assertEqual(res.status, "resolved")
        self.assertEqual(res.identity.resolved.value, "claude-sonnet-5-5")

    def test_observed_identity_mismatch_is_typed_for_exact_only(self):
        exact = self.resolve("opus5", "hermes_claude", selection_mode="exact").identity
        ident, mismatch, substitution = identity.verify_observed(
            exact, "claude-sonnet-5", "cli_payload:modelUsage")
        self.assertEqual(ident.observed.value, "claude-sonnet-5")
        self.assertEqual(ident.observed.source, "cli_payload:modelUsage")
        self.assertEqual(mismatch.stage, "observed")
        self.assertEqual(mismatch.failure_class, "exact-route-mismatch")
        self.assertIsNone(substitution)
        ok, mismatch, _ = identity.verify_observed(exact, "claude-opus-5-5", "x")
        self.assertIsNone(mismatch)
        self.assertEqual(ok.observed.value, "claude-opus-5-5")
        preferred = self.resolve("opus5", "hermes_claude").identity
        _, mismatch, substitution = identity.verify_observed(preferred, "claude-sonnet-5", "x")
        self.assertIsNone(mismatch)
        self.assertEqual(substitution["observed"], "claude-sonnet-5")

    def test_cli_resolves_alias_argument_not_canonical_and_exact_is_unsupported(self):
        res = self.resolve("sonnet", "claude_cli", selection_mode="exact")
        self.assertEqual(res.status, "unsupported")
        self.assertEqual(res.failure_class, "capability")
        self.assertEqual(res.identity.requested.value, "claude-sonnet-5-5")
        self.assertEqual(res.identity.requested.source, "cli_alias_map:claude_opus_bridge.CLAUDE_REVIEW_MODELS.sonnet")
        self.assertEqual(res.identity.resolved.value, "sonnet")
        self.assertFalse(res.identity.resolved.canonical)
        self.assertIn("exact_model", " ".join(res.reasons))

    def test_cli_exact_needs_capability_evidence(self):
        for status, expected in (("unknown", "unsupported"), ("unsupported", "unsupported"),
                                 ("supported", "resolved")):
            identity.clear_cache()
            res = self.resolve("sonnet", "claude_cli", selection_mode="exact",
                               snapshot=cli_exact_snapshot(status))
            self.assertEqual(res.status, expected, status)
        self.assertEqual(res.identity.resolved.value, "claude-sonnet-5-5")
        self.assertTrue(res.identity.resolved.canonical)

    def test_cli_preferred_is_resolved_with_alias_unverified_note(self):
        res = self.resolve("opus", "claude_cli")
        self.assertEqual(res.status, "resolved")
        self.assertEqual(res.identity.resolved.value, "opus")
        self.assertTrue(any("not verified" in r for r in res.reasons))

    def test_default_snapshot_is_used_for_cli_exact_when_none_given(self):
        res = self.resolve("opus", "claude_cli", selection_mode="exact")
        self.assertEqual(res.status, "unsupported")


class CacheTests(Base):
    def count_computes(self):
        real = identity._compute
        calls = []

        def spy(*a, **k):
            calls.append(1)
            return real(*a, **k)
        return calls, patch.object(identity, "_compute", spy)

    def test_repeat_resolution_is_cached(self):
        calls, p = self.count_computes()
        with p:
            first = self.resolve("opus5", "hermes_claude")
            second = self.resolve("opus5", "hermes_claude")
        self.assertEqual(len(calls), 1)
        self.assertEqual(first, second)

    def test_changed_host_alias_invalidates(self):
        calls, p = self.count_computes()
        with p:
            before = self.resolve("sonnet5", "hermes_codex")
            self.host["delegation"]["targets"]["sonnet5"]["model"] = "claude-sonnet-5-5"
            after = self.resolve("sonnet5", "hermes_codex")
        self.assertEqual(len(calls), 2)
        self.assertEqual(before.identity.resolved.value, "claude-sonnet-5")
        self.assertEqual(after.identity.resolved.value, "claude-sonnet-5-5")
        self.assertIsNone(after.substitution)

    def test_changed_claude_tier_and_cli_alias_map_invalidate(self):
        calls, p = self.count_computes()
        with p:
            self.resolve("sonnet", "claude_cli")
            self.cfg["claude_delegation"]["tiers"]["sonnet"] = "claude-sonnet-5"
            self.resolve("sonnet", "claude_cli")
            with patch.dict(bridge.CLAUDE_REVIEW_MODELS, {"sonnet": "claude-sonnet-x"}):
                res = self.resolve("sonnet", "claude_cli")
        self.assertEqual(len(calls), 3)
        self.assertEqual(res.identity.requested.value, "claude-sonnet-x")

    def test_changed_capability_evidence_invalidates(self):
        calls, p = self.count_computes()
        with p:
            a = self.resolve("sonnet", "claude_cli", selection_mode="exact",
                             snapshot=cli_exact_snapshot("unknown"))
            b = self.resolve("sonnet", "claude_cli", selection_mode="exact",
                             snapshot=cli_exact_snapshot("supported"))
        self.assertEqual((a.status, b.status), ("unsupported", "resolved"))
        self.assertEqual(len(calls), 2)

    def test_ttl_expiry_recomputes(self):
        calls, p = self.count_computes()
        with p, patch.object(identity.time, "monotonic") as clock:
            clock.return_value = 1000.0
            self.resolve("opus5", "hermes_claude")
            clock.return_value = 1000.0 + identity.CACHE_TTL_SECONDS - 1
            self.resolve("opus5", "hermes_claude")
            clock.return_value = 1000.0 + identity.CACHE_TTL_SECONDS + 1
            self.resolve("opus5", "hermes_claude")
        self.assertEqual(len(calls), 2)

    def test_cache_is_bounded(self):
        for n in range(identity.CACHE_MAX_ENTRIES + 10):
            self.cfg["models"]["terra"] = f"gpt-{n}"
            self.resolve("terra", "hermes_codex")
        self.assertLessEqual(identity.cache_info()["entries"], identity.CACHE_MAX_ENTRIES)

    def test_cached_result_is_immutable_and_lock_exists(self):
        res = self.resolve("opus5", "hermes_claude")
        with self.assertRaises(dataclasses.FrozenInstanceError):
            res.status = "x"
        self.assertTrue(hasattr(identity._LOCK, "acquire"))


class DriftDiagnosticTests(Base):
    def diagnose(self, **kw):
        kw.setdefault("cfg", self.cfg)
        kw.setdefault("host_cfg", self.host)
        kw.setdefault("observed", OBSERVED)
        return identity.drift_diagnostic(**kw)

    def tier(self, diag, name):
        return next(t for t in diag["tiers"] if t["tier"] == name)

    def test_baseline_sonnet_disagreement_is_reproduced_with_owners(self):
        diag = self.diagnose()
        sonnet = self.tier(diag, "sonnet")
        self.assertFalse(sonnet["agree"])
        owners = {(v["owner"], v["key"]): v["value"] for v in sonnet["values"]}
        self.assertEqual(owners[("router_config", "claude_delegation.tiers.sonnet")], "claude-sonnet-5-5")
        self.assertEqual(owners[("cli_alias_map", "claude_opus_bridge.CLAUDE_REVIEW_MODELS.sonnet")],
                         "claude-sonnet-5-5")
        self.assertEqual(owners[("host_config", "delegation.targets.sonnet5.model")], "claude-sonnet-5")
        self.assertEqual(owners[("host_config", "delegation.fallback_providers[0].model")],
                         "claude-sonnet-5")
        self.assertEqual(owners[("host_config", "fallback_providers[0].model")], "claude-sonnet-5")
        kinds = {f["kind"] for f in diag["findings"] if f["tier"] == "sonnet"}
        self.assertEqual(kinds, {"config_drift", "observed_mismatch"})
        drift = next(f for f in diag["findings"] if f["kind"] == "config_drift")
        self.assertEqual({d["owner"] for d in drift["disagreeing"]}, {"host_config"})
        self.assertEqual(drift["expected"]["owner"], "router_config")
        obs = next(f for f in diag["findings"] if f["kind"] == "observed_mismatch")
        self.assertEqual(obs["observed"][0]["source"], "audit:cli-review-1")
        self.assertEqual(len(obs["observed"]), 3)

    def test_opus_agrees_and_parent_default_is_attributed(self):
        opus = self.tier(self.diagnose(), "opus")
        self.assertTrue(opus["agree"])
        self.assertIn(("host_config", "model.default"), {(v["owner"], v["key"]) for v in opus["values"]})

    def test_fixed_host_config_clears_the_drift(self):
        self.host["delegation"]["targets"]["sonnet5"]["model"] = "claude-sonnet-5-5"
        self.host["delegation"]["fallback_providers"][0]["model"] = "claude-sonnet-5-5"
        self.host["fallback_providers"][0]["model"] = "claude-sonnet-5-5"
        diag = self.diagnose(observed=None)
        self.assertTrue(self.tier(diag, "sonnet")["agree"])
        self.assertEqual(diag["findings"], [])

    def test_default_owner_is_named_when_router_config_omits_tiers(self):
        self.cfg["claude_delegation"] = {}
        sonnet = self.tier(self.diagnose(), "sonnet")
        owners = {(v["owner"], v["key"]) for v in sonnet["values"]}
        self.assertIn(("claude_delegation.DEFAULTS", "tiers.sonnet"), owners)

    def test_missing_host_config_is_reported_not_raised(self):
        diag = self.diagnose(host_cfg={}, observed=None)
        self.assertFalse(diag["host_config"]["available"])
        self.assertTrue(self.tier(diag, "sonnet")["agree"])

    def test_cli_exact_status_is_reported_from_capability_evidence(self):
        diag = self.diagnose()
        self.assertEqual(diag["cli_exact_model"]["status"], "unknown")
        diag = self.diagnose(snapshot=cli_exact_snapshot("supported"))
        self.assertEqual(diag["cli_exact_model"]["status"], "supported")

    def test_diagnostic_needs_no_network_or_subprocess(self):
        def boom(*a, **k):
            raise AssertionError("network/subprocess used")
        with patch.object(socket.socket, "connect", boom), \
                patch.object(socket, "create_connection", boom), \
                patch.object(subprocess, "run", boom), patch.object(subprocess, "Popen", boom):
            diag = self.diagnose()
        self.assertFalse(diag["network_used"])
        json.dumps(diag)

    def test_diagnostic_does_not_mutate_inputs(self):
        cfg, host = deepcopy(self.cfg), deepcopy(self.host)
        self.diagnose()
        self.assertEqual((cfg, host), (self.cfg, self.host))


class RouterEntrypointAndMigrationTests(Base):
    def test_router_identity_diagnostic_reads_host_config_read_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(HOST_YAML, encoding="utf-8")
            before = path.read_bytes()
            mtime = path.stat().st_mtime_ns
            with patch.object(router, "_HERMES_CONFIG_PATH", path):
                diag = router.identity_diagnostic(cfg=self.cfg, observed=OBSERVED)
            self.assertEqual(path.read_bytes(), before)
            self.assertEqual(path.stat().st_mtime_ns, mtime)
        sonnet = next(t for t in diag["tiers"] if t["tier"] == "sonnet")
        self.assertFalse(sonnet["agree"])
        self.assertTrue(diag["host_config"]["available"])

    def test_router_identity_diagnostic_survives_missing_host_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            with patch.object(router, "_HERMES_CONFIG_PATH", Path(tmp) / "absent.yaml"):
                diag = router.identity_diagnostic(cfg=self.cfg)
        self.assertFalse(diag["host_config"]["available"])

    def test_migration_is_a_diff_text_and_applies_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.yaml"
            path.write_text(HOST_YAML, encoding="utf-8")
            text = path.read_text(encoding="utf-8")
            diff = identity.propose_host_migration(text, self.cfg)
            self.assertEqual(path.read_text(encoding="utf-8"), HOST_YAML)
        self.assertIn("--- config.yaml", diff)
        self.assertIn("-      model: claude-sonnet-5\n", diff)
        self.assertIn("+      model: claude-sonnet-5-5\n", diff)
        self.assertIn("-    model: claude-sonnet-5\n", diff)
        self.assertNotIn("claude-opus-5-5\n+", diff)

    def test_migration_is_empty_when_nothing_drifts(self):
        self.assertEqual(identity.propose_host_migration(HOST_YAML.replace(
            "claude-sonnet-5\n", "claude-sonnet-5-5\n"), self.cfg), "")


class ContractHelpersTests(unittest.TestCase):
    def test_check_exact_ignores_unobserved_and_noncanonical_resolution(self):
        ident = contracts.TargetIdentity(
            provider="anthropic", account="unknown", transport="claude_cli", alias="sonnet",
            selection_mode="exact",
            requested=contracts.ModelFact("claude-sonnet-5-5", "x"),
            resolved=contracts.ModelFact("sonnet", "cli_alias_argument", canonical=False),
            observed=contracts.ModelFact(),
            effort=contracts.EffortFact())
        self.assertIsNone(contracts.check_exact(ident))
        seen = dataclasses.replace(ident, observed=contracts.ModelFact("claude-sonnet-5", "payload"))
        self.assertEqual(contracts.check_exact(seen).stage, "observed")

    def test_invalid_identity_fields_rejected(self):
        with self.assertRaises(ValueError):
            contracts.TargetIdentity(provider="p", account="a", transport="nope", alias="x",
                                     selection_mode="exact")
        with self.assertRaises(ValueError):
            contracts.TargetIdentity(provider="p", account="a", transport="claude_cli", alias="x",
                                     selection_mode="maybe")


if __name__ == "__main__":
    unittest.main()
