"""R12-6 round-2 regression and installed-chain evidence."""
import contextvars
import json
import tempfile
import threading
import time
import unittest
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import model_tools
import model_router as router
from hermes_cli import plugins
from hermes_cli.plugins import PluginContext, PluginManifest, PluginManager
from model_router import claude_delegation as claude
from model_router import execution_adapters as ea, triage, usage_guard
from model_router import test_r12_6_fix1 as fix1


def _cfg(path):
    return {
        "enabled": True,
        "models": {"terra": "gpt-5.6-terra", "sol": "gpt-6.1-sol"},
        "tier_providers": {"terra": "openai-codex", "sol": "openai-codex"},
        "callable": {"terra": True, "sol": True, "haiku": True, "sonnet5": True, "opus5": True},
        "default_model": "terra",
        "triage": {"enabled": True, "max_workers_per_turn": 4, "max_children_per_worker": 1,
                   "path": str(Path(path) / "triage.jsonl")},
        "orchestration": {"enabled": False}, "shadow": {"enabled": False},
    }


class R126Fix2ControllerRed(unittest.TestCase):
    def setUp(self):
        self.fixture = fix1.R126Fix1Tests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.args = {"goal": "same", "tier": "haiku"}
        self.turn = "r12-fix1:turn"
        self.root = self.fixture.root

    def _hooks(self, *, maximum=2):
        return patch.object(router, "_load_config", return_value=self.fixture.config), \
            patch.object(router, "_delegation_targets_detail", return_value={}), \
            patch.object(triage, "_caller", return_value=self.root), \
            patch.object(ea, "RESERVATIONS", ea.ReservationBook()), \
            patch.object(ea, "_MAX_RECORDS", maximum)

    def test_i3_host_later_refusal_must_retire_owner_before_fresh_retry(self):
        # Portable on BASE: deliver the refusal through the installed lifecycle,
        # not a newly added helper that BASE cannot import.
        chain = RegisteredClaudeChainTests()
        chain.setUp()
        self.addCleanup(chain.doCleanups)
        blocker = chain.ctx.register_hook("pre_tool_call", lambda **kw:
            {"action": "block", "message": "host later refusal"})
        self.assertIn("error", json.loads(chain._deliver()))
        before = triage._TURN_WORKERS["r12:live"]
        blocker.dispose()
        chain._deliver()
        self.assertEqual(before + 1, triage._TURN_WORKERS["r12:live"],
                         "I3: later host refusal left a reusable unbound authorization")
        self.assertEqual(len(chain.raw_calls), 1)

    def test_i3_retired_pending_owner_cannot_fall_through_to_new_raw_claim(self):
        raw = []
        patches = self._hooks(maximum=2)
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            self.assertIsNone(self.fixture.hook("same", self.args))
            pending_context = contextvars.copy_context()
            for number in range(5):
                self.assertIsNone(self.fixture.hook(
                    f"other-{number}", {"goal": str(number), "tier": "haiku"},
                    turn_id=f"r12-fix1:other-{number}"))
            stable = ea.stable_legacy_invocation("delegate_claude", self.args,
                session_id="r12-fix1-root", turn_id=self.turn, tool_call_id="same")
            self.assertNotIn(stable.key, triage._RETAINED_ADMISSIONS,
                             "test setup did not retire the pending admission")
            response = pending_context.run(
                self.fixture.guard, "same", self.args,
                lambda _: raw.append("retired-token-launch") or '{"delegation_id":"new"}')
        self.assertIn("error", json.loads(response))
        self.assertEqual(raw, [], "I3: retired pending token launched a new raw execution")
        self.assertEqual(triage._TURN_WORKERS[self.turn], 1)

    def test_normal_outer_owner_binds_inner_hook_before_generation_eviction(self):
        # Supplemental direct boundary control for the executor's outer-F05,
        # inner-pre-hook ordering; not the registered Claude I5 evidence.
        raw = []
        patches = self._hooks(maximum=1)
        def downstream(sent):
            self.assertIsNone(self.fixture.hook("same", sent))
            raw.append("same")
            return '{"delegation_id":"normal"}'
        with patches[0], patches[1], patches[2], patches[3], patches[4]:
            first = self.fixture.guard("same", self.args, downstream)
            charges = triage._TURN_WORKERS[self.turn]
            self.assertEqual(self.fixture.guard("same", self.args, downstream), first)
            self.assertEqual(triage._TURN_WORKERS[self.turn], charges)
            # Uncorrelated legacy F05 execution evicts the finished outer G1.
            contextvars.Context().run(self.fixture.guard, "other", {"goal": "other", "tier": "haiku"},
                lambda _: '{"delegation_id":"other"}')
            self.fixture.guard("same", self.args, downstream)
            self.assertEqual(triage._TURN_WORKERS[self.turn], charges + 1,
                             "normal outer G2 reused an unbound inner-hook admission")
            self.assertEqual(raw, ["same", "same"])

    def test_late_pre_hook_cannot_admit_after_its_host_lifecycle_closed(self):
        # Supplemental closure-boundary simulation: use the actual installed
        # post-hook delivery with the context captured by request middleware.
        # This is not a provider timeout or the mandatory I5 refusal proof.
        chain = RegisteredClaudeChainTests()
        chain.setUp()
        self.addCleanup(chain.doCleanups)
        captured = {}
        chain.ctx.register_middleware("tool_request", lambda **kw:
            captured.update(context=contextvars.copy_context()))
        entered, resume = threading.Event(), threading.Event()
        original_spawn = triage.spawn_block
        def delayed_spawn(*args, **kwargs):
            entered.set()
            self.assertTrue(resume.wait(10))
            return original_spawn(*args, **kwargs)
        with patch.object(triage, "spawn_block", delayed_spawn), ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(chain._deliver)
            try:
                self.assertTrue(entered.wait(10))
                captured["context"].run(plugins.invoke_hook, "post_tool_call",
                    tool_name="delegate_claude", args={"goal": "registered", "tier": "haiku"},
                    session_id="r12-live", turn_id="r12:live", tool_call_id="same", status="blocked")
                resume.set()
                self.assertIn("error", json.loads(future.result(10)), "late pre-hook resurrected a closed host call")
                self.assertEqual(len(chain.raw_calls), 0)
                self.assertEqual(len(triage._RETAINED_ADMISSIONS), 0)
                self.assertEqual(triage._TURN_WORKERS.get("r12:live", 0), 0)
            finally:
                resume.set()

    def test_i2_late_old_binding_cannot_overwrite_newer_generation_or_recharge_replay(self):
        raw = []
        started, finish_g1 = threading.Event(), threading.Event()
        old_bind_entered, release_old_bind = threading.Event(), threading.Event()
        original_begin = ea.ReservationBook._begin

        def paused_begin(book, *args, **kwargs):
            outcome = original_begin(book, *args, **kwargs)
            if threading.current_thread().name.startswith("duplicate"):
                self.assertFalse(outcome[0], "duplicate did not capture old G1")
                old_bind_entered.set()
                self.assertTrue(release_old_bind.wait(10), "old binding release timed out")
            return outcome

        def g1_raw(_):
            raw.append("G1")
            started.set()
            self.assertTrue(finish_g1.wait(10), "G1 release timed out")
            return '{"delegation_id":"G1"}'

        def delivery(raw_next):
            self.assertIsNone(self.fixture.hook("same", self.args))
            return self.fixture.guard("same", self.args, raw_next)

        patches = self._hooks(maximum=1)
        try:
            with patches[0], patches[1], patches[2], patches[3], patches[4], \
                 patch.object(ea.ReservationBook, "_begin", paused_begin), \
                 ThreadPoolExecutor(max_workers=1, thread_name_prefix="g1") as first, \
                 ThreadPoolExecutor(max_workers=1, thread_name_prefix="duplicate") as duplicate:
                g1 = first.submit(delivery, g1_raw)
                self.assertTrue(started.wait(10), "G1 did not reach raw")
                delayed = duplicate.submit(delivery, lambda _: raw.append("unexpected-D") or "wrong")
                self.assertTrue(old_bind_entered.wait(10), "duplicate did not reach old bind")
                finish_g1.set()
                g1.result(10)
                self.assertIsNone(self.fixture.hook("other", {"goal": "other", "tier": "haiku"}))
                self.fixture.guard("other", {"goal": "other", "tier": "haiku"},
                                   lambda _: '{"delegation_id":"other"}')
                self.assertIsNone(self.fixture.hook("same", self.args))
                g2 = self.fixture.guard("same", self.args,
                                        lambda _: raw.append("G2") or '{"delegation_id":"G2"}')
                stable = ea.stable_legacy_invocation("delegate_claude", self.args,
                    session_id="r12-fix1-root", turn_id=self.turn, tool_call_id="same")
                generation = triage._RETAINED_ADMISSIONS[stable.key].generation
                release_old_bind.set()
                delayed.result(10)
                self.assertEqual(triage._RETAINED_ADMISSIONS[stable.key].generation, generation,
                                 "I2: stale old binding overwrote the newer generation")
                charges = triage._TURN_WORKERS[self.turn]
                self.assertIsNone(self.fixture.hook("same", self.args))
                replay = self.fixture.guard("same", self.args, lambda _: raw.append("G3") or "wrong")
            self.assertEqual(replay, g2)
            self.assertEqual(triage._TURN_WORKERS[self.turn], charges,
                             "I2: retained G2 replay charged after a stale old binding")
            self.assertEqual(raw, ["G1", "G2"])
        finally:
            finish_g1.set()
            release_old_bind.set()


class RegisteredClaudeChainTests(unittest.TestCase):
    """Real manager -> host guards -> registry -> middleware -> Claude handler."""
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.cfg = _cfg(self.tmp.name)
        self.root = SimpleNamespace(session_id="r12-live-root", _delegate_depth=0)
        self.manager = PluginManager()
        self.manager._discovered = True  # private manager, no discovery of user plugins
        self.manifest = PluginManifest(name="r12-live", source="test", key="r12-live")
        self.ctx = PluginContext(self.manifest, self.manager)
        self.before_active = claude._ACTIVE
        self.before_deadline = None
        try:
            from agent import tool_executor
            self.before_deadline = tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS
        except Exception:
            pass
        self.global_manager = getattr(plugins, "_plugin_manager", None)
        self.before_agent_tools = model_tools._AGENT_LOOP_TOOLS.copy()
        self.manager_patch = patch.object(plugins, "_plugin_manager", self.manager)
        self.manager_patch.start()
        self.addCleanup(self.manager_patch.stop)
        self.addCleanup(self._restore_registration)
        triage.reset_state()
        self.addCleanup(triage.reset_state)
        self.stack = ExitStack()
        self.addCleanup(self.stack.close)
        self.raw_calls, self.audits, self.seams = [], [], []
        self.raw_result = '{"delegation_id":"registered-child"}'
        self.raw_callback = lambda **kw: self.raw_result
        for item in (
            patch.object(router, "_load_config", return_value=self.cfg),
            patch.object(router, "_delegation_targets_detail", return_value={}),
            patch.object(triage, "_caller", return_value=self.root),
            patch.object(ea, "RESERVATIONS", ea.ReservationBook()),
            patch.object(claude, "_host", return_value=(self._raw, lambda: self.root)),
            patch.object(claude.usage_guard, "read", return_value=usage_guard.Reading(10.0, 0.0, None, None, time.time())),
            patch.object(claude, "_audit", side_effect=lambda *a, **kw: self.audits.append((a, kw))),
        ):
            self.stack.enter_context(item)
        self.before_tools = {name: model_tools.registry.get_entry(name) for name in ("delegate_claude", "triage_task")}
        from tools import registry as registry_module
        self.registry_module = registry_module
        self.before_caches = {name: getattr(registry_module, name).copy() for name in (
            "_check_fn_cache", "_check_fn_last_good", "_check_fn_ever_good", "_check_fn_core_drop_warned", "_NO_CACHE_CHECK_FNS")}
        self.ctx.register_hook("pre_tool_call", lambda **kw: self.seams.append(("pre", threading.get_ident())))
        self.ctx.register_middleware("tool_execution", self._execution_observer)
        router.register(self.ctx)
        self.ctx.register_hook("post_tool_call", lambda **kw: self.seams.append(("post", threading.get_ident())))
        self.assertIs(model_tools.registry.get_entry("delegate_claude").handler, claude.handle_delegate_claude)
        self.assertIn(ea.guard_legacy_tool_execution, self.manager._middleware["tool_execution"])

    def _raw(self, **kwargs):
        self.seams.append(("raw", threading.get_ident()))
        self.raw_calls.append(kwargs)
        self.assertFalse(triage._LOCK.locked(), "triage lock spans raw execution")
        self.assertFalse(ea.RESERVATIONS._lock.locked(), "F05 lock spans raw execution")
        return self.raw_callback(**kwargs)

    def _execution_observer(self, **kwargs):
        self.seams.append(("middleware", threading.get_ident()))
        return kwargs["next_call"](kwargs["args"])

    def _restore_registration(self):
        for registration in reversed(list(self.manager._registration_order)):
            registration.dispose()
        claude._ACTIVE = self.before_active
        self.assertEqual(model_tools._AGENT_LOOP_TOOLS, self.before_agent_tools)
        for name, previous in self.before_tools.items():
            self.assertIs(model_tools.registry.get_entry(name), previous, "registration disposal lost previous tool")
        for name, snapshot in self.before_caches.items():
            cache = getattr(self.registry_module, name)
            cache.clear()
            cache.update(snapshot)
        if self.before_deadline is not None:
            from agent import tool_executor
            tool_executor._SEQUENTIAL_DEADLINE_EXEMPT_TOOLS = self.before_deadline

    def _deliver(self, call_id="same", *, session="r12-live", turn="r12:live", args=None, tool="delegate_claude"):
        return model_tools.handle_function_call(
            tool, args if args is not None else {"goal": "registered", "tier": "haiku"},
            tool_call_id=call_id, session_id=session, turn_id=turn,
        )

    def _stable(self, call_id="same", *, turn="r12:live"):
        return ea.stable_legacy_invocation("delegate_claude", {"goal": "registered", "tier": "haiku"},
            session_id="r12-live", turn_id=turn, tool_call_id=call_id)

    def test_i5_five_sealed_deliveries_exhaustion_and_policy_changes(self):
        owner_thread = threading.get_ident()
        responses = [self._deliver() for _ in range(5)]
        self.assertEqual(responses, [responses[0]] * 5)
        self.assertEqual(json.loads(responses[0])["claude_tier"], "haiku")
        self.assertIn("usage", json.loads(responses[0]))
        self.assertEqual(len(self.raw_calls), 1)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 1)
        self.assertEqual(len(self.audits), 1)
        claim = ea.RESERVATIONS._claims[self._stable().key]
        self.assertTrue(claim.sealed and claim.cached and not claim.protected)
        self.assertEqual(claim.response, responses[0], "F05 did not cache complete Claude public bytes")
        self.assertEqual(claim.receipt.transport, "hermes_claude")
        self.assertEqual([name for name, _ in self.seams].count("middleware"), 5)
        self.assertEqual([name for name, _ in self.seams].count("pre"), 5)
        self.assertTrue(all(tid != owner_thread for name, tid in self.seams if name == "pre"),
                        "fixture did not run actual bounded callback contexts")
        self.assertTrue(all(tid == owner_thread for name, tid in self.seams if name in {"middleware", "raw"}))
        for call in ("b", "c", "d"):
            self.assertNotIn("error", json.loads(self._deliver(call)))
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 4)
        self.assertEqual(self._deliver(), responses[0])
        self.assertIn("Worker budget", json.loads(self._deliver("exhausted"))["error"])
        self.cfg["callable"]["haiku"] = False
        self.assertEqual(self._deliver(), responses[0])
        self.assertIn("switched off", json.loads(self._deliver("haiku-off"))["error"])
        self.cfg["callable"].update(sonnet5=False, opus5=False)
        self.assertEqual(self._deliver(), responses[0])
        self.assertIn("error", json.loads(self._deliver("all-off")))
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 4)
        self.assertEqual(len(self.raw_calls), 4)

    def test_i5_changed_payload_tool_and_authority_do_not_inherit_exemption(self):
        first = self._deliver()
        self.assertIn("identity changed", json.loads(self._deliver(args={"goal": "altered", "tier": "haiku"}))["error"])
        self.assertIn("agent loop", json.loads(self._deliver(tool="delegate_task"))["error"])
        # Authority corruption control at the mirror, not an invented host identity field.
        triage._RETAINED_ADMISSIONS[self._stable().key].scope = "public:other"
        self.assertIn("identity changed", json.loads(self._deliver())["error"])
        triage._RETAINED_ADMISSIONS[self._stable().key].scope = self._stable().scope
        self.cfg["callable"].update(haiku=False, sonnet5=False, opus5=False)
        self.assertIn("error", json.loads(self._deliver(session="other-session")))
        self.assertEqual(self._deliver(), first)
        self.assertEqual(len(self.raw_calls), 1)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 1)

    def test_i1_real_callback_replay_survives_eviction_after_pre_hook(self):
        entered, resume = threading.Event(), threading.Event()
        original_consume = triage.consume_retained_admission

        def paused_consume(invocation):
            if threading.current_thread().name.startswith("replay"):
                entered.set()
                self.assertTrue(resume.wait(10))
            return original_consume(invocation)

        with patch.object(ea, "_MAX_RECORDS", 1), patch.object(triage, "consume_retained_admission", paused_consume):
            first = self._deliver()
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="replay") as pool:
                future = pool.submit(self._deliver)
                try:
                    self.assertTrue(entered.wait(10))
                    self._deliver("other")
                    self.assertIsNone(ea.retained_invocation_generation(self._stable()))
                    self.raw_result = '{"delegation_id":"G2"}'
                    g2 = self._deliver()
                    self.assertNotEqual(g2, first)
                    generation = triage._RETAINED_ADMISSIONS[self._stable().key].generation
                finally:
                    resume.set()
                self.assertEqual(future.result(10), first,
                                 "I1: callback-local replay capture was lost across host context")
            self.assertEqual(triage._RETAINED_ADMISSIONS[self._stable().key].generation, generation)
            self.assertEqual(self._deliver(), g2)
            self.assertEqual(len(self.raw_calls), 3, "I1: evicted G1 exemption launched another generation")
            self.assertEqual(triage._TURN_WORKERS["r12:live"], 3)

    def test_i5_pending_before_claim_inflight_and_raw_finished_before_public_seal(self):
        entered, resume = threading.Event(), threading.Event()
        original_begin = ea.ReservationBook._begin

        def paused_begin(book, *args, **kwargs):
            if threading.current_thread().name.startswith("owner"):
                entered.set()
                self.assertTrue(resume.wait(10))
            return original_begin(book, *args, **kwargs)

        with patch.object(ea.ReservationBook, "_begin", paused_begin):
            with ThreadPoolExecutor(max_workers=1, thread_name_prefix="owner") as pool:
                pending = pool.submit(self._deliver)
                try:
                    self.assertTrue(entered.wait(10))
                    # A pre-F05 join may become the one owner, never a second charge.
                    first = self._deliver()
                    self.assertEqual(len(self.raw_calls), 1)
                finally:
                    resume.set()
                self.assertEqual(pending.result(10), first)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 1)
        triage.reset_state()
        ea.RESERVATIONS = ea.ReservationBook()
        self.raw_calls.clear()
        raw_entered, raw_release = threading.Event(), threading.Event()
        seal_entered, seal_release = threading.Event(), threading.Event()
        original_annotate = claude._annotate

        def raw(**kwargs):
            raw_entered.set()
            self.assertTrue(raw_release.wait(10))
            return self.raw_result

        def annotate(*args, **kwargs):
            seal_entered.set()
            self.assertTrue(seal_release.wait(10))
            return original_annotate(*args, **kwargs)

        self.raw_callback = raw
        with patch.object(claude, "_annotate", annotate):
            with ThreadPoolExecutor(max_workers=1) as pool:
                future = pool.submit(self._deliver)
                try:
                    self.assertTrue(raw_entered.wait(10))
                    with patch.object(ea, "_RECORD_TTL_SECONDS", 0), patch.object(ea, "_MAX_RECORDS", 1):
                        self.assertIn("already started", json.loads(self._deliver())["error"])
                    claim = ea.RESERVATIONS._claims[self._stable().key]
                    self.assertTrue(claim.in_flight and claim.protected)
                    raw_release.set()
                    self.assertTrue(seal_entered.wait(10))
                    self.assertFalse(claim.in_flight)
                    self.assertTrue(claim.protected and not claim.sealed)
                    with patch.object(ea, "_RECORD_TTL_SECONDS", 0), patch.object(ea, "_MAX_RECORDS", 1):
                        self.assertIn("already started", json.loads(self._deliver())["error"])
                    self.assertEqual(len(self.raw_calls), 1)
                    self.assertEqual(triage._TURN_WORKERS["r12:live"], 1)
                finally:
                    raw_release.set()
                    seal_release.set()
                first = future.result(10)
        self.assertEqual(self._deliver(), first)

    def test_i3_real_context_pending_loss_size_ttl_and_replaced_nonce_refuses_raw(self):
        for loss in ("size", "ttl", "nonce"):
            with self.subTest(loss=loss):
                triage.reset_state()
                ea.RESERVATIONS = ea.ReservationBook()
                self.raw_calls.clear()
                entered, resume, others_resume = threading.Event(), threading.Event(), threading.Event()
                others_entered = {str(i): threading.Event() for i in range(4)}
                original_consume = triage.consume_retained_admission

                def paused(invocation):
                    if invocation.key[3] == "same":
                        entered.set()
                        self.assertTrue(resume.wait(10))
                    else:
                        others_entered[invocation.key[3]].set()
                        self.assertTrue(others_resume.wait(10))
                    return original_consume(invocation)

                clock = [100.0]
                with patch.object(ea, "_MAX_RECORDS", 2), \
                     patch.object(triage, "_monotonic", side_effect=lambda: clock[0]), \
                     patch.object(triage, "consume_retained_admission", paused), \
                     ThreadPoolExecutor(max_workers=5) as pool:
                    future = pool.submit(self._deliver)
                    others = []
                    try:
                        self.assertTrue(entered.wait(10))
                        if loss == "size":
                            for number in range(4):
                                call = str(number)
                                others.append(pool.submit(self._deliver, call, turn=f"r12:other-{number}"))
                                self.assertTrue(others_entered[call].wait(10))
                            self.assertNotIn(self._stable().key, triage._RETAINED_ADMISSIONS,
                                             "size-pressure setup did not evict the pending owner")
                            self.assertLessEqual(len(triage._RETAINED_ADMISSIONS), 2)
                        elif loss == "ttl":
                            clock[0] += ea._RECORD_TTL_SECONDS + 1
                        else:
                            triage._RETAINED_ADMISSIONS[self._stable().key].nonce = "replacement"
                        before_raw = len(self.raw_calls)
                        resume.set()
                        result = future.result(10)
                        self.assertIn("error", json.loads(result), f"I3: {loss}-lost callback handoff launched raw")
                        self.assertEqual(len(self.raw_calls), before_raw, f"I3: {loss}-lost token borrowed authorization")
                        self.assertEqual(triage._TURN_WORKERS["r12:live"], 1)
                    finally:
                        resume.set()
                        others_resume.set()
                    for other in others:
                        other.result(10)

    def test_i3_real_join_refusal_cannot_retire_pending_owner(self):
        entered, resume = threading.Event(), threading.Event()
        original_consume = triage.consume_retained_admission

        def paused(invocation):
            if threading.current_thread().name.startswith("pending-owner"):
                entered.set()
                self.assertTrue(resume.wait(10))
            return original_consume(invocation)

        self.ctx.register_hook("pre_tool_call", lambda **kw:
            {"action": "block", "message": "join refused"} if threading.current_thread().name.startswith("hermes-hook")
            and join_phase[0] else None)
        join_phase = [False]
        with patch.object(triage, "consume_retained_admission", paused), \
             ThreadPoolExecutor(max_workers=1, thread_name_prefix="pending-owner") as pool:
            future = pool.submit(self._deliver)
            try:
                self.assertTrue(entered.wait(10))
                join_phase[0] = True
                refused = self._deliver()
                join_phase[0] = False
                self.assertIn("join refused", json.loads(refused)["error"])
                self.assertIn(self._stable().key, triage._RETAINED_ADMISSIONS,
                              "I3: refused join retired original pending owner")
            finally:
                resume.set()
            self.assertNotIn("error", json.loads(future.result(10)))
        self.assertEqual(len(self.raw_calls), 1)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 1)

    def test_i2_real_chain_late_old_begin_cannot_bind_over_g2(self):
        started, finish = threading.Event(), threading.Event()
        entered, resume = threading.Event(), threading.Event()
        original_begin = ea.ReservationBook._begin
        original_raw = self.raw_callback

        def raw(**kwargs):
            if threading.current_thread().name.startswith("G1"):
                started.set()
                self.assertTrue(finish.wait(10))
                return '{"delegation_id":"G1"}'
            return '{"delegation_id":"G2"}'

        def paused_begin(book, *args, **kwargs):
            outcome = original_begin(book, *args, **kwargs)
            if threading.current_thread().name.startswith("duplicate"):
                self.assertFalse(outcome[0])
                entered.set()
                self.assertTrue(resume.wait(10))
            return outcome

        self.raw_callback = raw
        with patch.object(ea, "_MAX_RECORDS", 1), patch.object(ea.ReservationBook, "_begin", paused_begin), \
             ThreadPoolExecutor(max_workers=1, thread_name_prefix="G1") as owner, \
             ThreadPoolExecutor(max_workers=1, thread_name_prefix="duplicate") as duplicate:
            first = owner.submit(self._deliver)
            try:
                self.assertTrue(started.wait(10))
                delayed = duplicate.submit(self._deliver)
                self.assertTrue(entered.wait(10))
                finish.set()
                first.result(10)
                self._deliver("other")
                self.assertIsNone(ea.retained_invocation_generation(self._stable()), "test did not evict G1")
                g2 = self._deliver()
                self.assertEqual(json.loads(g2)["delegation_id"], "G2")
                generation = triage._RETAINED_ADMISSIONS[self._stable().key].generation
                resume.set()
                delayed.result(10)
                self.assertEqual(triage._RETAINED_ADMISSIONS[self._stable().key].generation, generation,
                                 "I2: real callback late old bind overwrote G2")
                charges = triage._TURN_WORKERS["r12:live"]
                self.assertEqual(self._deliver(), g2)
                self.assertEqual(triage._TURN_WORKERS["r12:live"], charges)
                self.assertEqual(len(self.raw_calls), 3)
            finally:
                finish.set()
                resume.set()
        self.raw_callback = original_raw

    def test_i3_delayed_old_post_callback_cannot_retire_new_nonce_owner(self):
        old_post, release_old = threading.Event(), threading.Event()
        new_pre, release_new = threading.Event(), threading.Event()
        original_consume = triage.consume_retained_admission
        # Observe the real host post callback before the router, so the old
        # delivery keeps its original copied context while a new delivery starts.
        def hold_old_post(**kwargs):
            if kwargs.get("status") == "blocked":
                old_post.set()
                self.assertTrue(release_old.wait(10))
        post = self.ctx.register_hook("post_tool_call", hold_old_post)
        callbacks = self.manager._hooks["post_tool_call"]
        callbacks.remove(hold_old_post)
        callbacks.insert(0, hold_old_post)
        blocker = self.ctx.register_hook("pre_tool_call", lambda **kw:
            {"action": "block", "message": "old refused"})

        def hold_new(invocation):
            if invocation.key[3] == "same":
                new_pre.set()
                self.assertTrue(release_new.wait(10))
            return original_consume(invocation)

        with patch.object(ea, "_MAX_RECORDS", 1), \
             ThreadPoolExecutor(max_workers=2) as pool:
            old = pool.submit(self._deliver)
            try:
                self.assertTrue(old_post.wait(10))
                old_nonce = triage._RETAINED_ADMISSIONS[self._stable().key].nonce
                blocker.dispose()
                self._deliver("other", turn="r12:other")
                with patch.object(triage, "consume_retained_admission", hold_new):
                    fresh = pool.submit(self._deliver)
                    self.assertTrue(new_pre.wait(10))
                    new_nonce = triage._RETAINED_ADMISSIONS[self._stable().key].nonce
                    self.assertNotEqual(new_nonce, old_nonce)
                    release_old.set()
                    old.result(10)
                    self.assertIn(self._stable().key, triage._RETAINED_ADMISSIONS,
                                  "I3: old callback retired replacement pending owner")
                    self.assertEqual(triage._RETAINED_ADMISSIONS[self._stable().key].nonce, new_nonce)
                    release_new.set()
                    self.assertNotIn("error", json.loads(fresh.result(10)))
            finally:
                release_old.set()
                release_new.set()
                post.dispose()
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 2)
        self.assertEqual(len(self.raw_calls), 2)

    def test_i1_unsealed_generation_handoff_cannot_inherit_replacement_at_begin(self):
        started, finish = threading.Event(), threading.Event()
        before_begin, resume = threading.Event(), threading.Event()
        original_begin = ea.ReservationBook._begin

        def raw(**kw):
            if threading.current_thread().name.startswith("active"):
                started.set()
                self.assertTrue(finish.wait(10))
            return self.raw_result

        def paused_begin(book, *args, **kwargs):
            if threading.current_thread().name.startswith("unsealed-join"):
                before_begin.set()
                self.assertTrue(resume.wait(10))
            return original_begin(book, *args, **kwargs)

        self.raw_callback = raw
        with patch.object(ea, "_MAX_RECORDS", 1), patch.object(ea.ReservationBook, "_begin", paused_begin), \
             ThreadPoolExecutor(max_workers=1, thread_name_prefix="active") as owner, \
             ThreadPoolExecutor(max_workers=1, thread_name_prefix="unsealed-join") as join:
            first = owner.submit(self._deliver)
            try:
                self.assertTrue(started.wait(10))
                delayed = join.submit(self._deliver)
                self.assertTrue(before_begin.wait(10))
                finish.set()
                first.result(10)
                self._deliver("other")
                self._deliver()
                charges = triage._TURN_WORKERS["r12:live"]
                resume.set()
                self.assertIn("error", json.loads(delayed.result(10)),
                              "I1: unsealed old generation inherited G2's reply/authorization")
                self.assertEqual(len(self.raw_calls), 3)
                self.assertEqual(triage._TURN_WORKERS["r12:live"], charges)
            finally:
                finish.set()
                resume.set()

    def test_i2_all_binding_modes_reject_evicted_generation_including_missing_entry(self):
        for create, missing in ((False, False), (True, False), (True, True)):
            with self.subTest(create=create, missing=missing):
                triage.reset_state()
                ea.RESERVATIONS = ea.ReservationBook()
                self.assertNotIn("error", json.loads(self._deliver()))
                stable = self._stable()
                old_generation = ea.retained_invocation_generation(stable)
                if missing:
                    triage._RETAINED_ADMISSIONS.pop(stable.key)
                entered, resume = threading.Event(), threading.Event()
                purge = triage._purge_retained_admissions_locked

                def paused_purge():
                    result = purge()
                    if threading.current_thread().name.startswith("metadata-bind"):
                        entered.set()
                        self.assertTrue(resume.wait(10))
                    return result

                with patch.object(triage, "_purge_retained_admissions_locked", paused_purge), \
                     patch.object(ea, "_MAX_RECORDS", 1), \
                     ThreadPoolExecutor(max_workers=1, thread_name_prefix="metadata-bind") as pool:
                    future = pool.submit(ea._bind_retained_triage_admission, stable, old_generation, create=create)
                    try:
                        self.assertTrue(entered.wait(10))
                        book = ea.RESERVATIONS
                        other = self._stable("other")
                        fresh, claim = book._begin(other.scope, other.key, None, 1, other.transport, other.fingerprint)
                        self.assertTrue(fresh)
                        book._seal_tool_response(other.key, claim, "other")
                        fresh, claim = book._begin(stable.scope, stable.key, None, 1, stable.transport, stable.fingerprint)
                        self.assertTrue(fresh)
                        book._seal_tool_response(stable.key, claim, "G2")
                        resume.set()
                        self.assertFalse(future.result(10), "I2: stale generation accepted by completion binding")
                    finally:
                        resume.set()
                # An initially valid G1 mirror may remain until lazy lookup;
                # rejecting this write must not turn it into a usable exemption.
                self.assertIsNone(triage.retained_replay_preflight("delegate_claude",
                    {"goal": "registered", "tier": "haiku"},
                    {"session_id": "r12-live", "turn_id": "r12:live", "tool_call_id": "same"}))
                self.assertNotIn(stable.key, triage._RETAINED_ADMISSIONS)

    def test_n1_lost_sealed_replay_refuses_later_host_rewrite_and_preserves_exact_original(self):
        original = self._deliver()
        stable = self._stable()
        generation = triage._RETAINED_ADMISSIONS[stable.key].generation
        modifier = self.ctx.register_hook("pre_tool_call", lambda **kw:
            {"action": "modify", "args": {"goal": "sealed-rewritten", "tier": "sonnet5"}})
        rewritten = self._deliver()
        self.assertIn("error", json.loads(rewritten),
                      "N1: lost sealed replay delivered original public bytes for rewritten payload")
        self.assertNotEqual(rewritten, original,
                            "N1: later rewritten payload received the sealed original response")
        self.assertEqual(len(self.raw_calls), 1)
        self.assertEqual(len(self.audits), 1)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 1)
        self.assertEqual(triage._RETAINED_ADMISSIONS[stable.key].generation, generation)
        modifier.dispose()
        self.assertEqual(self._deliver(), original,
                         "N1: exact original replay was lost after refusing rewritten payload")
        self.assertEqual(len(self.raw_calls), 1)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 1)

    def test_i3_r_unretainable_later_rewrite_retires_original_owner_before_fresh_retry(self):
        for suffix, unretainable in (("nan", float("nan")), ("bytes", b"unretainable")):
            with self.subTest(shape=suffix):
                call_id = f"unretainable-{suffix}"
                stable = self._stable(call_id)
                charges_before = triage._TURN_WORKERS.get("r12:live", 0)
                raw_before, audits_before = len(self.raw_calls), len(self.audits)
                modifier = self.ctx.register_hook("pre_tool_call", lambda **kw:
                    {"action": "modify", "args": {"goal": "unretainable-rewrite", "tier": "haiku",
                                                       "unretainable": unretainable}})
                refused = self._deliver(call_id)
                self.assertIn("error", json.loads(refused))
                self.assertEqual(len(self.raw_calls), raw_before)
                self.assertNotIn(stable.key, triage._RETAINED_ADMISSIONS,
                                 "I3-R: unretainable final rewrite stranded the original pending owner")
                modifier.dispose()
                fresh = self._deliver(call_id)
                self.assertNotIn("error", json.loads(fresh))
                self.assertEqual(len(self.raw_calls), raw_before + 1)
                self.assertEqual(len(self.audits), audits_before + 1)
                self.assertEqual(triage._TURN_WORKERS["r12:live"], charges_before + 2,
                                 "I3-R: valid retry borrowed the refused admission instead of charging fresh")

    def test_i5_completed_generation_ttl_expiry_requires_new_charge(self):
        clock = [100.0]
        with patch.object(ea, "_now", side_effect=lambda: clock[0]), \
             patch.object(triage, "_monotonic", side_effect=lambda: clock[0]), \
             patch.object(ea, "_RECORD_TTL_SECONDS", 10):
            first = self._deliver()
            generation = ea.retained_invocation_generation(self._stable())
            clock[0] = 111.0
            self.raw_result = '{"delegation_id":"expired-replacement"}'
            second = self._deliver()
            self.assertNotEqual(second, first)
            self.assertNotEqual(ea.retained_invocation_generation(self._stable()), generation)
            self.assertEqual(self._deliver(), second)
        self.assertEqual(len(self.raw_calls), 2)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 2)

    def test_i5_unretainable_ids_remain_distinct_legacy_admissions(self):
        self._deliver("", session="", turn="r12:legacy")
        self._deliver("", session="", turn="r12:legacy")
        self.assertEqual(len(self.raw_calls), 2)
        self.assertEqual(triage._TURN_WORKERS["r12:legacy"], 2)
        self.assertEqual(len(triage._RETAINED_ADMISSIONS), 0)

    def test_i3_host_payload_rewrite_refuses_and_retires_original_pending_owner(self):
        modifier = self.ctx.register_hook("pre_tool_call", lambda **kw:
            {"action": "modify", "args": {"goal": "rewritten-after-router"}})
        refused = self._deliver()
        self.assertIn("error", json.loads(refused), "changed host payload reused original admission")
        self.assertEqual(len(self.raw_calls), 0)
        self.assertEqual(len(triage._RETAINED_ADMISSIONS), 0,
                         "host rewrite refusal left reusable no-F05 ownership")
        modifier.dispose()
        self.assertNotIn("error", json.loads(self._deliver()))
        self.assertEqual(len(self.raw_calls), 1)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 2)

    def test_i3_five_real_later_refusals_across_turns_are_retired_without_refund(self):
        statuses = []
        self.ctx.register_hook("post_tool_call", lambda **kw: statuses.append(kw["status"]))
        blocker = self.ctx.register_hook("pre_tool_call", lambda **kw:
            {"action": "block", "message": "later guard"})
        with patch.object(ea, "_MAX_RECORDS", 2):
            for number in range(5):
                result = self._deliver(f"refused-{number}", turn=f"r12:refused-{number}")
                self.assertIn("later guard", json.loads(result)["error"])
                self.assertEqual(len(triage._RETAINED_ADMISSIONS), 0)
                self.assertEqual(triage._TURN_WORKERS[f"r12:refused-{number}"], 1)
            self.assertEqual(len(ea.RESERVATIONS.records()), 0)
            self.assertEqual(len(self.raw_calls), 0)
            self.assertEqual(sum(triage._TURN_WORKERS.values()), 5)
            blocker.dispose()
            first = self._deliver("refused-0", turn="r12:refused-0")
            self.assertEqual(self._deliver("refused-0", turn="r12:refused-0"), first)
        self.assertEqual(statuses, ["blocked"] * 5 + ["ok", "ok"])
        self.assertEqual(triage._TURN_WORKERS["r12:refused-0"], 2)
        self.assertEqual(len(self.raw_calls), 1)

    def test_i3_consumed_but_unbound_handoffs_are_size_bounded_before_begin(self):
        entered = {str(i): threading.Event() for i in range(5)}
        resume = {str(i): threading.Event() for i in range(5)}
        original_begin = ea.ReservationBook._begin
        def paused_begin(book, scope, key, *args, **kwargs):
            entered[key[3]].set()
            self.assertTrue(resume[key[3]].wait(10))
            return original_begin(book, scope, key, *args, **kwargs)
        with patch.object(ea, "_MAX_RECORDS", 2), patch.object(ea.ReservationBook, "_begin", paused_begin), \
             ThreadPoolExecutor(max_workers=5) as pool:
            pending = []
            try:
                for number in range(5):
                    call = str(number)
                    pending.append(pool.submit(self._deliver, call, turn=f"r12:started-{number}"))
                    self.assertTrue(entered[call].wait(10))
                self.assertLessEqual(len(triage._RETAINED_ADMISSIONS), 2,
                                     "I3: consumed but no-F05 bookkeeping escaped the pending size bound")
                self.assertEqual(len(ea.RESERVATIONS.records()), 0)
                self.assertEqual(len(self.raw_calls), 0)
                resume["0"].set()
                self.assertIn("error", json.loads(pending[0].result(10)))
                self.assertEqual(len(self.raw_calls), 0, "lost consumed owner launched raw")
            finally:
                for event in resume.values():
                    event.set()
            for future in pending:
                future.result(10)
        self.assertEqual(sum(triage._TURN_WORKERS.values()), 5)

    def test_i5_real_registered_host_later_refusal_retires_then_recharges_fresh_delivery(self):
        raw_calls, audits, order, post_seen = [], [], [], []
        raw = json.dumps({"delegation_id": "registered-child"})

        def block_after_router(**kwargs):
            order.append("later-host-guard")
            return {"action": "block", "message": "host later refusal"}

        self.ctx.register_hook("post_tool_call", lambda **kwargs: post_seen.append(kwargs.get("status")))
        blocker = self.ctx.register_hook("pre_tool_call", block_after_router)
        with patch.object(router, "_load_config", return_value=self.cfg), \
             patch.object(router, "_delegation_targets_detail", return_value={}), \
             patch.object(triage, "_caller", return_value=self.root), \
             patch.object(claude, "_host", return_value=(lambda **kw: raw_calls.append(kw) or raw, lambda: self.root)), \
             patch.object(claude.usage_guard, "read", return_value=usage_guard.Reading(10.0, 0.0, None, None, time.time())), \
             patch.object(claude, "_audit", side_effect=lambda *args, **kwargs: audits.append((args, kwargs))):
            refused = self._deliver()
            admissions_after_refusal = len(triage._RETAINED_ADMISSIONS)
            states_after_refusal = [entry.state for entry in triage._RETAINED_ADMISSIONS.values()]
            blocker.dispose()
            first = self._deliver()
            replay = self._deliver()
        self.assertIn("host later refusal", json.loads(refused)["error"])
        self.assertEqual(post_seen, ["blocked", "ok", "ok"],
                         "I5 setup did not traverse the host post-tool lifecycle")
        self.assertEqual(admissions_after_refusal, 0,
                         f"I5: real post_tool_call did not retire the host-refused pending owner ({states_after_refusal})")
        self.assertEqual(len(triage._RETAINED_ADMISSIONS), 1,
                         "I5 setup must leave only the fresh registered invocation mirror")
        self.assertEqual(first, replay)
        self.assertEqual(json.loads(first)["claude_tier"], "haiku")
        self.assertEqual(raw_calls and len(raw_calls), 1)
        self.assertEqual(triage._TURN_WORKERS["r12:live"], 2,
                         "I5/I3: real host later refusal left a reusable unbound authorization")
        self.assertEqual(order, ["later-host-guard"])
        self.assertEqual(len(audits), 1)


if __name__ == "__main__":
    unittest.main()
