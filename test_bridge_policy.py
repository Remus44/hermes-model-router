import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import model_router as router
from model_router import _maybe_run_opus5, route_llm_request, usage_guard


class BridgePolicyTests(unittest.TestCase):
    def run_bridge(self, label, switches, weekly=10):
        cfg = {"callable": switches, "coding_agent": {"delegated_review": {"enabled": True}},
               "usage_guard": {"accounts": {"anthropic": {
                   "soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}}}}}
        request = {"messages": [{"role": "user", "content": f"[{label}-review] Review parser"}]}
        with patch("model_router._verified_delegated_claude_review", return_value=(Path("/tmp"), label)), \
             patch("model_router.usage_guard.read", return_value=usage_guard.Reading(weekly, 10, None, None, time.time())), \
             patch("model_router._run_opus5_bridge", return_value={"result": "reviewed", "model": "claude-sonnet-5-5"}) as bridge:
            _maybe_run_opus5(request, cfg, platform="subagent", api_mode="codex_responses")
        return bridge

    def test_sonnet_obeys_its_own_switch(self):
        self.run_bridge("sonnet", {"sonnet5": False, "opus5": True}).assert_not_called()
        self.run_bridge("sonnet", {"sonnet5": True, "opus5": False}).assert_called_once()

    def test_hard_limit_blocks_both_cli_tiers(self):
        for label in ("opus", "sonnet"):
            self.run_bridge(label, {"sonnet5": True, "opus5": True}, weekly=95).assert_not_called()

    def test_soft_limit_selects_only_an_enabled_lighter_tier(self):
        bridge = self.run_bridge("opus", {"sonnet5": True, "opus5": True}, weekly=75)
        self.assertEqual(bridge.call_args.kwargs["model"], "sonnet")
        self.assertEqual(bridge.call_args.kwargs["requested_alias"], "opus")
        self.assertIn("opus5→sonnet5", bridge.call_args.kwargs["adjustment"])
        self.run_bridge("opus", {"sonnet5": False, "opus5": True}, weekly=75).assert_not_called()

    def test_append_user_instruction_preserves_supported_wire_shapes(self):
        cases = (
            ({"messages": [{"role": "user", "content": "review"}]}, "messages", "text"),
            ({"input": [{"role": "user", "content": [{"type": "input_text", "text": "review"}]}]},
             "input", "input_text"),
            ({"messages": [{"role": "user", "content": [{"type": "text", "text": "review"}]}]},
             "messages", "text"),
        )
        for request, key, expected_type in cases:
            with self.subTest(key=key, expected_type=expected_type):
                router._append_user_instruction(request, " replacement provenance")
                content = request[key][-1]["content"]
                if isinstance(content, str):
                    self.assertEqual(content, "review replacement provenance")
                else:
                    self.assertEqual(content[-1], {"type": expected_type, "text": " replacement provenance"})


class DelegatedReviewRepositoryTests(unittest.TestCase):
    def setUp(self):
        for cache_name in ("_REPO_DIRECTORY_CACHE", "_DISPATCH_REVIEW_REPOSITORIES"):
            cache = getattr(router, cache_name, None)
            if cache is not None:
                cache.clear()
        self.addCleanup(self._clear_router_caches)

    def _clear_router_caches(self):
        for cache_name in ("_REPO_DIRECTORY_CACHE", "_DISPATCH_REVIEW_REPOSITORIES"):
            cache = getattr(router, cache_name, None)
            if cache is not None:
                cache.clear()

    def _config(self, **coding_overrides):
        coding = {"delegated_review": {"enabled": True, "models": ["sonnet", "opus"]}}
        coding.update(coding_overrides)
        return {"callable": {"sonnet5": True, "opus5": True}, "coding_agent": coding}

    def _git_repo(self, directory):
        repo = Path(directory) / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", str(repo)], check=True, capture_output=True)
        return repo

    def test_goal_absolute_path_uses_its_git_top_level(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            child = repo / "nested"
            child.mkdir()
            with patch("model_router.shutil.which", return_value="/claude"):
                routed = router._verified_delegated_claude_review(
                    f"[sonnet-review] Review repository {child}.", self._config())
        self.assertEqual(routed, (repo.resolve(), "sonnet"))

    def test_git_probe_timeout_is_not_a_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch("model_router.subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)) as run:
                resolved = router._repo_directory(directory, git_top_level=True)
        self.assertIsNone(resolved)
        self.assertEqual(run.call_args.kwargs["timeout"], 5)

    def test_git_probe_is_memoised_for_sixty_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            completed = subprocess.CompletedProcess([], 0, stdout=str(root) + "\n")
            with patch("model_router.subprocess.run", return_value=completed) as run, \
                 patch("model_router.time.monotonic", side_effect=[100.0, 159.0]):
                first = router._repo_directory(root, git_top_level=True)
                second = router._repo_directory(root, git_top_level=True)
        self.assertEqual(first, root)
        self.assertEqual(second, root)
        run.assert_called_once()

    def test_failed_git_probe_is_memoised_for_sixty_seconds(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory).resolve()
            with patch("model_router.subprocess.run", side_effect=subprocess.TimeoutExpired("git", 5)) as run, \
                 patch("model_router.time.monotonic", side_effect=[100.0, 159.0]):
                first = router._repo_directory(root, git_top_level=True)
                second = router._repo_directory(root, git_top_level=True)
        self.assertIsNone(first)
        self.assertIsNone(second)
        run.assert_called_once()

    def test_goal_repository_skips_a_non_git_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch-notes"
            scratch.mkdir()
            resolved = router._goal_repository(f"[sonnet-review] Review {scratch}")
        self.assertIsNone(resolved)

    def test_goal_repository_skips_a_non_git_directory_before_a_repository(self):
        with tempfile.TemporaryDirectory() as directory:
            scratch = Path(directory) / "scratch-notes"
            scratch.mkdir()
            repo = self._git_repo(directory)
            resolved = router._goal_repository(f"[sonnet-review] Review {scratch} then {repo}")
        self.assertEqual(resolved, repo.resolve())

    def test_goal_repository_resolves_a_backticked_repository_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            resolved = router._goal_repository(f"[sonnet-review] Review `{repo}`")
        self.assertEqual(resolved, repo.resolve())

    def test_goal_repository_resolves_a_parenthesised_repository_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            resolved = router._goal_repository(f"[sonnet-review] Review ({repo})")
        self.assertEqual(resolved, repo.resolve())

    def test_goal_repository_still_resolves_a_plain_repository_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            resolved = router._goal_repository(f"[sonnet-review] Review {repo}")
        self.assertEqual(resolved, repo.resolve())

    def test_goal_repository_does_not_take_a_url_as_a_path(self):
        self.assertIsNone(router._goal_repository("[sonnet-review] Review https://example.com/a"))

    def test_workspace_path_in_child_request_resolves_a_review_without_goal_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            request = {
                "instructions": f"WORKSPACE PATH:\n{repo}\nUse this exact path.",
                "messages": [{"role": "user", "content": "[sonnet-review] Review parser"}],
            }
            with patch("model_router.shutil.which", return_value="/claude"):
                routed = router._verified_delegated_claude_review(
                    "[sonnet-review] Review parser", self._config(), request=request)
        self.assertEqual(routed, (repo.resolve(), "sonnet"))

    def test_workspace_path_uses_the_last_host_appended_block(self):
        with tempfile.TemporaryDirectory() as directory:
            context_repo = self._git_repo(directory)
            workspace_repo = Path(directory) / "workspace"
            workspace_repo.mkdir()
            subprocess.run(["git", "init", str(workspace_repo)], check=True, capture_output=True)
            request = {
                "instructions": (
                    f"CONTEXT:\nWORKSPACE PATH:\n{context_repo}\nPasted brief.\n"
                    f"WORKSPACE PATH:\n{workspace_repo}\nUse this exact path."
                ),
                "messages": [{"role": "user", "content": "[sonnet-review] Review parser"}],
            }
            with patch("model_router.shutil.which", return_value="/claude"):
                routed = router._verified_delegated_claude_review(
                    "[sonnet-review] Review parser", self._config(), request=request)
        self.assertEqual(routed, (workspace_repo.resolve(), "sonnet"))

    def test_workspace_path_in_a_chat_system_message_resolves_a_review(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            request = {
                "messages": [
                    {"role": "system", "content": f"WORKSPACE PATH:\n{repo}\nUse this exact path."},
                    {"role": "user", "content": "[sonnet-review] Review parser"},
                ],
            }
            with patch("model_router.shutil.which", return_value="/claude"):
                routed = router._verified_delegated_claude_review(
                    "[sonnet-review] Review parser", self._config(), request=request)
        self.assertEqual(routed, (repo.resolve(), "sonnet"))

    def test_nonexistent_goal_and_shipped_aliases_are_skipped(self):
        cfg = self._config(repo_aliases={"router": "/nonexistent/model-router-alias-target"})
        with patch("model_router.shutil.which", return_value="/claude"):
            routed = router._verified_delegated_claude_review(
                "[sonnet-review] Review /not/a/repository, router", cfg)
        self.assertIsNone(routed)

    def test_a_path_the_os_refuses_is_skipped_rather_than_raised(self):
        # A component over NAME_MAX makes is_dir() raise ENAMETOOLONG; raising here
        # would let Hermes skip the admission guard, so it must read as "no repo".
        cfg = self._config()
        with patch("model_router.shutil.which", return_value="/claude"):
            routed = router._verified_delegated_claude_review(
                "[sonnet-review] Review /tmp/" + "x" * 300 + "/src", cfg)
        self.assertIsNone(routed)

    def test_resolved_sonnet_review_calls_the_cli_bridge(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            request = {
                "instructions": f"WORKSPACE PATH:\n{repo}\nUse this exact path.",
                "messages": [{"role": "user", "content": "[sonnet-review] Review parser"}],
            }
            with patch("model_router.shutil.which", return_value="/claude"), \
                 patch("model_router.usage_guard.read", return_value=usage_guard.Reading(10, 0, None, None, time.time())), \
                 patch("model_router._run_opus5_bridge", return_value={
                     "result": "reviewed", "effective_model": "claude-sonnet-5-5"}) as bridge:
                result = _maybe_run_opus5(request, self._config(), platform="subagent",
                                          api_mode="codex_responses")
        self.assertEqual(result.model, "claude-sonnet-5-5")
        bridge.assert_called_once()
        identity = bridge.call_args.kwargs["identity"]
        self.assertEqual(identity.transport, "claude_cli")
        self.assertEqual(identity.observed.value, "unknown")
        self.assertEqual(bridge.call_args.kwargs["repo"], str(repo.resolve()))

    def test_dispatch_resolved_repository_carries_to_child_without_workspace_path(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            goal = "[sonnet-review] Review parser"
            child_request = {"messages": [{"role": "user", "content": goal}]}
            with patch("model_router.shutil.which", return_value="/claude"), \
                 patch("model_router.usage_guard.read", return_value=usage_guard.Reading(10, 0, None, None, time.time())), \
                 patch("model_router._run_opus5_bridge", return_value={
                     "result": "reviewed", "effective_model": "claude-sonnet-5-5"}) as bridge:
                dispatch, reason = router._delegated_claude_review_status(
                    goal, self._config(), dispatch_cwd=repo)
                result = _maybe_run_opus5(child_request, self._config(), platform="subagent",
                                          api_mode="codex_responses")
        self.assertEqual(reason, "")
        self.assertEqual(dispatch, (repo.resolve(), "sonnet"))
        self.assertEqual(result.model, "claude-sonnet-5-5")
        self.assertEqual(bridge.call_args.kwargs["repo"], str(repo.resolve()))

    def test_dispatch_resolved_repository_expires_after_one_hour(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            goal = "[sonnet-review] Review parser"
            clock = {"value": 100.0}
            with patch("model_router.shutil.which", return_value="/claude"), \
                 patch("model_router.time.monotonic", side_effect=lambda: clock["value"]):
                dispatch, reason = router._delegated_claude_review_status(
                    goal, self._config(), dispatch_cwd=repo)
                clock["value"] = 3701.0
                execution = router._verified_delegated_claude_review(
                    goal, self._config(), request={"messages": [{"role": "user", "content": goal}]})
        self.assertEqual(reason, "")
        self.assertEqual(dispatch, (repo.resolve(), "sonnet"))
        self.assertIsNone(execution)

    def test_dispatch_repository_matches_a_goal_with_an_appended_worktree_note(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            goal = "[sonnet-review] Review the parser's isolated worktree behavior"
            execution_text = goal + "\n\nWorktree: child isolation appended this note."
            with patch("model_router.shutil.which", return_value="/claude"):
                dispatch, reason = router._delegated_claude_review_status(
                    goal, self._config(), dispatch_cwd=repo)
                execution = router._verified_delegated_claude_review(
                    execution_text, self._config(), request={
                        "messages": [{"role": "user", "content": execution_text}],
                    })
        self.assertEqual(reason, "")
        self.assertEqual(dispatch, (repo.resolve(), "sonnet"))
        self.assertEqual(execution, (repo.resolve(), "sonnet"))

    def test_a_too_short_goal_does_not_prefix_match(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            goal = "[sonnet-review] hi"
            execution_text = goal + " please, this is urgent"
            with patch("model_router.shutil.which", return_value="/claude"):
                dispatch, reason = router._delegated_claude_review_status(
                    goal, self._config(), dispatch_cwd=repo)
                execution = router._verified_delegated_claude_review(
                    execution_text, self._config(), request={
                        "messages": [{"role": "user", "content": execution_text}],
                    })
        self.assertEqual(reason, "")
        self.assertEqual(dispatch, (repo.resolve(), "sonnet"))
        self.assertIsNone(execution)

    def test_same_goal_with_two_repositories_falls_through_to_workspace(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._git_repo(directory)
            second = Path(directory) / "second"
            second.mkdir()
            subprocess.run(["git", "init", str(second)], check=True, capture_output=True)
            goal = "[sonnet-review] Review the parser's isolated worktree behavior"
            router._remember_dispatch_review_repository(goal, first)
            router._remember_dispatch_review_repository(goal, second)
            request = {
                "instructions": f"WORKSPACE PATH:\n{first}\nUse this exact path.",
                "messages": [{"role": "user", "content": goal}],
            }
            resolved = router._delegated_review_repository(goal, self._config(), request=request)
        self.assertIsNone(router._remembered_dispatch_review_repository(goal))
        self.assertEqual(resolved, first.resolve())

    def test_dispatch_from_a_second_repository_is_not_overridden_by_the_first(self):
        with tempfile.TemporaryDirectory() as directory:
            first = self._git_repo(directory)
            second = Path(directory) / "second"
            second.mkdir()
            subprocess.run(["git", "init", str(second)], check=True, capture_output=True)
            goal = "[sonnet-review] Review the parser's isolated worktree behavior"
            with patch("model_router.shutil.which", return_value="/claude"):
                first_dispatch, first_reason = router._delegated_claude_review_status(
                    goal, self._config(), dispatch_cwd=first, at_dispatch=True)
                second_dispatch, second_reason = router._delegated_claude_review_status(
                    goal, self._config(), dispatch_cwd=second, at_dispatch=True)
        self.assertEqual(first_reason, "")
        self.assertEqual(second_reason, "")
        self.assertEqual(first_dispatch, (first.resolve(), "sonnet"))
        self.assertEqual(second_dispatch, (second.resolve(), "sonnet"))
        self.assertIsNone(router._remembered_dispatch_review_repository(goal))

    def test_prefix_match_requires_a_word_boundary(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            goal = "[sonnet-review] Review the parser"
            execution_text = "[sonnet-review] Review the parsers of the other project"
            with patch("model_router.shutil.which", return_value="/claude"):
                dispatch, reason = router._delegated_claude_review_status(
                    goal, self._config(), dispatch_cwd=repo)
                execution = router._verified_delegated_claude_review(
                    execution_text, self._config(), request={
                        "messages": [{"role": "user", "content": execution_text}],
                    })
        self.assertEqual(reason, "")
        self.assertEqual(dispatch, (repo.resolve(), "sonnet"))
        self.assertIsNone(execution)

    def test_a_remembered_repository_that_vanished_is_dropped(self):
        with tempfile.TemporaryDirectory() as directory:
            repo = self._git_repo(directory)
            goal = "[sonnet-review] Review parser"
            with patch("model_router.shutil.which", return_value="/claude"):
                dispatch, reason = router._delegated_claude_review_status(
                    goal, self._config(), dispatch_cwd=repo)
                shutil.rmtree(repo)
                execution = router._verified_delegated_claude_review(
                    goal, self._config(), request={
                        "messages": [{"role": "user", "content": goal}],
                    })
        self.assertEqual(reason, "")
        self.assertEqual(dispatch, (repo.resolve(), "sonnet"))
        self.assertIsNone(execution)

    def test_cache_helpers_use_the_shared_repository_cache_lock(self):
        class CountingLock:
            def __init__(self):
                self.entries = 0

            def __enter__(self):
                self.entries += 1

            def __exit__(self, *_):
                return False

        cache = router.OrderedDict()
        lock = CountingLock()
        with patch.object(router, "_REPOSITORY_CACHE_LOCK", lock):
            router._bounded_cache_put(cache, "repo", Path("/tmp"), max_entries=1)
            found, value = router._bounded_cache_get(cache, "repo", 60)
        self.assertTrue(found)
        self.assertEqual(value, Path("/tmp"))
        self.assertEqual(lock.entries, 2)

    def test_review_model_must_match_its_claude_tier_before_admission(self):
        cfg = self._config()
        with patch("model_router.shutil.which", return_value="/claude"), \
             patch("model_router._delegation_targets_detail", return_value={
                 "sonnet5": {"provider": "anthropic", "model": "claude-sonnet-5-5"},
                 "terra": {"provider": "openai-codex", "model": "gpt-terra"},
             }):
            claude, claude_reason = router._delegated_claude_review_status(
                "[sonnet-review] Review parser", cfg, requested_model="sonnet5")
            other, other_reason = router._delegated_claude_review_status(
                "[sonnet-review] Review parser", cfg, requested_model="terra")
        self.assertIsNone(claude)
        self.assertEqual(claude_reason, "no repository could be resolved")
        self.assertIsNone(other)
        self.assertEqual(other_reason, "task names model terra")

    def test_route_reason_names_a_missing_claude_cli_for_a_review_leaf(self):
        cfg = {
            "enabled": True, "provider": "openai-codex", "models": {"terra": "gpt-terra"},
            "callable": {"terra": True, "sonnet5": True},
            "tier_providers": {"terra": "openai-codex", "sonnet5": "openai-codex"},
            "coding_agent": {"delegated_review": {"enabled": True, "models": ["sonnet"]}},
        }
        request = {"model": "gpt-terra", "messages": [{"role": "user", "content":
                   "[sonnet-review] Review parser"}]}
        with patch("model_router._load_config", return_value=cfg), \
             patch("model_router.shutil.which", return_value=None), \
             patch("model_router._log_decision"):
            routed = route_llm_request(request=request, provider="openai-codex", model="gpt-terra",
                                       platform="subagent", turn_id="root:sa-1", api_call_count=1)
        self.assertIn("Claude review not taken (Claude CLI is unavailable)", routed["reason"])


class AccountOfExecutionTests(unittest.TestCase):
    def _route(self, *, codex, claude, bridge_error=None, pre_bridge_error=None, label='sonnet', worker=True, eligible=True):
        from unittest.mock import Mock
        from model_router import run_llm_with_transient_failover
        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'opus5': True, 'sonnet5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {'enabled': True}},
            'usage_guard': {'accounts': {
                'anthropic': {'soft_percent': 70, 'hard_percent': 90},
                'openai-codex': {'soft_percent': 70, 'hard_percent': 90},
            }},
        }
        reading = lambda weekly: usage_guard.Reading(weekly, 10, None, None, time.time())
        downstream = Mock(return_value='Codex ran')
        bridge = Mock(return_value={'result': 'Claude reviewed', 'effective_model':
                                    'claude-sonnet-5-5' if label == 'sonnet' else 'claude-opus-5-5'})
        if bridge_error:
            bridge.side_effect = bridge_error
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   f'[{label}-review] Review parser'}]}
        with patch('model_router._load_config', return_value=cfg), \
             patch('model_router._verified_delegated_claude_review',
                   return_value=(Path('/tmp'), label) if eligible else None,
                   side_effect=pre_bridge_error), \
             patch('model_router.usage_guard.read', return_value=reading(claude)) as claude_read, \
             patch('model_router.usage_guard.peek', return_value=reading(codex)), \
             patch('model_router._run_opus5_bridge', bridge):
            result = run_llm_with_transient_failover(
                request=request, original_request=request, next_call=downstream,
                provider='openai-codex', api_mode='codex_responses', api_call_count=1,
                platform='subagent' if worker else 'cli', turn_id='s:sa-1' if worker else 'root')
        return result, downstream, bridge, claude_read

    def _host_review_attempt(self, *, selection_mode, bridge_result, alias='sonnet', log_path=None,
                             routes_path=None):
        from hermes_cli.middleware import run_llm_execution_middleware
        from model_router.execution_contracts import ModelFact, TargetIdentity
        from model_router.target_identity import RESOLVED

        identity = TargetIdentity('anthropic', 'unknown', 'claude_cli', alias, selection_mode,
            requested=ModelFact(f'claude-{alias}-5-5', 'fixture'))
        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'sonnet5': True, 'opus5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {
                'enabled': True, 'selection_mode': selection_mode,
                'requested_model': f'claude-{alias}-5-5',
            }},
        }
        if log_path is not None:
            cfg['claude_delegation'] = {'log_path': str(log_path)}
        if routes_path is not None:
            cfg['logging'] = {'enabled': True, 'path': str(routes_path)}
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   f'[{alias}-review] Review parser'}]}
        downstream = Mock(return_value=SimpleNamespace(model='gpt-terra', provider='openai-codex'))
        manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                  _report_hook_failure=Mock())
        with patch('hermes_cli.plugins._delivery_manager', return_value=manager), \
             patch.object(router, '_load_config', return_value=cfg), \
             patch.object(router, '_verified_delegated_claude_review', return_value=(Path('/tmp'), alias)), \
             patch('model_router.target_identity.resolve_target', return_value=SimpleNamespace(
                 status=RESOLVED, identity=identity, reasons=())), \
             patch.object(router.usage_guard, 'guarded', return_value=False), \
             patch.object(router.worker_admission, 'refusal', return_value=''), \
             patch.object(router, '_run_opus5_bridge', side_effect=bridge_result) as bridge:
            response = run_llm_execution_middleware(request, downstream, provider='openai-codex',
                api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
        return response, downstream, bridge

    def test_closed_codex_does_not_block_healthy_claude(self):
        result, codex, bridge, _ = self._route(codex=95, claude=10)
        self.assertEqual(result.model, 'claude-sonnet-5-5')
        bridge.assert_called_once()
        codex.assert_not_called()

    def test_closed_claude_can_fall_back_only_to_open_codex(self):
        result, codex, bridge, _ = self._route(codex=10, claude=95)
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
        bridge.assert_not_called()

    def test_both_closed_or_failed_bridge_never_reach_codex(self):
        for claude, error in ((95, None), (10, RuntimeError('CLI failed'))):
            result, codex, bridge, _ = self._route(codex=95, claude=claude, bridge_error=error)
            self.assertIn('ROUTER WORKER STOPPED', result.output_text)
            codex.assert_not_called()
            self.assertEqual(bridge.call_count, 0 if claude == 95 else 1)

    def test_a01_exact_refusal_through_host_runner_does_not_dispatch(self):
        from hermes_cli.middleware import run_llm_execution_middleware

        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'sonnet5': True, 'opus5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {
                'enabled': True, 'selection_mode': 'exact', 'requested_model': 'claude-sonnet-5-5',
            }},
        }
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   '[sonnet-review] Review parser'}]}
        downstream = Mock(return_value='ORDINARY PROVIDER EXECUTED')
        manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                  _report_hook_failure=Mock())
        with patch('hermes_cli.plugins._delivery_manager', return_value=manager), \
             patch.object(router, '_load_config', return_value=cfg), \
             patch.object(router, '_verified_delegated_claude_review', return_value=(Path('/tmp'), 'sonnet')), \
             patch.object(router.claude_delegation, '_log'), \
             patch.object(router, '_run_opus5_bridge') as bridge:
            stopped = run_llm_execution_middleware(request, downstream, provider='openai-codex',
                api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
        bridge.assert_not_called()
        self.assertEqual(downstream.call_count, 0, 'exact refusal failed open through the host runner')
        self.assertEqual(stopped.usage.total_tokens, 0)
        self.assertIn('ROUTER WORKER STOPPED', stopped.output_text)
        self.assertIn('No work was performed', stopped.output_text)
        from agent.transports.codex import ResponsesApiTransport
        normalized = ResponsesApiTransport().normalize_response(stopped)
        self.assertIn('ROUTER WORKER STOPPED', normalized.content or '')

    def test_exact_preselection_refusal_does_not_fall_through_when_no_review_route_exists(self):
        from hermes_cli.middleware import run_llm_execution_middleware
        from model_router.hermes_paths import hermes_home

        with tempfile.TemporaryDirectory() as directory:
            routes = Path(directory) / 'routes.jsonl'
            audits = hermes_home() / 'f01-claude.jsonl'
            cfg = {
                'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
                'callable': {'sonnet5': False, 'opus5': True},
                'coding_agent': {'enabled': False, 'delegated_review': {
                    'enabled': True, 'selection_mode': 'exact', 'requested_model': 'claude-sonnet-5-5',
                }},
                'logging': {'enabled': True, 'path': str(routes)},
                'claude_delegation': {'log_path': '~/.hermes/f01-claude.jsonl'},
            }
            request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                       '[sonnet-review] Review parser'}]}
            downstream = Mock(return_value='ORDINARY PROVIDER EXECUTED')
            manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                      _report_hook_failure=Mock())
            with patch('hermes_cli.plugins._delivery_manager', return_value=manager), \
                 patch.object(router, '_load_config', return_value=cfg), \
                 patch.object(router, '_verified_delegated_claude_review', return_value=None), \
                 patch.object(router, '_run_opus5_bridge') as bridge:
                stopped = run_llm_execution_middleware(request, downstream, provider='openai-codex',
                    api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
            audit_entries = [json.loads(line) for line in audits.read_text().splitlines()] if audits.exists() else []
        downstream.assert_not_called()
        bridge.assert_not_called()
        self.assertIn('ROUTER WORKER STOPPED', stopped.output_text)
        self.assertFalse(routes.exists(), 'refusal must not create a routed provider-call record')
        self.assertTrue(audits.exists(), 'refusal must persist a Claude audit record')
        self.assertEqual(len(audit_entries), 1)
        self.assertEqual(audit_entries[0]['outcome'], 'refused')

    def test_a08_exact_attempt_failure_does_not_substitute(self):
        from hermes_cli.middleware import run_llm_execution_middleware
        from model_router.claude_opus_bridge import ClaudeBridgeFailure
        from model_router.execution_contracts import ModelFact, TargetIdentity

        # Future/conditional cell: injected support; the installed CLI remains unsupported.
        identity = TargetIdentity('anthropic', 'unknown', 'claude_cli', 'sonnet', 'exact',
            requested=ModelFact('claude-sonnet-5-5', 'fixture'))
        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'sonnet5': True, 'opus5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {
                'enabled': True, 'selection_mode': 'exact', 'requested_model': 'claude-sonnet-5-5',
            }},
        }
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   '[sonnet-review] Review parser'}]}
        downstream = Mock(return_value=SimpleNamespace(model='gpt-terra'))
        manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                  _report_hook_failure=Mock())
        with patch('hermes_cli.plugins._delivery_manager', return_value=manager), \
             patch.object(router, '_load_config', return_value=cfg), \
             patch.object(router, '_verified_delegated_claude_review', return_value=(Path('/tmp'), 'sonnet')), \
             patch('model_router.target_identity.resolve_target', return_value=SimpleNamespace(
                 status='resolved', identity=identity, reasons=())), \
             patch.object(router.usage_guard, 'guarded', return_value=False), \
             patch.object(router.worker_admission, 'refusal', return_value=''), \
             patch.object(router.claude_delegation, '_log'), \
             patch.object(router, '_run_opus5_bridge', side_effect=ClaudeBridgeFailure(
                 'wrong model', 'model-mismatch', identity=identity)) as bridge:
            stopped = run_llm_execution_middleware(request, downstream, provider='openai-codex',
                api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
        bridge.assert_called_once()
        self.assertEqual(downstream.call_count, 0, 'exact attempted failure substituted on ordinary provider')
        self.assertEqual(stopped.usage.total_tokens, 0)
        self.assertIn('ROUTER WORKER STOPPED', stopped.output_text)

    def test_exact_attempt_failure_kinds_stop_through_host_runner_without_identity(self):
        from model_router.claude_opus_bridge import ClaudeBridgeFailure

        failures = (
            ('timeout', ClaudeBridgeFailure('Claude Code timed out', 'timeout')),
            ('malformed-output', {'result': '', 'effective_model': 'claude-sonnet-5-5'}),
            ('model-mismatch', ClaudeBridgeFailure('wrong model', 'model-mismatch')),
            ('budget', ClaudeBridgeFailure('Claude Code exhausted its budget', 'budget')),
            ('max-turn', ClaudeBridgeFailure('Claude Code reached max turns', 'max-turn')),
            ('nonzero-exit', ClaudeBridgeFailure('Claude Code failed with exit 1: ' + ('x' * 500), 'nonzero-exit')),
            ('process-start', FileNotFoundError('claude')),
        )
        for name, failure in failures:
            with self.subTest(failure=name):
                stopped, downstream, bridge = self._host_review_attempt(
                    selection_mode='exact', bridge_result=failure)
                bridge.assert_called_once()
                downstream.assert_not_called()
                self.assertEqual(stopped.usage.total_tokens, 0)
                self.assertTrue(router._router_stopped_summary(stopped.output_text),
                    'the parent-side stop detector must recognize every exact failure')

    def _exact_preselection_run(self, directory, *, callable_sonnet=True, which='/claude', goal='[sonnet-review] Review parser',
                                cooling=0, repo=Path('/tmp'), allowed=None, resolve_error=None, patches=()):
        from contextlib import ExitStack
        from hermes_cli.middleware import run_llm_execution_middleware

        routes = Path(directory) / 'routes.jsonl'
        audits = Path(directory) / 'claude-audit.jsonl'
        policy = {'enabled': True, 'selection_mode': 'exact', 'requested_model': 'claude-sonnet-5-5',
                  'max_chars': 8000}
        if allowed is not None:
            policy['models'] = allowed
        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'sonnet5': callable_sonnet, 'opus5': True},
            'coding_agent': {'enabled': False, 'delegated_review': policy},
            'logging': {'enabled': True, 'path': str(routes)},
            'claude_delegation': {'log_path': str(audits)},
        }
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content': goal}]}
        downstream = Mock(return_value='ORDINARY PROVIDER EXECUTED')
        manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                  _report_hook_failure=Mock())
        with ExitStack() as stack:
            stack.enter_context(patch('hermes_cli.plugins._delivery_manager', return_value=manager))
            stack.enter_context(patch.object(router, '_load_config', return_value=cfg))
            stack.enter_context(patch.object(router.shutil, 'which', return_value=which))
            stack.enter_context(patch.object(router, '_delegated_review_repository', return_value=repo))
            stack.enter_context(patch.object(router, '_tier_cooldown_remaining',
                                             return_value=cooling))
            stack.enter_context(patch.object(router.worker_admission, 'refusal', return_value=''))
            if resolve_error is not None:
                stack.enter_context(patch('model_router.target_identity.resolve_target',
                                          side_effect=resolve_error))
            for item in patches:
                stack.enter_context(item)
            bridge = stack.enter_context(patch.object(router, '_run_opus5_bridge'))
            launches = stack.enter_context(patch('model_router.claude_opus_bridge.subprocess.run'))
            stopped = run_llm_execution_middleware(request, downstream, provider='openai-codex',
                api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
        audit_entries = [json.loads(line) for line in audits.read_text().splitlines()] if audits.exists() else []
        return stopped, downstream, bridge, launches, routes, audit_entries

    def _assert_stopped_refusal(self, run, expected_audits=1):
        stopped, downstream, bridge, launches, routes, audit_entries = run
        downstream.assert_not_called()
        bridge.assert_not_called()
        launches.assert_not_called()
        self.assertEqual(stopped.usage.total_tokens, 0)
        self.assertTrue(router._router_stopped_summary(stopped.output_text),
                        'the parent detector must see the refusal as not completed')
        self.assertFalse(routes.exists(), 'a refusal must not create a routed provider-call record')
        self.assertEqual(len(audit_entries), expected_audits)
        return stopped

    def test_exact_preselection_refusals_stop_through_host_runner(self):
        cases = {
            'cooling tier': dict(cooling=30),
            'disabled tier': dict(callable_sonnet=False),
            'unavailable Claude CLI': dict(which=None),
            'unavailable repository': dict(repo=None),
            'unavailable context': dict(goal='[sonnet-review] Review parser ' + 'x' * 9000),
            'unknown tier alias map': dict(patches=(
                patch('model_router.claude_opus_bridge.CLAUDE_REVIEW_MODELS', {}),)),
            # Tier/alias map stays known; only the exact_model capability status is unknown.
            'unknown exact_model capability': dict(patches=(
                patch('model_router.target_identity._cli_exact_capability',
                      return_value=('unknown', 'fixture: no exact_model evidence')),)),
        }
        for name, options in cases.items():
            with self.subTest(refusal=name), tempfile.TemporaryDirectory() as directory:
                run = self._exact_preselection_run(directory, **options)
                self._assert_stopped_refusal(run)
                self.assertEqual(run[5][0]['outcome'], 'refused')
                if name == 'unknown exact_model capability':
                    text = json.dumps(run[5][0]) + run[0].output_text
                    self.assertIn('exact_model', text)
                    self.assertNotIn('tier is unknown', text)
                elif name == 'unknown tier alias map':
                    self.assertIn('tier is unknown', json.dumps(run[5][0]))

    def test_exact_preselection_resolve_target_raise_still_stops(self):
        with tempfile.TemporaryDirectory() as directory:
            run = self._exact_preselection_run(directory, repo=None, resolve_error=RuntimeError('boom'))
            self._assert_stopped_refusal(run)

    def test_a08_stopped_evidence_survives_normalizer_and_persists_only_audit(self):
        from agent.transports.codex import ResponsesApiTransport
        from model_router.claude_opus_bridge import ClaudeBridgeFailure

        with tempfile.TemporaryDirectory() as directory:
            routes = Path(directory) / 'routes.jsonl'
            audits = Path(directory) / 'claude-audit.jsonl'
            # Real route logger and real audit logger; counted before the directory is removed.
            stopped, downstream, bridge = self._host_review_attempt(
                selection_mode='exact', log_path=audits, routes_path=routes,
                bridge_result=ClaudeBridgeFailure('wrong model', 'model-mismatch'))
            route_records = ([line for line in routes.read_text().splitlines() if line.strip()]
                            if routes.exists() else [])
            audit_entries = [json.loads(line) for line in audits.read_text().splitlines()]
        downstream.assert_not_called()
        bridge.assert_called_once()
        self.assertEqual(route_records, [], 'A08 stop must persist no routed provider-call record')
        self.assertEqual([entry['outcome'] for entry in audit_entries], ['error'])
        self.assertNotIn('substitution', audit_entries[0])
        normalized = ResponsesApiTransport().normalize_response(stopped)
        self.assertTrue(router._router_stopped_summary(normalized.content or ''),
                        'normalization dropped the not-completed evidence')

    def test_preferred_attempt_failure_kinds_still_replace_once_through_host_runner(self):
        from model_router.claude_opus_bridge import ClaudeBridgeFailure

        failures = (
            ('timeout', ClaudeBridgeFailure('Claude Code timed out', 'timeout')),
            ('malformed-output', {'result': '', 'effective_model': 'claude-sonnet-5-5'}),
            ('model-mismatch', ClaudeBridgeFailure('wrong model', 'model-mismatch')),
            ('budget', ClaudeBridgeFailure('Claude Code exhausted its budget', 'budget')),
            ('max-turn', ClaudeBridgeFailure('Claude Code reached max turns', 'max-turn')),
            ('nonzero-exit', ClaudeBridgeFailure('Claude Code failed with exit 1', 'nonzero-exit')),
            ('process-start', FileNotFoundError('claude')),
        )
        for name, failure in failures:
            with self.subTest(failure=name):
                response, downstream, bridge = self._host_review_attempt(
                    selection_mode='profile_preferred', bridge_result=failure)
                bridge.assert_called_once()
                downstream.assert_called_once()
                self.assertEqual(response.model, 'gpt-terra')

    def test_exact_attempt_failure_audit_has_no_planned_replacement(self):
        from model_router.claude_opus_bridge import ClaudeBridgeFailure

        with tempfile.TemporaryDirectory() as directory:
            audit_path = Path(directory) / 'claude-audit.jsonl'
            stopped, downstream, bridge = self._host_review_attempt(
                selection_mode='exact', bridge_result=ClaudeBridgeFailure('wrong model', 'model-mismatch'),
                log_path=audit_path)
            entries = [json.loads(line) for line in audit_path.read_text().splitlines()]
        bridge.assert_called_once()
        downstream.assert_not_called()
        self.assertTrue(router._router_stopped_summary(stopped.output_text))
        self.assertEqual(len(entries), 1)
        self.assertEqual(entries[0]['outcome'], 'error')
        self.assertNotIn('substitution', entries[0])

    def _parent_visible_provenance(self, response):
        from agent.transports.codex import ResponsesApiTransport
        from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry

        normalized = ResponsesApiTransport().normalize_response(response)
        entry = _build_result_entry(SimpleNamespace(model='parent-visible-fixture'), {
            'final_response': normalized.content, 'completed': True, 'api_calls': 1,
        }, 0, 0.1, _SchemaOutcome(None, None, [], 0))
        prefix = '[ROUTER SUBSTITUTION PROVENANCE v1] '
        self.assertIn(prefix, entry['summary'], 'parent-visible result lost substitution provenance')
        visible, marker = entry['summary'].split(prefix, 1)
        self.assertTrue(visible.strip(), 'provenance marker replaced useful verdict output')
        return json.loads(marker.splitlines()[0])

    # Audit A02: remove when F04 lands
    def test_a02_successful_cli_conversion_preserves_identity_and_substitution(self):
        result = router._opus5_response({
            'result': 'review verdict', 'effective_model': 'claude-sonnet-5-5',
            'identity': {
                'schema_version': 1, 'provider': 'anthropic', 'account': 'unknown',
                'transport': 'claude_cli', 'alias': 'sonnet', 'selection_mode': 'profile_preferred',
                'requested': {'value': 'claude-opus-5-5', 'source': 'operator_request', 'canonical': True},
                'resolved': {'value': 'sonnet', 'source': 'claude_cli.argv', 'canonical': False},
                'observed': {'value': 'claude-sonnet-5-5', 'source': 'claude_cli.result.modelUsage', 'canonical': True},
                'effort': {'requested': 'not_applicable', 'applied': 'unknown', 'source': 'not_observed'},
            },
            'substitution': {
                'policy': 'profile_preferred', 'requested': 'claude-opus-5-5',
                'resolved': 'sonnet', 'observed': 'claude-sonnet-5-5',
                'reason': 'usage step-down',
            },
        })
        provenance = self._parent_visible_provenance(result)
        self.assertEqual(provenance['kind'], 'claude_cli_step_down')
        self.assertEqual(provenance['carrier'], 'middleware')
        self.assertEqual(provenance['requested'], 'claude-opus-5-5')
        self.assertEqual(provenance['resolved'], 'sonnet')
        self.assertEqual(provenance['actual'], {'model': 'unknown', 'provider': 'unknown'})
        self.assertTrue(provenance['ref'])
        self.assertFalse(provenance['satisfies_cross_provider_review'])

    # Audit A02: remove when F04 lands
    def test_a02_replacement_provenance_survives_host_normalization(self):
        verdict = SimpleNamespace(
            output=[SimpleNamespace(type='message', status='completed', content=[
                SimpleNamespace(type='output_text', text='PASS'),
            ])],
            output_text='PASS', status='completed', model='gpt-spark', provider='openai-codex',
        )
        result = router._record_ordinary_replacement_result(verdict, {
            'policy': 'legacy_ordinary_provider_route',
            'requested_review': {'tier': 'opus', 'transport': 'claude_cli'},
            'failure_kind': 'model-mismatch',
            'planned': {'tier': 'terra', 'model': 'gpt-terra', 'provider': 'openai-codex',
                        'source': 'ordinary_provider.request_configuration'},
            'identity': {'requested': {'value': 'claude-opus-5-5', 'source': 'operator_request', 'canonical': True},
                         'resolved': {'value': 'claude-opus-5-5', 'source': 'configured_target', 'canonical': True},
                         'observed': {'value': 'unknown', 'source': 'not_observed', 'canonical': True},
                         'effort': {'requested': 'not_applicable', 'applied': 'unknown', 'source': 'not_observed'}},
        })
        provenance = self._parent_visible_provenance(result)
        self.assertEqual(provenance['kind'], 'ordinary_replacement')
        self.assertEqual(provenance['reason'], 'model-mismatch')
        self.assertEqual(provenance['actual']['model'], 'gpt-spark')
        self.assertEqual(provenance['actual']['provider'], 'openai-codex')
        self.assertEqual(provenance['requested'], 'claude-opus-5-5')
        self.assertFalse(provenance['satisfies_cross_provider_review'])

    def test_replacement_provenance_survives_an_immutable_response(self):
        class ImmutableResponse:
            def __init__(self):
                object.__setattr__(self, 'output', [SimpleNamespace(type='message', status='completed', content=[
                    SimpleNamespace(type='output_text', text='immutable verdict'),
                ])])
                object.__setattr__(self, 'output_text', 'immutable verdict')
                object.__setattr__(self, 'status', 'completed')
                object.__setattr__(self, 'model', None)
                object.__setattr__(self, 'provider', None)
                object.__setattr__(self, '_sealed', True)

            def __setattr__(self, name, value):
                if getattr(self, '_sealed', False):
                    raise TypeError('immutable fixture')
                object.__setattr__(self, name, value)

        result = router._record_ordinary_replacement_result(ImmutableResponse(), {
            'policy': 'legacy_ordinary_provider_route',
            'requested_review': {'tier': 'opus', 'transport': 'claude_cli'},
            'failure_kind': 'timeout',
            'planned': {'tier': 'terra', 'model': 'gpt-terra', 'provider': 'openai-codex',
                        'source': 'ordinary_provider.request_configuration'},
        })
        provenance = self._parent_visible_provenance(result)
        self.assertEqual(provenance['actual']['model'], 'unknown')
        self.assertEqual(provenance['actual']['provider'], 'unknown')

    # --- F04 fix round 1 (review I1-I4): real host normalizer, child projection,
    # installed openai SDK types and the real resolver -> admission -> bridge chain.
    PREFIX = '[ROUTER SUBSTITUTION PROVENANCE v1] '
    REPLACEMENT = {
        'policy': 'legacy_ordinary_provider_route',
        'requested_review': {'tier': 'opus', 'transport': 'claude_cli'},
        'failure_kind': 'timeout', 'planned': {'tier': 'terra', 'model': 'gpt-terra'},
    }

    def _normalized_and_summary(self, response):
        from agent.transports.codex import ResponsesApiTransport
        from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry

        normalized = ResponsesApiTransport().normalize_response(response)
        entry = _build_result_entry(SimpleNamespace(model='parent-visible-fixture'), {
            'final_response': normalized.content, 'completed': True, 'api_calls': 1,
        }, 0, 0.1, _SchemaOutcome(None, None, [], 0))
        return normalized, entry['summary']

    def _marker_line(self, text):
        """The marker must sit on its own line and be exactly one JSON object."""
        lines = [line for line in text.splitlines() if self.PREFIX in line]
        self.assertEqual(len(lines), 1, f'expected exactly one marker line in {text!r}')
        self.assertTrue(lines[0].startswith(self.PREFIX), 'marker shares its line with other text')
        return json.loads(lines[0][len(self.PREFIX):])

    @staticmethod
    def _tool_item():
        return SimpleNamespace(type='function_call', id='fc_review', call_id='call_review',
                               name='read_file', arguments='{"path":"parser.py"}', status='completed')

    def test_f04_i1_installed_sdk_response_keeps_tool_call_usage_and_one_marker(self):
        from openai.types.responses import (Response, ResponseFunctionToolCall,
                                            ResponseOutputMessage, ResponseOutputText)
        text = ResponseOutputText.model_construct(type='output_text', text='SDK verdict', annotations=[])
        message = ResponseOutputMessage.model_construct(
            type='message', status='completed', content=[text], role='assistant', id='msg_review')
        call = ResponseFunctionToolCall.model_construct(
            type='function_call', id='fc_review', call_id='call_review', name='read_file',
            arguments='{"path":"parser.py"}', status='completed')
        usage = SimpleNamespace(input_tokens=9, output_tokens=2, total_tokens=11)
        response = Response.model_construct(output=[message, call], status='completed',
                                            model='gpt-terra', usage=usage)
        before, _ = self._normalized_and_summary(response)

        annotated = router._record_ordinary_replacement_result(response, dict(self.REPLACEMENT))
        normalized, summary = self._normalized_and_summary(annotated)

        self.assertEqual(before.finish_reason, 'tool_calls')
        self.assertEqual(normalized.finish_reason, 'tool_calls', 'annotation changed the finish reason')
        self.assertEqual([c.function.name for c in normalized.tool_calls], ['read_file'])
        self.assertEqual(summary.count(self.PREFIX), 1)
        self.assertEqual(normalized.content.count(self.PREFIX), 1)
        self.assertEqual(self._marker_line(summary)['kind'], 'ordinary_replacement')
        self.assertTrue(summary.startswith('SDK verdict'))
        self.assertIs(annotated.usage, usage)
        self.assertEqual((annotated.status, annotated.model), ('completed', 'gpt-terra'))
        # The caller's response object is never mutated.
        self.assertEqual(len(response.output), 2)
        self.assertEqual(response.output[0].content[0].text, 'SDK verdict')
        self.assertNotIn(self.PREFIX, response.output_text)

    def test_f04_i1_tool_only_replacement_keeps_tool_calls_finish_reason(self):
        usage = SimpleNamespace(input_tokens=9, output_tokens=2, total_tokens=11)
        response = SimpleNamespace(output=[self._tool_item()], output_text='', status='completed',
                                   model='gpt-terra', usage=usage)
        annotated = router._record_ordinary_replacement_result(response, dict(self.REPLACEMENT))
        normalized, summary = self._normalized_and_summary(annotated)

        self.assertEqual(normalized.finish_reason, 'tool_calls', 'tool-only reply was turned into a stop')
        self.assertEqual(len(normalized.tool_calls), 1)
        self.assertEqual(self._marker_line(summary)['kind'], 'ordinary_replacement')
        self.assertIs(annotated.usage, usage)
        self.assertEqual(response.output[0].type, 'function_call')

    def test_f04_i1_slotted_immutable_response_keeps_items_usage_status_and_text(self):
        from collections import namedtuple
        Slotted = namedtuple('Slotted', 'output status model provider usage incomplete_details')
        usage = SimpleNamespace(total_tokens=11)
        output = [SimpleNamespace(type='message', status='completed', content=[
                      SimpleNamespace(type='output_text', text='VERDICT'),
                      SimpleNamespace(type='output_text', text='ACCEPTANCE EVIDENCE')]),
                  self._tool_item()]
        response = Slotted(output, 'completed', 'gpt-terra', 'openai-codex', usage, None)
        annotated = router._record_ordinary_replacement_result(response, dict(self.REPLACEMENT))
        normalized, summary = self._normalized_and_summary(annotated)

        self.assertIs(getattr(annotated, 'usage', None), usage, 'token usage was dropped')
        self.assertEqual((annotated.status, annotated.model, annotated.provider),
                         ('completed', 'gpt-terra', 'openai-codex'))
        self.assertEqual(normalized.finish_reason, 'tool_calls')
        self.assertEqual(len(normalized.tool_calls), 1)
        self.assertIn('VERDICTACCEPTANCE EVIDENCE', summary)
        self.assertEqual(self._marker_line(summary)['actual']['model'], 'gpt-terra')
        self.assertEqual(len(response.output), 2, 'the original output list was mutated')

    def test_f04_i1_failed_aggregate_write_leaves_no_partial_annotation(self):
        tool = self._tool_item()

        class ReadOnlyAggregate:
            def __init__(self):
                self.output = [SimpleNamespace(type='message', status='completed', content=[
                    SimpleNamespace(type='output_text', text='VERDICT')]), tool]
                self.status, self.model, self.usage = 'completed', 'gpt-terra', SimpleNamespace(total_tokens=3)

            @property
            def output_text(self):
                return 'VERDICT'

        response = ReadOnlyAggregate()
        annotated = router._record_ordinary_replacement_result(response, dict(self.REPLACEMENT))
        normalized, summary = self._normalized_and_summary(annotated)

        self.assertEqual(len(response.output), 2, 'items were appended before the failed write')
        self.assertEqual(response.output[0].content[0].text, 'VERDICT')
        self.assertEqual(summary.count(self.PREFIX), 1)
        self.assertEqual(len(normalized.tool_calls), 1)
        self.assertIs(annotated.usage, response.usage)

    def test_f04_i1_reannotation_never_adds_a_second_marker(self):
        response = SimpleNamespace(output=[SimpleNamespace(type='message', status='completed', content=[
            SimpleNamespace(type='output_text', text='VERDICT')])], output_text='VERDICT', status='completed')
        once = router._record_ordinary_replacement_result(response, dict(self.REPLACEMENT))
        twice = router._record_ordinary_replacement_result(once, dict(self.REPLACEMENT))
        _, summary = self._normalized_and_summary(twice)
        self.assertEqual(summary.count(self.PREFIX), 1)

    def test_f04_i1_reasoning_only_reply_is_not_turned_into_a_final_answer(self):
        response = SimpleNamespace(output=[SimpleNamespace(type='reasoning', id='rs_1', summary=[
            SimpleNamespace(type='summary_text', text='thinking')])], status='completed')
        normalized_before, _ = self._normalized_and_summary(
            SimpleNamespace(output=list(response.output), status='completed'))
        annotated = router._record_ordinary_replacement_result(response, dict(self.REPLACEMENT))
        from agent.transports.codex import ResponsesApiTransport
        after = ResponsesApiTransport().normalize_response(annotated)
        self.assertEqual(after.finish_reason, normalized_before.finish_reason)
        self.assertNotIn(self.PREFIX, after.content or '')

    def test_f04_i2_commentary_then_final_puts_marker_in_projected_final_answer(self):
        commentary = SimpleNamespace(type='message', status='completed', phase='commentary', content=[
            SimpleNamespace(type='output_text', text='Inspecting parser')])
        final = SimpleNamespace(type='message', status='completed', phase='final_answer', content=[
            SimpleNamespace(type='output_text', text='PASS: acceptance evidence')])
        response = SimpleNamespace(output=[commentary, final],
                                   output_text='Inspecting parser\nPASS: acceptance evidence',
                                   status='completed', model='gpt-terra', provider='openai-codex')
        annotated = router._record_ordinary_replacement_result(response, dict(self.REPLACEMENT))
        normalized, summary = self._normalized_and_summary(annotated)

        self.assertEqual(normalized.finish_reason, 'stop')
        self.assertNotIn(self.PREFIX, normalized.reasoning or '', 'marker went to the commentary channel')
        self.assertIn('Inspecting parser', normalized.reasoning or '')
        visible, _ = summary.split(self.PREFIX, 1)
        self.assertIn('PASS: acceptance evidence', visible, 'marker is not after the final text')
        self.assertEqual(self._marker_line(summary)['kind'], 'ordinary_replacement')

    def test_f04_i2_multipart_marker_is_its_own_parseable_line_after_all_text(self):
        response = SimpleNamespace(output=[SimpleNamespace(type='message', status='completed', content=[
            SimpleNamespace(type='output_text', text='VERDICT'),
            SimpleNamespace(type='output_text', text='ACCEPTANCE EVIDENCE')])],
            output_text='VERDICT\nACCEPTANCE EVIDENCE', status='completed')
        annotated = router._record_ordinary_replacement_result(response, dict(self.REPLACEMENT))
        _, summary = self._normalized_and_summary(annotated)

        visible, _ = summary.split(self.PREFIX, 1)
        self.assertIn('ACCEPTANCE EVIDENCE', visible)
        self.assertEqual(self._marker_line(summary)['kind'], 'ordinary_replacement')
        self.assertTrue(summary.rstrip().endswith('}'), 'text follows the marker line')

    def test_f04_i3_i4_real_resolution_step_down_reports_admitted_alias(self):
        from model_router import claude_opus_bridge as bridge, target_identity

        cfg = {'coding_agent': {}, 'models': {}}
        identity = target_identity.resolve_target('opus', transport='claude_cli', cfg=cfg).identity
        payload = {'type': 'result', 'subtype': 'success', 'result': 'PASS: parser inspected',
                   'modelUsage': {'claude-sonnet-5-5': {'inputTokens': 9}}, 'num_turns': 1}
        with tempfile.TemporaryDirectory() as repo, \
             patch.object(bridge, '_load_config', return_value=cfg), \
             patch.object(bridge.subprocess, 'run', return_value=SimpleNamespace(
                 stdout=json.dumps(payload), stderr='', returncode=0)) as raw, \
             patch.object(bridge, '_log_decision'), patch.object(bridge, '_append_lifecycle'):
            result = bridge.dispatch('[opus-review] Review parser', Path(repo), review=True,
                                     model='sonnet', requested_alias='opus',
                                     adjustment='usage step-down', identity=identity)
        command = raw.call_args.args[0]
        provenance = self._marker_line(self._normalized_and_summary(router._opus5_response(result))[1])

        self.assertEqual(command[command.index('--model') + 1], 'sonnet')
        self.assertEqual(provenance['kind'], 'claude_cli_step_down')
        self.assertEqual(provenance['requested'], 'claude-opus-5-5')
        self.assertEqual(provenance['resolved'], 'sonnet')
        self.assertEqual(provenance['actual'], {'model': 'unknown', 'provider': 'unknown'})

    def _middleware_step_down_failure(self, run_effect):
        """Real resolver -> real admission branch (Sonnet step-down) -> real public
        bridge with only subprocess/log I/O stubbed; the review then fails and the
        ordinary provider replies."""
        from model_router import claude_opus_bridge as bridge

        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'opus5': True, 'sonnet5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {'enabled': True}},
        }
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   '[opus-review] Review parser'}]}
        response = SimpleNamespace(
            model='gpt-terra', provider='openai-codex', status='completed',
            output=[SimpleNamespace(type='message', status='completed', content=[
                SimpleNamespace(type='output_text', text='ordinary verdict')])],
            output_text='ordinary verdict')
        with tempfile.TemporaryDirectory() as repo, \
             patch('model_router._load_config', return_value=cfg), \
             patch.object(bridge, '_load_config', return_value=cfg), \
             patch('model_router._verified_delegated_claude_review', return_value=(Path(repo), 'opus')), \
             patch('model_router.usage_guard.guarded', return_value=True), \
             patch('model_router.usage_guard.read', return_value=None), \
             patch('model_router.usage_guard.apply', return_value=SimpleNamespace(
                 refused='', tier='sonnet5', adjusted='opus5→sonnet5')), \
             patch('model_router.claude_delegation._log') as audit, \
             patch.object(bridge.subprocess, 'run', side_effect=run_effect) as raw, \
             patch.object(bridge, '_log_decision'), patch.object(bridge, '_append_lifecycle'):
            result = router.run_llm_with_transient_failover(
                request=request, original_request=request, next_call=Mock(return_value=response),
                provider='openai-codex', api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
        return result, raw, audit

    def test_f04_i3_i4_step_down_failure_replacement_reports_admitted_alias(self):
        timeout = subprocess.TimeoutExpired(['claude'], 600)
        payload = {'type': 'result', 'subtype': 'error_max_turns', 'is_error': True,
                   'modelUsage': {'claude-sonnet-5-5': {'inputTokens': 9}}, 'num_turns': 16}
        max_turns = lambda *args, **kwargs: SimpleNamespace(stdout=json.dumps(payload), stderr='', returncode=1)
        for name, effect in (('timeout', timeout), ('max-turn', max_turns)):
            with self.subTest(failure=name):
                result, raw, audit = self._middleware_step_down_failure(effect)
                command = raw.call_args.args[0]
                self.assertEqual(command[command.index('--model') + 1], 'sonnet')
                provenance = self._marker_line(self._normalized_and_summary(result)[1])
                self.assertEqual(provenance['kind'], 'ordinary_replacement')
                self.assertEqual(provenance['reason'], name, audit.call_args_list)
                self.assertEqual(provenance['requested'], 'claude-opus-5-5')
                self.assertEqual(provenance['resolved'], 'sonnet')
                self.assertEqual(provenance['actual']['model'], 'gpt-terra')
                audited = audit.call_args.args[1]['substitution']['identity']
                self.assertEqual(audited['resolved']['value'], 'sonnet')
                self.assertEqual(audited['effort']['applied'], 'unknown')

    # --- F04 fix round 2 (re-review N1/N2): provenance across a host continuation
    # and on refusal-only final text, through the real middleware, the real host
    # normalizer and the real child projection.
    N1_CFG = {
        'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
        'callable': {'opus5': True, 'sonnet5': True},
        'coding_agent': {'enabled': False, 'delegated_review': {'enabled': True}},
    }

    @staticmethod
    def _n1_response(kind, text='PASS: parser inspected'):
        if kind == 'reasoning':
            item = SimpleNamespace(type='reasoning', id='rs_n1', summary=[
                SimpleNamespace(type='summary_text', text='still inspecting')])
        elif kind == 'tool':
            item = SimpleNamespace(type='function_call', id='fc_n1', call_id='call_n1', name='read_file',
                                   arguments='{"path":"parser.py"}', status='completed')
        else:
            part = (SimpleNamespace(type='refusal', refusal=text) if kind == 'refusal'
                    else SimpleNamespace(type='output_text', text=text))
            item = SimpleNamespace(type='message', status='completed', role='assistant',
                                   phase='commentary' if kind == 'commentary' else None, content=[part])
        output_text = text if kind in ('final', 'commentary') else ''
        return SimpleNamespace(output=[item], output_text=output_text, model='gpt-terra',
                               provider='openai-codex', status='completed')

    def _n1_call(self, response, api_call_count, turn_id, session_id='child-n1', cfg=None):
        """One real middleware call; the CLI review fails on API call 1."""
        from model_router.claude_opus_bridge import ClaudeBridgeFailure

        if not getattr(self, '_n1_state_reset', False):
            # Module state outlives a test: start and end each test with none.
            self._n1_state_reset = True
            pending = getattr(router, '_PENDING_REPLACEMENTS', None)
            if pending is not None:
                pending.clear()
                self.addCleanup(pending.clear)

        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content': '[opus-review] Review parser'}]}
        with patch.object(router, '_load_config', return_value=cfg or self.N1_CFG), \
             patch.object(router, '_verified_delegated_claude_review', return_value=(Path('/tmp'), 'opus')), \
             patch.object(router, '_run_opus5_bridge',
                          side_effect=ClaudeBridgeFailure('Claude Code timed out', 'timeout')) as bridge, \
             patch.object(router.claude_delegation, '_log'), \
             patch.object(router.worker_admission, 'refusal', return_value=''), \
             patch.object(router.usage_guard, 'guarded', return_value=False):
            result = router.run_llm_with_transient_failover(
                request=request, original_request=request, next_call=Mock(return_value=response),
                api_call_count=api_call_count, provider='openai-codex', api_mode='codex_responses',
                platform='subagent', session_id=session_id, turn_id=turn_id)
        self.assertEqual(bridge.call_count, 1 if api_call_count == 1 else 0)
        return result

    def _codex_normalized(self, response):
        from agent.transports.codex import ResponsesApiTransport
        return ResponsesApiTransport().normalize_response(response, issuer_kind='codex_backend')

    def _projected(self, response):
        from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry

        normalized = self._codex_normalized(response)
        entry = _build_result_entry(SimpleNamespace(model='configured-child'), {
            'final_response': normalized.content, 'completed': True, 'api_calls': 2,
        }, 0, 0.1, _SchemaOutcome(None, None, [], 0))
        return normalized, entry

    def test_f04_n1_reasoning_or_commentary_interim_keeps_provenance_in_final_answer(self):
        for kind in ('reasoning', 'commentary'):
            with self.subTest(kind=kind):
                turn = f'child-n1:task:{kind}'
                first = self._codex_normalized(self._n1_call(self._n1_response(kind, 'still inspecting'), 1, turn))
                # The interim reply stays a continuation, never a marker-only answer.
                self.assertEqual((first.finish_reason, first.content), ('incomplete', ''))
                self.assertNotIn(self.PREFIX, first.reasoning or '')

                normalized, entry = self._projected(self._n1_call(self._n1_response('final'), 2, turn))
                self.assertEqual(normalized.finish_reason, 'stop')
                self.assertEqual(entry['status'], 'completed')
                self.assertTrue(entry['summary'].startswith('PASS: parser inspected'))
                marker = self._marker_line(entry['summary'])
                self.assertEqual(marker['kind'], 'ordinary_replacement')
                self.assertEqual(marker['reason'], 'timeout')
                self.assertEqual(marker['requested'], 'claude-opus-5-5')
                self.assertEqual(marker['actual']['model'], 'gpt-terra')
                self.assertFalse(marker['satisfies_cross_provider_review'])

    def test_f04_n1_tool_call_continuation_keeps_provenance_in_final_answer(self):
        turn = 'child-n1:task:tool'
        first = self._codex_normalized(self._n1_call(self._n1_response('tool'), 1, turn))
        self.assertEqual(first.finish_reason, 'tool_calls')
        self.assertEqual(len(first.tool_calls), 1)
        interim = self._codex_normalized(self._n1_call(self._n1_response('reasoning'), 2, turn))
        self.assertEqual((interim.finish_reason, interim.content), ('incomplete', ''))

        normalized, entry = self._projected(self._n1_call(self._n1_response('final'), 3, turn))
        self.assertEqual(normalized.finish_reason, 'stop')
        self.assertTrue(entry['summary'].startswith('PASS: parser inspected'))
        self.assertEqual(self._marker_line(entry['summary'])['kind'], 'ordinary_replacement')

    def test_f04_n1_retained_provenance_is_scoped_to_its_turn_and_cleared(self):
        def marked(response):
            return self.PREFIX in self._projected(response)[1]['summary']

        final = lambda: self._n1_response('final')
        self._n1_call(self._n1_response('reasoning'), 1, 'child-n1:task:turn-a')
        self.assertFalse(marked(self._n1_call(final(), 2, 'other-child:task:turn-x', session_id='other-child')),
                         'provenance leaked into another session')
        self.assertFalse(marked(self._n1_call(final(), 2, 'child-n1:task:turn-b')),
                         'provenance leaked into another turn')
        self.assertFalse(marked(self._n1_call(final(), 2, 'parent:task:turn-p', session_id='parent')),
                         'provenance leaked into the parent')
        # Compression may rotate the session id inside the turn; the turn id stays.
        self.assertTrue(marked(self._n1_call(final(), 2, 'child-n1:task:turn-a', session_id='child-n1-compressed')),
                        'final answer of the turn lost provenance')
        # The host can still continue after a terminal-looking reply (ack/stall
        # nudges); the reply that really ends the turn keeps the marker.
        self.assertTrue(marked(self._n1_call(final(), 3, 'child-n1:task:turn-a')),
                        'a nudged continuation lost provenance')
        # The turn-end hook clears the entry; nothing reaches the next turn.
        with patch.object(router, '_load_config', return_value=self.N1_CFG):
            router.on_post_llm_call(session_id='child-n1', turn_id='child-n1:task:turn-a', assistant_response='')
        self.assertEqual(len(router._PENDING_REPLACEMENTS), 0)
        self.assertFalse(marked(self._n1_call(final(), 4, 'child-n1:task:turn-a')),
                         'state was not cleared at turn end')

        # An undelivered entry (interim only) is cleared at turn end too.
        self._n1_call(self._n1_response('reasoning'), 1, 'child-n1:task:turn-c')
        with patch.object(router, '_load_config', return_value=self.N1_CFG):
            router.on_post_llm_call(session_id='child-n1', turn_id='child-n1:task:turn-c', assistant_response='')
        self.assertFalse(marked(self._n1_call(final(), 2, 'child-n1:task:turn-c')),
                         'turn end did not clear the entry')

    def test_f04_n1_retained_provenance_is_bounded_by_size_and_age(self):
        final = lambda: self._n1_response('final')
        marked = lambda response: self.PREFIX in self._projected(response)[1]['summary']
        clock = [1000.0]
        with patch.object(router, '_PENDING_REPLACEMENT_MAX', 2, create=True), \
             patch.object(router, '_PENDING_REPLACEMENT_TTL_SECONDS', 100.0, create=True), \
             patch.object(router, '_pending_replacement_clock', lambda: clock[0], create=True):
            for turn in ('bound-1', 'bound-2', 'bound-3'):
                self._n1_call(self._n1_response('reasoning'), 1, turn)
            self.assertTrue(marked(self._n1_call(final(), 2, 'bound-3')), 'newest entry missing')
            self.assertFalse(marked(self._n1_call(final(), 2, 'bound-1')), 'oldest entry was not evicted')
            self._n1_call(self._n1_response('reasoning'), 1, 'age-1')
            self._n1_call(self._n1_response('reasoning'), 1, 'age-2')
            clock[0] += 99.0
            self.assertTrue(marked(self._n1_call(final(), 2, 'age-1')), 'entry expired before its TTL')
            clock[0] += 2.0
            self.assertFalse(marked(self._n1_call(final(), 2, 'age-2')), 'entry outlived its TTL')
        self.assertLessEqual(len(getattr(router, '_PENDING_REPLACEMENTS', {})), 2)

    def test_f04_n1_final_on_host_fallback_provider_keeps_provenance(self):
        """A reasoning-only stall makes the host switch provider for the rest of the turn."""
        turn = 'child-n1:task:fallback'
        self._n1_call(self._n1_response('reasoning'), 1, turn)
        final = SimpleNamespace(output=[SimpleNamespace(type='message', status='completed', role='assistant',
                                                        content=[SimpleNamespace(type='output_text', text='PASS')])],
                                output_text='PASS', model='fallback-model', provider='other-provider',
                                status='completed')
        request = {'model': 'fallback-model', 'messages': [{'role': 'user', 'content': '[opus-review] Review parser'}]}
        with patch.object(router, '_load_config', return_value=self.N1_CFG), \
             patch.object(router.worker_admission, 'refusal', return_value=''):
            result = router.run_llm_with_transient_failover(
                request=request, original_request=request, next_call=Mock(return_value=final),
                api_call_count=4, provider='other-provider', api_mode='codex_responses',
                platform='subagent', session_id='child-n1', turn_id=turn)
        _, entry = self._projected(result)
        marker = self._marker_line(entry['summary'])
        self.assertEqual(marker['actual']['model'], 'fallback-model')
        self.assertEqual(marker['actual']['provider'], 'other-provider')
        self.assertEqual(len(router._PENDING_REPLACEMENTS), 1)  # kept until the turn ends

    def test_f04_n2_refusal_only_replacement_keeps_refusal_and_one_marker(self):
        refusal = 'I cannot review this request.'
        response = self._n1_response('refusal', refusal)
        original_part = response.output[0].content[0]
        normalized, entry = self._projected(self._n1_call(response, 1, 'child-n1:task:refusal'))

        self.assertEqual(normalized.finish_reason, 'stop')
        self.assertEqual(entry['status'], 'completed')
        self.assertTrue(entry['summary'].startswith(refusal), 'refusal text was not kept first')
        marker = self._marker_line(entry['summary'])
        self.assertEqual(marker['kind'], 'ordinary_replacement')
        self.assertEqual(marker['actual']['model'], 'gpt-terra')
        self.assertIs(response.output[0].content[0], original_part)
        self.assertEqual((original_part.type, original_part.refusal), ('refusal', refusal))

    # --- F04 fix round 3 (re-review N3/N4): the text that really ends the turn,
    # including text the execution middleware never sees, through the real host
    # finalizer seams, a real host PluginManager holding exactly the hooks
    # router.register wires, the real normalizer and the real child projection.
    # Always pass a logger to the host seams: without one they import
    # agent.conversation_loop, which pulls run_agent's dependency sync.
    HOOK_LOG = logging.getLogger('test_bridge_policy.f04_n3_n4')

    def _host_hook_manager(self):
        from hermes_cli.plugins import PluginManager

        manager = PluginManager()
        ctx = SimpleNamespace(
            register_hook=lambda name, fn: manager._hooks.setdefault(name, []).append(fn),
            register_middleware=lambda name, fn: manager._middleware.setdefault(name, []).append(fn))
        with patch.object(router.claude_delegation, 'register', return_value=True):
            router.register(ctx)
        patcher = patch('hermes_cli.plugins._delivery_manager', return_value=manager)
        patcher.start()
        self.addCleanup(patcher.stop)
        return manager

    @staticmethod
    def _child_entry(final_response, completed):
        from tools.delegate_tool_child_run import _SchemaOutcome, _build_result_entry
        return _build_result_entry(SimpleNamespace(model='configured-child'), {
            'final_response': final_response, 'completed': completed, 'failed': False, 'api_calls': 3,
        }, 0, 0.1, _SchemaOutcome(None, None, [], 0))

    def _iteration_limit_turn(self, turn, session_id):
        """Real budget fallback -> real handle_max_iterations -> real Codex summary
        builder/normalizer (only the provider result stubbed) -> real output hooks."""
        from agent import chat_completion_helpers as helpers
        from agent.transports.codex import ResponsesApiTransport
        from agent.turn_finalizer import _apply_output_hooks, _resolve_budget_fallback

        summary_call = Mock(return_value=self._n1_response('final'))
        agent = SimpleNamespace(
            max_iterations=3, iteration_budget=SimpleNamespace(remaining=0), quiet_mode=True,
            suppress_status_output=True, api_mode='codex_responses', session_id=session_id,
            model='gpt-terra', platform='subagent', _persist_disabled=False,
            _emit_diagnostic_status=Mock(), _safe_print=Mock(),
            _build_api_kwargs=Mock(return_value={'model': 'gpt-terra', 'tools': []}),
            _interruptible_api_call=summary_call, _get_transport=ResponsesApiTransport)
        agent._handle_max_iterations = lambda messages, count: helpers.handle_max_iterations(agent, messages, count)
        messages = [{'role': 'user', 'content': '[opus-review] Review parser'}]
        log = self.HOOK_LOG
        env = {key: value for key, value in os.environ.items() if key != 'HERMES_KANBAN_TASK'}
        with patch.object(helpers, '_iteration_summary_api_messages', side_effect=lambda a, m: m), \
             patch('agent.relay_llm.complete_logical_call'), \
             patch.object(router, '_load_config', return_value=self.N1_CFG), \
             patch.object(router, 'run_llm_with_transient_failover') as middleware, \
             patch.dict(os.environ, env, clear=True):
            final, reason, _ = _resolve_budget_fallback(
                agent, final_response=None, api_call_count=3, interrupted=False, failed=False,
                messages=messages, _turn_exit_reason='budget_exhausted', _pending_verification_response=None,
                _pending_verification_response_previewed=False, logger=log)
            final, transformed, pre_transform = _apply_output_hooks(
                agent, final, log, platform='subagent', effective_task_id='sa-0', turn_id=turn,
                original_user_message='[opus-review] Review parser', messages=messages)
        middleware.assert_not_called()
        self.assertEqual(summary_call.call_count, 1)
        self.assertTrue(reason.startswith('max_iterations_reached'), reason)
        return final, transformed, pre_transform

    def test_f04_n3_iteration_limit_summary_carries_one_marker_to_the_parent(self):
        self._host_hook_manager()
        for first_kind in ('reasoning', 'tool'):
            with self.subTest(first_reply=first_kind):
                turn = f'child-n3:sa-0:{first_kind}'
                # The replacement's first reply; a tool call already carries the
                # round-1 marker, which must not suppress the summary's own.
                self._n1_call(self._n1_response(first_kind), 1, turn)
                final, transformed, pre_transform = self._iteration_limit_turn(turn, 'child-n3')
                entry = self._child_entry(final, completed=False)

                self.assertEqual((entry['status'], entry['exit_reason'], entry['truncated']),
                                 ('completed', 'max_iterations', True))
                self.assertEqual(pre_transform, 'PASS: parser inspected')
                self.assertTrue(transformed)
                self.assertTrue(entry['summary'].startswith('PASS: parser inspected'))
                marker = self._marker_line(entry['summary'])
                self.assertEqual(marker['kind'], 'ordinary_replacement')
                self.assertEqual(marker['carrier'], 'final_output')
                self.assertEqual(marker['reason'], 'timeout')
                self.assertEqual(marker['requested'], 'claude-opus-5-5')
                self.assertEqual(marker['actual'], {'model': 'unknown', 'provider': 'unknown'})
                self.assertFalse(marker['satisfies_cross_provider_review'])
                # post_llm_call runs after the transform and clears the turn's entry.
                self.assertIsNone(router._pending_replacement_get(turn))

    def test_f04_n3_non_substituted_iteration_summary_is_unchanged(self):
        self._host_hook_manager()
        final, transformed, _ = self._iteration_limit_turn('child-n3:sa-0:plain', 'child-n3')
        entry = self._child_entry(final, completed=False)
        self.assertEqual(final, 'PASS: parser inspected')
        self.assertFalse(transformed)
        self.assertEqual((entry['status'], entry['exit_reason'], entry['truncated'], entry['summary']),
                         ('completed', 'max_iterations', True, 'PASS: parser inspected'))

    @staticmethod
    def _n4_response(status='completed', reason=None, phase='final_answer', item_status='completed',
                     text='PASS: parser inspected'):
        item = SimpleNamespace(type='message', status=item_status, role='assistant', phase=phase,
                               content=[SimpleNamespace(type='output_text', text=text)])
        return SimpleNamespace(output=[item], output_text=text, model='gpt-terra', provider='openai-codex',
                               status=status,
                               incomplete_details=SimpleNamespace(reason=reason) if reason else None)

    def test_f04_n4_terminal_predicate_matches_the_installed_host_normalizer(self):
        cases = {
            'final_phase_max_output_tokens': dict(status='incomplete', reason='max_output_tokens'),
            'phaseless_max_output_tokens': dict(status='incomplete', reason='max_output_tokens', phase=None),
            'final_phase_content_filter': dict(status='incomplete', reason='content_filter'),
            'queued': dict(status='queued'),
            'in_progress': dict(status='in_progress'),
            'item_incomplete': dict(item_status='incomplete'),
            'item_in_progress': dict(item_status='in_progress'),
            'commentary_only': dict(phase='commentary'),
            'completed_final_phase': dict(),
            'completed_phaseless': dict(phase=None),
        }
        expected_finish = {'final_phase_max_output_tokens': 'stop', 'final_phase_content_filter': 'content_filter',
                           'completed_final_phase': 'stop', 'completed_phaseless': 'stop'}
        for name, shape in cases.items():
            with self.subTest(case=name):
                host = self._codex_normalized(self._n4_response(**shape))
                self.assertEqual(host.finish_reason, expected_finish.get(name, 'incomplete'))
                self.assertEqual(router._provenance_shape(self._n4_response(**shape))['terminal'],
                                 host.finish_reason == 'stop' and not host.tool_calls)

    def test_f04_n4_final_phase_reply_on_incomplete_response_ends_turn_with_one_marker(self):
        from agent.turn_finalizer import apply_llm_output_transform

        self._host_hook_manager()
        turn = 'child-n4:sa-0:status'
        self._n1_call(self._n1_response('reasoning'), 1, turn)
        before = self._codex_normalized(self._n4_response(status='incomplete', reason='max_output_tokens'))
        delivered = self._n1_call(self._n4_response(status='incomplete', reason='max_output_tokens'), 2, turn)
        normalized = self._codex_normalized(delivered)
        self.assertEqual((before.finish_reason, normalized.finish_reason), ('stop', 'stop'),
                         'the host finish reason changed')
        # finish_text_response's seam: the host's once-per-turn output transform.
        agent = SimpleNamespace(session_id='child-n4', model='gpt-terra', platform='subagent')
        final, transformed, _ = apply_llm_output_transform(agent, normalized.content, turn_id=turn, logger=self.HOOK_LOG)
        self.assertFalse(transformed, 'the final-output hook marked an already marked reply')
        entry = self._child_entry(final, completed=True)
        self.assertEqual((entry['status'], entry['exit_reason']), ('completed', 'completed'))
        self.assertTrue(entry['summary'].startswith('PASS: parser inspected'))
        marker = self._marker_line(entry['summary'])
        self.assertEqual(marker['actual']['model'], 'gpt-terra')
        self.assertEqual(marker['reason'], 'timeout')

    def test_f04_n4_unmarked_final_text_of_a_substituted_turn_gets_one_marker_at_delivery(self):
        """Backstop: a final reply the middleware did not mark (any shape the
        terminal mirror misjudges) still reaches the parent with one marker."""
        from agent.turn_finalizer import apply_llm_output_transform

        self._host_hook_manager()
        turn = 'child-n4:sa-0:backstop'
        self._n1_call(self._n1_response('reasoning'), 1, turn)
        agent = SimpleNamespace(session_id='child-n4', model='gpt-terra', platform='subagent')
        final, transformed, raw = apply_llm_output_transform(agent, 'PASS: parser inspected', turn_id=turn, logger=self.HOOK_LOG)
        self.assertTrue(transformed)
        self.assertEqual(raw, 'PASS: parser inspected')
        self.assertEqual(self._marker_line(self._child_entry(final, completed=True)['summary'])['kind'],
                         'ordinary_replacement')
        # Another turn, and the same text with no retained entry, stay unchanged.
        other = SimpleNamespace(session_id='other', model='gpt-terra', platform='subagent')
        text = 'PASS: parser inspected'
        unchanged, transformed, _ = apply_llm_output_transform(other, text, turn_id='other:sa-1:turn',
                                                                logger=self.HOOK_LOG)
        self.assertIs(unchanged, text)
        self.assertFalse(transformed)

    # --- F04 fix round 4 (re-review N5): the child summary returned to the
    # parent is budgeted after child projection. At the host's 2,000-character
    # floor, both a middleware carrier and the final-output backstop carrier
    # must leave exactly one complete compact marker in the returned tail,
    # whether or not the full summary spill succeeds.
    def _parent_at_summary_floor(self):
        return SimpleNamespace(
            context_compressor=SimpleNamespace(context_length=200000, max_tokens=4000),
            _last_prompt_size_tokens=199000,
        )

    def _finalize_parent_summary(self, entry, *, spill_fails):
        from tools import delegate_tool_results as results

        patches = [
            patch('tools.delegate_tool._load_config', return_value={'max_summary_chars': 24000}),
            patch.object(results, '_notify_memory_manager'),
            patch.object(results, '_fire_subagent_stop_hooks', return_value=0),
        ]
        if spill_fails:
            patches.append(patch.object(results, '_spill_summary_to_file', return_value=None))
        with patches[0], patches[1], patches[2]:
            if spill_fails:
                with patches[3]:
                    results._finalize_child_results([entry], [{'goal': 'review parser'}], [],
                                                     self._parent_at_summary_floor())
            else:
                results._finalize_child_results([entry], [{'goal': 'review parser'}], [],
                                                 self._parent_at_summary_floor())
        return entry

    def test_f04_n5_compact_marker_is_hard_bounded_at_worst_case_field_lengths(self):
        huge = 'x' * 4096
        marker = router._substitution_provenance(
            huge,
            identity={
                'requested': {'value': huge, 'source': huge},
                'resolved': {'value': huge, 'source': huge},
            },
            substitution={huge: huge},
            replacement={
                'policy': huge, 'reason': huge, 'failure_kind': huge,
                'requested_review': {'tier': huge, 'transport': huge},
                'planned': {'tier': huge, 'model': huge, 'provider': huge},
            },
            executed={'model': huge, 'provider': huge},
        )
        line = router._provenance_annotation(marker)
        bound = getattr(router, '_PARENT_VISIBLE_PROVENANCE_MAX_CHARS', 400)
        value_bound = getattr(router, '_PARENT_PROVENANCE_VALUE_MAX_CHARS', 32)
        self.assertEqual(bound, 400)
        self.assertLessEqual(len(line), bound)
        parsed = json.loads(line[len(self.PREFIX):])
        self.assertEqual(parsed['v'], 1)
        self.assertEqual(parsed['requested'], 'x' * value_bound)
        self.assertEqual(parsed['resolved'], 'x' * value_bound)
        self.assertEqual(parsed['actual']['model'], 'x' * value_bound)
        self.assertFalse(parsed['satisfies_cross_provider_review'])

    def test_f04_n5_middleware_marker_survives_parent_budget_with_or_without_spill(self):
        for spill_fails in (False, True):
            with self.subTest(spill_fails=spill_fails):
                turn = f'child-n5:middleware:{spill_fails}'
                response = self._n1_response('final', 'PASS: parser inspected\n' + ('Evidence line.\n' * 400))
                normalized = self._codex_normalized(self._n1_call(response, 1, turn))
                entry = self._child_entry(normalized.content, completed=True)
                self._finalize_parent_summary(entry, spill_fails=spill_fails)

                self.assertEqual((entry['status'], entry['exit_reason']), ('completed', 'completed'))
                self.assertTrue(entry['summary_truncated'])
                marker = self._marker_line(entry['summary'])
                self.assertEqual(marker['carrier'], 'middleware')
                self.assertEqual(marker['kind'], 'ordinary_replacement')
                self.assertFalse(marker['satisfies_cross_provider_review'])
                if spill_fails:
                    self.assertNotIn('summary_full_path', entry)
                else:
                    self.assertIn('summary_full_path', entry)

    def test_f04_n5_final_output_marker_survives_parent_budget_with_or_without_spill(self):
        from agent.turn_finalizer import apply_llm_output_transform

        self._host_hook_manager()
        for spill_fails in (False, True):
            with self.subTest(spill_fails=spill_fails):
                turn = f'child-n5:final-output:{spill_fails}'
                self._n1_call(self._n1_response('reasoning', 'still inspecting'), 1, turn)
                final, changed, _ = apply_llm_output_transform(
                    SimpleNamespace(session_id='child-n5', model='gpt-terra', platform='subagent'),
                    'PASS: parser inspected\n' + ('Evidence line.\n' * 400),
                    turn_id=turn, logger=self.HOOK_LOG,
                )
                self.assertTrue(changed)
                entry = self._child_entry(final, completed=True)
                self._finalize_parent_summary(entry, spill_fails=spill_fails)

                self.assertEqual((entry['status'], entry['exit_reason']), ('completed', 'completed'))
                self.assertTrue(entry['summary_truncated'])
                marker = self._marker_line(entry['summary'])
                self.assertEqual(marker['carrier'], 'final_output')
                self.assertEqual(marker['actual'], {'model': 'unknown', 'provider': 'unknown'})
                self.assertFalse(marker['satisfies_cross_provider_review'])
                if spill_fails:
                    self.assertNotIn('summary_full_path', entry)
                else:
                    self.assertIn('summary_full_path', entry)

    # Audit H01: remove when F07 lands
    @unittest.expectedFailure
    def test_h01_installed_host_runner_supports_retry_callback(self):
        """H01 is inherited (existed at 70125ce)."""
        from hermes_cli.middleware import run_llm_execution_middleware

        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'sonnet5': True, 'opus5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {'enabled': True}},
        }
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   '[sonnet-review] Review parser'}]}
        manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                  _report_hook_failure=Mock())
        downstream = Mock(side_effect=[RuntimeError('temporary provider failure'), SimpleNamespace(model='gpt-spark')])
        with patch('hermes_cli.plugins._delivery_manager', return_value=manager), \
             patch.object(router, '_load_config', return_value=cfg), \
             patch.object(router, '_maybe_run_opus5', return_value=None), \
             patch.object(router.worker_admission, 'refusal', return_value=''), \
             patch.object(router, '_is_transient_provider_failure', return_value=True), \
             patch.object(router, '_record_tier_failure'), \
             patch.object(router, '_log_decision'), \
             patch.object(router, '_transient_fallback_model', return_value='gpt-spark'):
            try:
                run_llm_execution_middleware(request, downstream, provider='openai-codex',
                    api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
            except Exception:
                pass
        self.assertEqual(downstream.call_count, 2, 'installed host runner did not supply a reusable retry callback')

    def test_exact_cli_route_without_capability_evidence_refuses_without_ordinary_provider_fallback(self):
        from hermes_cli.middleware import run_llm_execution_middleware

        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'opus5': True, 'sonnet5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {
                'enabled': True, 'selection_mode': 'exact',
                'requested_model': 'claude-sonnet-5-5',
            }},
        }
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   '[sonnet-review] Review parser'}]}
        downstream = Mock(return_value='Codex ran')
        manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                  _report_hook_failure=Mock())
        with patch('hermes_cli.plugins._delivery_manager', return_value=manager), \
             patch('model_router._load_config', return_value=cfg), \
             patch('model_router._verified_delegated_claude_review', return_value=(Path('/tmp'), 'sonnet')), \
             patch('model_router._run_opus5_bridge') as bridge, \
             patch('model_router.claude_delegation._log') as audit:
            stopped = run_llm_execution_middleware(request, downstream, provider='openai-codex',
                api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
        bridge.assert_not_called()
        downstream.assert_not_called()
        self.assertIn('ROUTER WORKER STOPPED', stopped.output_text)
        self.assertIn('exact canonical CLI selection needs exact_model evidence', stopped.output_text)
        self.assertEqual(audit.call_args.args[1]['outcome'], 'refused')
        self.assertEqual(audit.call_args.args[1]['tier_used'], 'none')
        self.assertIn('exact canonical CLI selection needs exact_model evidence', audit.call_args.args[1]['message'])

    def test_exact_cli_route_refuses_a_usage_step_down_without_ordinary_provider_fallback(self):
        from hermes_cli.middleware import run_llm_execution_middleware
        from types import SimpleNamespace
        from model_router.execution_contracts import ModelFact, TargetIdentity
        from model_router.target_identity import RESOLVED
        from model_router.claude_opus_bridge import ClaudeBridgeFailure

        identity = TargetIdentity(
            provider='anthropic', account='unknown', transport='claude_cli', alias='opus', selection_mode='exact',
            requested=ModelFact('claude-opus-5-5', 'fixture'),
        )
        cfg = {
            'enabled': True, 'provider': 'openai-codex', 'models': {'terra': 'gpt-terra'},
            'callable': {'opus5': True, 'sonnet5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {
                'enabled': True, 'selection_mode': 'exact', 'requested_model': 'claude-opus-5-5',
            }},
        }
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   '[opus-review] Review parser'}]}
        downstream = Mock(return_value='Codex ran')
        manager = SimpleNamespace(_middleware={'llm_execution': [router.run_llm_with_transient_failover]},
                                  _report_hook_failure=Mock())
        with patch('hermes_cli.plugins._delivery_manager', return_value=manager), \
             patch('model_router._load_config', return_value=cfg), \
             patch('model_router._verified_delegated_claude_review', return_value=(Path('/tmp'), 'opus')), \
             patch('model_router.target_identity.resolve_target', return_value=SimpleNamespace(
                 status=RESOLVED, identity=identity, reasons=())), \
             patch('model_router.usage_guard.guarded', return_value=True), \
             patch('model_router.usage_guard.apply', return_value=SimpleNamespace(
                 refused='', tier='sonnet5', adjusted='opus5→sonnet5')), \
             patch('model_router._run_opus5_bridge') as bridge:
            stopped = run_llm_execution_middleware(request, downstream, provider='openai-codex',
                api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
        bridge.assert_not_called()
        self.assertIn('ROUTER WORKER STOPPED', stopped.output_text)
        self.assertIn('cannot substitute', stopped.output_text)
        downstream.assert_not_called()

    def test_failed_review_records_actual_replacement_result_after_transient_failover(self):
        from types import SimpleNamespace
        from claude_opus_bridge import ClaudeBridgeFailure

        cfg = {
            'enabled': True, 'provider': 'openai-codex',
            'models': {'terra': 'gpt-terra', 'spark': 'gpt-spark'},
            'callable': {'opus5': True, 'sonnet5': True},
            'coding_agent': {'enabled': False, 'delegated_review': {'enabled': True}},
        }
        request = {'model': 'gpt-terra', 'messages': [{'role': 'user', 'content':
                   '[sonnet-review] Review parser'}]}
        response = SimpleNamespace(
            model='gpt-spark', provider='openai-codex',
            output=[SimpleNamespace(type='message', status='completed', content=[
                SimpleNamespace(type='output_text', text='retry verdict'),
            ])],
            output_text='retry verdict', status='completed',
        )
        downstream = Mock(side_effect=RuntimeError('temporary provider failure'))
        retry = Mock(return_value=response)
        failure = ClaudeBridgeFailure('Claude Code reached max turns', 'max-turn')
        with patch('model_router._load_config', return_value=cfg), \
             patch('model_router._verified_delegated_claude_review', return_value=(Path('/tmp'), 'sonnet')), \
             patch('model_router._run_opus5_bridge', side_effect=failure), \
             patch('model_router._is_transient_provider_failure', return_value=True), \
             patch('model_router._transient_fallback_model', return_value='gpt-spark'):
            result = router.run_llm_with_transient_failover(
                request=request, original_request=request, next_call=downstream, retry_call=retry,
                provider='openai-codex', api_mode='codex_responses', platform='subagent', turn_id='s:sa-1')
        self.assertIs(result, response)
        self.assertEqual(result.replacement_provenance['planned']['model'], 'gpt-terra')
        self.assertEqual(result.replacement_provenance['executed']['model'], 'gpt-spark')
        self.assertEqual(result.replacement_provenance['executed']['model_source'], 'ordinary_provider.response.model')
        self.assertEqual(result.replacement_provenance['executed']['provider'], 'openai-codex')
        self.assertIn('actual replacement executed on gpt-spark', result.route_reason)
        provenance = self._parent_visible_provenance(result)
        self.assertEqual(provenance['actual']['model'], 'gpt-spark')
        self.assertEqual(provenance['actual']['provider'], 'openai-codex')
        retry.assert_called_once()

    def test_failed_review_bridge_keeps_one_route_call_and_records_visible_failure_audit(self):
        with patch('model_router._log_decision') as route_log, \
             patch('model_router.claude_delegation._log') as audit:
            result, codex, bridge, _ = self._route(
                codex=10, claude=10, bridge_error=RuntimeError('Claude Code reached max turns'))
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
        bridge.assert_called_once()
        # The request middleware already logged the sole provider call. A bridge
        # failure is a correlated Claude audit, never a second routed-call entry.
        route_log.assert_not_called()
        audit.assert_called_once()
        event = audit.call_args.args[1]
        self.assertEqual(event['outcome'], 'error')
        self.assertEqual(event['tier_requested'], 'sonnet')
        self.assertEqual(event['failure_kind'], 'max-turn')
        self.assertEqual(event['substitution']['tier'], 'terra')
        self.assertEqual(event['substitution']['model'], 'gpt-terra')
        self.assertEqual(event['turn_id'], 's:sa-1')
        self.assertIn('Claude Code reached max turns', event['message'])
        self.assertIn('CLI review did not complete', codex.call_args.args[0]['messages'][-1]['content'])

    def test_typed_bridge_failure_preserves_identity_and_failure_class_in_audit(self):
        from model_router.claude_opus_bridge import ClaudeBridgeFailure

        identity = {"requested": "claude-sonnet-5-5", "observed": "claude-opus-5-5"}
        failure = ClaudeBridgeFailure("served a different model", "model-mismatch", identity=identity)
        with patch('model_router.claude_delegation._log') as audit:
            result, codex, bridge, _ = self._route(codex=10, claude=10, bridge_error=failure)
        self.assertEqual(result, 'Codex ran')
        bridge.assert_called_once()
        codex.assert_called_once()
        event = audit.call_args.args[1]
        self.assertEqual(event['failure_kind'], 'model-mismatch')
        self.assertEqual(event['failure_class'], 'exact-route-mismatch')
        self.assertEqual(event['tier_used'], 'sonnet')
        self.assertEqual(event['substitution']['planned']['model'], 'gpt-terra')

    def test_pre_bridge_review_exception_is_not_audited_as_a_claude_failure(self):
        with patch('model_router.claude_delegation._log') as audit:
            result, codex, bridge, _ = self._route(
                codex=10, claude=10, pre_bridge_error=RuntimeError('bad review config'))
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
        bridge.assert_not_called()
        audit.assert_not_called()

    def test_root_review_label_without_a_bridge_attempt_is_not_audited(self):
        with patch('model_router.claude_delegation._log') as audit:
            result, codex, bridge, _ = self._route(codex=10, claude=10, worker=False)
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
        bridge.assert_not_called()
        audit.assert_not_called()

    def test_no_eligible_bridge_does_not_probe_claude(self):
        result, codex, bridge, read = self._route(codex=10, claude=10, eligible=False)
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
        bridge.assert_not_called()
        read.assert_not_called()

    def test_failed_review_persists_one_routed_call_and_one_correlated_audit(self):
        import json
        from unittest.mock import Mock

        with tempfile.TemporaryDirectory() as directory:
            routes = Path(directory) / 'routes.jsonl'
            audits = Path(directory) / 'claude.jsonl'
            cfg = {
                'enabled': True, 'provider': 'openai-codex',
                'models': {'terra': 'gpt-terra'},
                'callable': {'terra': True, 'sonnet5': True},
                'coding_agent': {'enabled': False, 'delegated_review': {'enabled': True}},
                'logging': {'enabled': True, 'path': str(routes)},
                'claude_delegation': {'log_path': str(audits)},
            }
            request = {'model': 'gpt-terra', 'messages': [
                {'role': 'user', 'content': '[sonnet-review] Review parser'}]}
            context = {'turn_id': 's:sa-1', 'api_call_count': 1, 'request': request}
            # Request middleware already wrote the one actual routed-call entry.
            router._log_decision(router.RouteDecision('terra', 'gpt-terra', 'review route'), context, cfg)
            downstream = Mock(return_value='Codex ran')
            with patch('model_router._load_config', return_value=cfg), \
                 patch('model_router._verified_delegated_claude_review', return_value=(Path(directory), 'sonnet')), \
                 patch('model_router._run_opus5_bridge', side_effect=RuntimeError('Claude Code reached max turns')):
                result = router.run_llm_with_transient_failover(
                    request=request, original_request=request, next_call=downstream,
                    provider='openai-codex', api_mode='codex_responses',
                    platform='subagent', turn_id='s:sa-1', api_call_count=1)
            route_entries = [json.loads(line) for line in routes.read_text().splitlines()]
            audit_entries = [json.loads(line) for line in audits.read_text().splitlines()]
        self.assertEqual(result, 'Codex ran')
        downstream.assert_called_once()
        self.assertEqual(len(route_entries), 1)
        self.assertEqual((route_entries[0]['turn_id'], route_entries[0]['api_call_count']), ('s:sa-1', 1))
        self.assertEqual(len(audit_entries), 1)
        self.assertEqual(audit_entries[0]['turn_id'], 's:sa-1')
        self.assertEqual(audit_entries[0]['outcome'], 'error')
        self.assertIn('Claude Code reached max turns', audit_entries[0]['message'])

    def test_root_is_not_stopped_by_worker_account_limit(self):
        result, codex, _, _ = self._route(codex=95, claude=95, worker=False)
        self.assertEqual(result, 'Codex ran')
        codex.assert_called_once()
