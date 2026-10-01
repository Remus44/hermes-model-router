import json
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
            'unknown capability': dict(patches=(
                patch('model_router.claude_opus_bridge.CLAUDE_REVIEW_MODELS', {}),)),
        }
        for name, options in cases.items():
            with self.subTest(refusal=name), tempfile.TemporaryDirectory() as directory:
                run = self._exact_preselection_run(directory, **options)
                self._assert_stopped_refusal(run)
                self.assertEqual(run[5][0]['outcome'], 'refused')

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
            with patch.object(router, '_log_decision') as route_log:
                stopped, downstream, bridge = self._host_review_attempt(
                    selection_mode='exact', log_path=audits, routes_path=routes,
                    bridge_result=ClaudeBridgeFailure('wrong model', 'model-mismatch'))
            audit_entries = [json.loads(line) for line in audits.read_text().splitlines()]
        downstream.assert_not_called()
        bridge.assert_called_once()
        route_log.assert_not_called()
        self.assertFalse(routes.exists())
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

    # Audit A02: remove when F04 lands
    @unittest.expectedFailure
    def test_a02_successful_cli_conversion_preserves_identity_and_substitution(self):
        result = router._opus5_response({
            'result': 'review verdict', 'effective_model': 'claude-sonnet-5-5',
            'identity': {'requested': 'claude-opus-5-5', 'observed': 'claude-sonnet-5-5'},
            'substitution': {'policy': 'profile_preferred', 'requested': 'claude-opus-5-5',
                             'observed': 'claude-sonnet-5-5'},
        })
        self.assertIsNotNone(getattr(result, 'substitution', None),
            'successful bridge response discarded substitution')

    # Audit A02: remove when F04 lands
    @unittest.expectedFailure
    def test_a02_replacement_provenance_survives_host_normalization(self):
        from agent.transports.codex import ResponsesApiTransport

        result = router._opus5_response({'result': 'PASS', 'effective_model': 'claude-sonnet-5-5'})
        result = router._record_ordinary_replacement_result(result, {
            'policy': 'legacy_ordinary_provider_route', 'requested_review': {'tier': 'opus'},
            'failure_kind': 'model-mismatch', 'planned': {'model': 'gpt-terra'},
        })
        normalized = ResponsesApiTransport().normalize_response(result)
        self.assertTrue(getattr(normalized, 'replacement_provenance', None) or
            (normalized.provider_data or {}).get('replacement_provenance'),
            'host normalization dropped replacement provenance')

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
        response = SimpleNamespace(model='gpt-spark', provider='openai-codex')
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
