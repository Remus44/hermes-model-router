import contextlib
import io
import json
import os
import tempfile
import threading
import time
import subprocess
import unittest
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from pathlib import Path

from unittest.mock import patch

import web_viewer
from web_viewer import HTML, Handler


class DashboardProbeMixin:
    """Helpers that slice the shipped dashboard source so probes stay honest.

    Separate from the test classes so a suite can reuse them without inheriting —
    and re-running — every test the other class declares.
    """

    def execution_source(self):
        start = HTML.index("/* Reference execution-tree renderer:")
        end = HTML.index("function executionState", start)
        return HTML[start:end]

    def i18n(self, key):
        """Return the (english, hungarian) pair declared for one i18n key.

        Structural assertions anchor on the key; this checks the copy itself,
        so a translation change can never silently break an unrelated test.
        """
        import re

        english = HTML[HTML.index("  en: {"):HTML.index("  hu: {")]
        hungarian = HTML[HTML.index("  hu: {"):HTML.index("\n};", HTML.index("  hu: {"))]
        pattern = r"^\s*'%s':\s*'((?:[^'\\]|\\.)*)'" % re.escape(key)
        found = []
        for block, language in ((english, "en"), (hungarian, "hu")):
            match = re.search(pattern, block, re.M)
            self.assertIsNotNone(match, f"i18n key {key!r} missing from the {language} dictionary")
            found.append(match.group(1))
        return tuple(found)

    def i18n_runtime(self, language="en"):
        """The dashboard's real I18N dictionary and t(), ready to run in node.

        The execution-tree renderer calls t(), so a probe without it dies with
        a ReferenceError. Slicing the live source keeps the probes honest —
        a renamed key fails here rather than silently falling back.
        """
        start = HTML.index("const I18N = {")
        end = HTML.index("function applyLanguage()")
        runtime = HTML[start:end].replace("localStorage.getItem('model-router-lang')", "null")
        return runtime.replace("let currentLang = null || 'en';", f"let currentLang = {language!r};")

    def javascript_function(self, name):
        start = HTML.index(f"function {name}")
        brace = HTML.index("{", start)
        depth = 0
        for index in range(brace, len(HTML)):
            if HTML[index] == "{":
                depth += 1
            elif HTML[index] == "}":
                depth -= 1
                if depth == 0:
                    return HTML[start:index + 1]
        self.fail(f"unterminated JavaScript function: {name}")

class ModelRouterDashboardTests(DashboardProbeMixin, unittest.TestCase):
    def test_execution_tree_counts_are_labeled_as_routing_decisions(self):
        self.assertIn('<div class="cards"><div class="card"><div class="n" id="total">0</div>', HTML)
        self.assertIn('<div class="k" data-i18n="card.total">', HTML)
        self.assertNotIn('.cards{display:none}', HTML)
        renderer = HTML[HTML.rindex("function render(){"):]
        self.assertIn("total.innerHTML=`<b>${t('total.routing.decisions')}</b>`", renderer)
        self.assertIn("workers.textContent=`${workerCalls} ${t('run.worker.routing')}`", renderer)
        self.assertEqual(self.i18n('total.routing.decisions'), ('TOTAL ROUTING DECISIONS', 'ÖSSZES ROUTING DÖNTÉS'))
        self.assertNotIn('ÖSSZES HÍVÁS', renderer)
        self.assertNotIn('worker-hívás', renderer)

    def test_the_two_claude_tiers_are_styled_the_same_way(self):
        """Duplicating a rule for a new tier is easy to get wrong: dropping the
        selector prefix turns `.task-tree-marker.opus5{background:…}` into a bare
        `.sonnet5{background:…}`, which then paints the whole summary card in the
        tier colour instead of a four-pixel marker."""
        import re

        css = "".join(re.findall(r"<style>(.*?)</style>", HTML, re.S))
        selectors = {
            tier: sorted(
                match.group(1).replace(tier, "<tier>")
                for match in re.finditer(r"([^{};]*\.%s[^{};]*)\{" % tier, css)
            )
            for tier in ("opus5", "sonnet5")
        }
        self.assertTrue(selectors["opus5"], "expected opus5 to carry tier styling")
        self.assertEqual(selectors["opus5"], selectors["sonnet5"])

    def test_final_router_renderer_refreshes_each_summary_counter(self):
        """Asserted against the cards themselves rather than a copied literal:
        adding a tier used to mean editing four separate lists, and a counter
        left out of one of them renders as a card frozen at zero."""
        import re

        renderer = HTML[HTML.rindex("function render(){"):]
        card_ids = re.findall(r'<div class="n" id="([a-z0-9]+)">', HTML)
        loop = re.search(r"for\(const id of \[([^\]]+)\]\)\$\(id\)\.textContent=summary\[id\]", renderer)
        self.assertIsNotNone(loop)
        refreshed = [name.strip("'") for name in loop.group(1).split(",")]
        # 'total' has its own line; every other card must be in the loop or it
        # renders frozen at zero, which is what the Qwen card did.
        self.assertEqual(sorted(refreshed), sorted(set(card_ids) - {"total"}))
        self.assertIn("$('total').textContent=summary.total", renderer)

    def test_recent_selector_is_a_root_prompt_limit(self):
        select_start = HTML.index('<select id="last">')
        label_start = HTML.rfind('<label>', 0, select_start)
        selector = HTML[label_start:HTML.index('</label>', select_start)]
        self.assertEqual(
            selector,
            '<label><span data-i18n="router.last.label">Utolsó root promptok</span>'
            '<select id="last"><option>1</option><option>5</option><option selected>10</option></select>',
        )
        self.assertEqual(self.i18n('router.last.label'), ('Last root prompts', 'Utolsó root promptok'))
        self.assertIn("fetch(`/api/entries?roots=${$('last').value}`", HTML)
        self.assertIn(
            "`${limitedRoots.length} ${t('status.rootprompts')} · "
            "${new Date().toLocaleTimeString(t('status.locale'))}`",
            HTML,
        )

    def test_root_limit_keeps_latest_roots_and_all_of_their_raw_records(self):
        source = "\n".join(self.javascript_function(name) for name in (
            "limitRootRuns", "rawEntryKey", "rawEntriesForRootRuns"
        ))
        roots = [
            {"id": "old", "rawEntries": [{"id": "old-root"}] + [{"id": f"old-child-{index}"} for index in range(230)]},
            {"id": "middle", "rawEntries": [{"id": "middle-root"}]},
            {"id": "latest", "rawEntries": [{"id": "latest-root"}, {"id": "latest-lifecycle"}]},
        ]
        all_entries = roots[0]["rawEntries"] + roots[1]["rawEntries"] + roots[2]["rawEntries"]
        probe = (
            source + "\nconst roots=" + json.dumps(roots) + ";"
            "const allEntries=" + json.dumps(all_entries) + ";"
            "const limited=limitRootRuns(roots,2);"
            "console.log(JSON.stringify({roots:limited.map(run=>run.id),raw:rawEntriesForRootRuns(limited,allEntries).map(entry=>entry.id)}));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout), {
            "roots": ["middle", "latest"],
            "raw": ["middle-root", "latest-root", "latest-lifecycle"],
        })

    def test_more_than_200_newer_lifecycle_records_do_not_hide_the_root_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "router.jsonl"
            records = [{
                "timestamp": "2026-08-03T00:00:00+00:00",
                "turn_id": "root-session:root-turn",
                "prompt_preview": "Visible root",
                "tier": "sol",
            }]
            records.extend({
                "timestamp": f"2026-08-03T00:{index // 60:02d}:{index % 60:02d}+00:00",
                "turn_id": f"root-session:sa-{index}",
                "parent_turn_id": "root-session:root-turn",
                "prompt_preview": f"[ASYNC DELEGATION COMPLETE — lifecycle-{index}]",
                "is_internal_prompt": True,
                "tier": "sol",
            } for index in range(1, 251))
            log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            original_path = Handler.log_path
            Handler.log_path = log_path
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(
                    f"http://127.0.0.1:{server.server_port}/api/entries?roots=5"
                ) as response:
                    payload = json.load(response)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)
                Handler.log_path = original_path

            source = "let agentActivity={parents:[]};const syntheticLifecyclePrefixes=[];\n" + "\n".join(
                self.javascript_function(name) for name in (
                    "childSessionIds",
                    "turnBelongsToChild",
                    "isSyntheticLifecyclePrompt",
                    "visibleEntries",
                    "sessionIdFromTurn",
                    "systemEntriesForRoot",
                    "promptKey",
                    "promptGroups",
                    "limitRootRuns",
                )
            )
            probe = (
                source + "\nconst entries=" + json.dumps(payload["entries"]) + ";"
                "const roots=limitRootRuns(promptGroups(visibleEntries(entries)),5);"
                "console.log(JSON.stringify({prompts:roots.map(group=>group[0].prompt_preview),"
                "lifecycle:systemEntriesForRoot(roots[0][0],entries).length}));"
            )
            result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)

            self.assertEqual(len(payload["entries"]), 251)
            self.assertEqual(payload["requested_root_limit"], 5)
            self.assertEqual(payload["raw_history_limit"], 10000)
            self.assertEqual(json.loads(result.stdout), {
                "prompts": ["Visible root"],
                "lifecycle": 250,
            })

    def test_only_top_summary_cards_use_expanded_model_family_labels(self):
        cards = HTML[HTML.index('<div class="cards">'):HTML.index('<div id="runs"')]
        for key, label in [('card.luna', 'GPT-6 Luna'), ('card.spark', 'GPT-5.3 Spark'),
                           ('card.terra', 'GPT-5.6 Terra'), ('card.sol', 'GPT-6 Sol'),
                           ('card.opus5', 'Claude Opus 5.5')]:
            self.assertIn(f'<div class="k" data-i18n="{key}">{label}</div>', cards)
            self.assertEqual(self.i18n(key), (label, label))
        # Since Task 7 the static #tier select only carries the "all" option;
        # the per-tier <option>s come from tierFilterOptions(), grouped by
        # account (see AccountGroupTests).
        self.assertIn("return effort?`${tier} · ${effort}`:tier", HTML)
        self.assertIn("String(raw?.tier||raw?.model||node.model||kind).toUpperCase()", HTML)
        router_run_markup = HTML[HTML.index('<div id="runs"'):HTML.index('<section id="settings-panel"')]
        self.assertNotIn('GPT-6 Sol', router_run_markup)
        self.assertIn('<div class="card sol">', cards)
        self.assertIn('.pill.sol{color:var(--sol)}', HTML)

    def test_workers_remain_inside_router_prompt_rows(self):
        self.assertNotIn('data-tab="agents"', HTML)
        self.assertIn("function parentForGroup(group)", HTML)
        self.assertIn("function childEntries(child,list,seen=new Set())", HTML)

    def test_root_prompt_disclosure_is_independent_of_word_wrap(self):
        self.assertIn(
            '<label class="check"><input id="word-wrap" type="checkbox"> '
            '<span data-i18n="router.wordwrap">Sortörés</span></label>',
            HTML,
        )
        self.assertIn("hasDetails=scope.nodes.length>0", HTML)
        self.assertIn("if(hasDetails&&open)", HTML)
        self.assertIn(".word-wrap .router-run-prompt,.word-wrap .execution-tree-description", HTML)
        self.assertNotIn("detailsEnabled=$('word-wrap').checked", HTML)

    def test_empty_main_run_has_no_disclosure_or_detail_panel(self):
        renderer = HTML[HTML.rindex("function render(){"):]
        self.assertIn("const header=document.createElement(hasDetails?'button':'div')", renderer)
        self.assertIn("router-run-header ${hasDetails?'':'no-details'}", renderer)
        self.assertIn("if(hasDetails){const toggle=document.createElement('span')", renderer)
        self.assertIn("if(hasDetails&&open)", renderer)

    def test_expanded_tree_has_no_inner_rounded_or_dark_panel_border(self):
        self.assertIn(".execution-tree-panel{margin:0;overflow:auto;background:transparent;border:0;border-radius:0}", HTML)
        self.assertNotIn(".execution-tree-panel{margin:16px 22px 22px;border:1px", HTML)

    def test_root_origin_and_nested_child_connector_contract(self):
        self.assertIn(".execution-tree-row:before{content:'';position:absolute;left:calc(34px + var(--tree-depth) * 38px);top:0;bottom:50%;border-left:1px solid #50637d}", HTML)
        self.assertIn(".execution-tree-row.task-tree-has-next-sibling:before{bottom:0}", HTML)
        self.assertIn(".execution-tree-children:before", HTML)
        self.assertIn("childrenEl.style.setProperty('--tree-depth',depth)", self.execution_source())
        self.assertIn("tree.append(childrenEl)", self.execution_source())

    def test_only_real_internal_node_has_one_clickable_disclosure(self):
        source = self.execution_source()
        self.assertIn("const disclosure=document.createElement(hasChildren?'button':'span')", source)
        self.assertIn("disclosure.className=hasChildren?'execution-tree-toggle':'execution-tree-spacer'", source)
        self.assertIn("disclosure.textContent=hasChildren?(open?'▾':'▸'):''", source)
        self.assertIn("event.stopPropagation();setTaskTreeOpen(node.id,!open);render()", source)
        self.assertNotIn("border-top:18px solid #c49bff", HTML)
        self.assertIn(".task-tree-marker.terra{background:#c49bff", HTML)

    def test_internal_disclosure_matches_root_and_leaf_has_no_toggle(self):
        source = self.execution_source()
        self.assertIn(".router-run-toggle,.execution-tree-toggle{color:#c4b5fd;font-size:27px;line-height:1}", HTML)
        self.assertIn(".execution-tree-toggle,.execution-tree-spacer{position:relative;z-index:1;width:24px;min-height:28px", HTML)
        self.assertIn("const disclosure=document.createElement(hasChildren?'button':'span')", source)
        self.assertIn("hasChildren?'execution-tree-toggle':'execution-tree-spacer'", source)

    def test_lifecycle_description_uses_own_prompt_or_short_safe_fallback(self):
        source = self.execution_source()
        self.assertIn("function lifecycleDescription(entry)", source)
        self.assertIn("return t('internal.router.step')", source)
        self.assertIn("task_description:lifecycleDescription(entry)", source)
        probe = self.i18n_runtime("hu") + source + "\nconst own={lifecycle_prompt:'Saját tárolt lifecycle feladat',prompt_preview:'[ASYNC DELEGATION COMPLETE — dump]'};const empty={prompt_preview:'[ASYNC DELEGATION COMPLETE — dump]'};console.log(JSON.stringify([lifecycleDescription(own),lifecycleDescription(empty)]));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout), ["Saját tárolt lifecycle feladat", "Belső router-lépés"])

    def test_lifecycle_description_prefers_human_completion_provenance(self):
        source = self.execution_source()
        probe = self.i18n_runtime() + source + "\nconst completion={event_kind:'async_delegation_completion',lifecycle_prompt:'Delegált feladat befejezési eseménye · Rövid redaktált feladat'};console.log(lifecycleDescription(completion));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(result.stdout.strip(), "Delegált feladat befejezési eseménye · Rövid redaktált feladat")

    def test_recursive_raw_accounting_makes_terra_include_spark_96(self):
        source = self.execution_source()
        self.assertIn("function executionCalls(node){return [...executionOwnCalls(node),...(node.children||[]).flatMap(executionCalls)]}", source)
        self.assertIn("const ownCalls=executionOwnCount(node),totalCalls=executionTotalCalls(node)", source)
        self.assertIn("appendRoutePills(routes,executionCalls(node))", source)
        self.assertIn("function executionScope(group,systemCalls,parent,tier='')", source)
        self.assertIn("accountingCalls=scope.calls", HTML)
        fixture = {"routed_calls": [{"tier": "terra", "effort": "medium"}] * 32,
                   "children": [{"routed_calls": [{"tier": "spark", "effort": "medium"}] * 48, "children": []},
                                {"routed_calls": [{"tier": "spark", "effort": "medium"}] * 48, "children": []}]}
        probe = source + "\nconst fixture=" + json.dumps(fixture) + ";console.log(JSON.stringify({total:executionTotalCalls(fixture),routes:executionCalls(fixture).map(executionRouteKey)}));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["total"], 128)
        self.assertEqual(observed["routes"].count("terra · medium"), 32)
        self.assertEqual(observed["routes"].count("spark · medium"), 96)

    def test_pricing_copy_lifecycle_replays_count_once(self):
        """Eight completion calls for one delegation must not inflate the total."""
        source = self.execution_source()
        group = [{"tier": "terra", "api_call_count": 1}, {"tier": "terra", "api_call_count": 2}]
        system = [{
            "event_kind": "async_delegation_completion",
            "delegation_id": "deleg_pricing",
            "turn_id": "parent:completion",
            "tier": "terra",
            "prompt_preview": "Delegált feladat befejezési eseménye",
            "api_call_count": index,
        } for index in range(1, 9)]
        parent = {"session_id": "parent", "children": [
            {"id": "conductor", "routed_calls": [{"tier": "terra"}] * 40, "children": []},
            {"id": "recon", "routed_calls": [{"tier": "sonnet5"}] * 10, "children": []},
            {"id": "worker", "routed_calls": [{"tier": "grok"}] * 40, "children": []},
        ]}
        probe = (
            self.i18n_runtime() + source
            + "\nconst scope=executionScope(" + json.dumps(group) + ","
            + json.dumps(system) + "," + json.dumps(parent) + ");"
            + "const lifecycle=scope.nodes.filter(node=>node.kind==='LIFECYCLE').flatMap(executionCalls);"
            + "console.log(JSON.stringify({lifecycle:lifecycle.length,total:scope.calls.length}));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["lifecycle"], 1)
        self.assertEqual(observed["total"], 93)

    def test_scope_model_filter_reaches_nested_workers_and_prunes_other_calls(self):
        source = self.execution_source()
        group = [{"tier": "terra", "effort": "medium"}] * 4
        system = [{"tier": "sol", "effort": "medium"}] * 2
        parent = {"session_id": "parent", "children": [{
            "id": "supervisor", "routed_calls": [{"tier": "sol", "effort": "medium"}] * 29,
            "children": [
                {"id": "sol-leaf", "routed_calls": [{"tier": "sol", "effort": "medium"}] * 10, "children": []},
                {"id": "mixed-leaf", "routed_calls": ([{"tier": "spark", "effort": "medium"}] * 5 +
                                                          [{"tier": "terra", "effort": "medium"}] * 5), "children": []},
            ],
        }]}
        probe = self.i18n_runtime() + source + "\nconst scope=executionScope(" + json.dumps(group) + "," + json.dumps(system) + "," + json.dumps(parent) + ",'sol');const workerCalls=scope.nodes.filter(node=>node.kind!=='LIFECYCLE').flatMap(executionCalls);console.log(JSON.stringify({calls:scope.calls.map(executionRouteKey),workerCalls:workerCalls.length,tree:scope.nodes}));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(len(observed["calls"]), 41)
        self.assertEqual(observed["workerCalls"], 39)
        self.assertEqual(set(observed["calls"]), {"sol · medium"})
        self.assertEqual(len(observed["tree"]), 3)
        stack = list(observed["tree"])
        routed_tiers = set()
        while stack:
            node = stack.pop()
            routed_tiers.update(call["tier"] for call in node.get("routed_calls", []))
            stack.extend(node.get("children", []))
        self.assertEqual(routed_tiers, {"sol"})

    def test_summary_uses_unique_calls_represented_by_visible_request_scopes(self):
        source = self.execution_source()
        runs = [
            ([{"tier": "terra"}] * 27 + [{"tier": "sol"}] * 3 + [{"tier": "spark"}] * 5),
            ([{"tier": "terra"}] * 9 + [{"tier": "sol"}] * 31 + [{"tier": "spark"}] * 5),
            [{"tier": "terra"}],
            [{"tier": "terra"}],
        ]
        probe = source + "\nconst summary=executionSummary(" + json.dumps(runs) + ");console.log(JSON.stringify(summary));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(
            json.loads(result.stdout),
            {"total": 82, "luna": 0, "spark": 10, "terra": 38, "sol": 34,
             "opus5": 0, "sonnet5": 0, "haiku": 0, "qwen": 0, "grok": 0},
        )
        renderer = HTML[HTML.rindex("function render(){"):]
        self.assertIn("const summary=executionSummary(runData.map(run=>run.scope.calls))", renderer)
        self.assertNotIn("$('total').textContent=allEntries.length", renderer)

    def test_spark_effort_pill_and_raw_route_precedence(self):
        source = self.execution_source()
        self.assertIn("const tier=String(call?.tier||call?.model||'?').toLowerCase(),effort=String(call?.effort||'').toLowerCase()", source)
        self.assertIn("return effort?`${tier} · ${effort}`:tier", source)
        self.assertIn("function executionOwnCalls(node){const raw=Array.isArray(node?.routed_calls)?node.routed_calls:[];return raw.length?raw:[]}", source)
        self.assertIn("String(raw?.tier||raw?.model||node.model||kind).toUpperCase()", source)

    def test_opus5_is_enumerated_colored_and_filterable_in_grouped_and_raw_views(self):
        self.assertIn("--opus5:#d695ff", HTML)
        # Since Task 7 the static #tier select only carries the "all" option;
        # the per-tier <option>s come from tierFilterOptions() (see
        # AccountGroupTests.test_tier_filter_options_are_grouped_by_account_label).
        self.assertIn('class="card opus5"', HTML)
        self.assertIn(".pill.opus5{color:var(--opus5)}", HTML)
        self.assertIn(".task-tree-marker.opus5{background:var(--opus5)", HTML)
        source = self.execution_source()
        self.assertIn("source.includes('opus5')||source.includes('claude-opus-5')", source)
        self.assertIn("opus5:0", source)
        parent = {"children": [{"id": "external", "model": "claude-opus-5-5", "routed_calls": [{"tier": "opus5", "model": "claude-opus-5-5", "effort": "external"}], "children": []}]}
        probe = source + "\nfunction sessionIdFromTurn(entry){return String(entry?.turn_id||'').split(':')[0]}\nconst scope=executionScope([{tier:'terra'}],[]," + json.dumps(parent) + ",'opus5');console.log(JSON.stringify({calls:scope.calls,kinds:scope.nodes.map(executionKind),summary:executionSummary([scope.calls])}));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["calls"][0]["tier"], "opus5")
        self.assertEqual(observed["kinds"], ["opus5"])
        self.assertEqual(observed["summary"]["opus5"], 1)

    def test_only_exact_prompt_group_in_same_session_inherits_running_parent(self):
        source = "\n".join(self.javascript_function(name) for name in (
            "sessionIdFromTurn", "promptsMatch", "parentForGroup", "executionState"
        ))
        # executionState returns a sentinel, not a label: its value used to be
        # the rendered Hungarian text, so translating the UI broke the branch.
        activity = {"parents": [{
            "session_id": "shared-session",
            "prompt": "Current live prompt",
            "children": [{"state": "running"}],
        }], "active_turns": []}
        old_group = [{"turn_id": "shared-session:old-turn", "prompt_preview": "Historical prompt"}]
        live_group = [{"turn_id": "shared-session:live-turn", "prompt_preview": "Current live prompt"}]
        probe = (
            "const agentActivity=" + json.dumps(activity) + ";\n" + source +
            "\nconst groups=" + json.dumps([old_group, live_group]) + ";" +
            "console.log(JSON.stringify(groups.map(group=>executionState(group,parentForGroup(group)))));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertEqual(json.loads(result.stdout), ["done", "running"])

    def test_last_sibling_connector_does_not_continue_below_leaf(self):
        self.assertIn(".execution-tree-row:before{content:'';position:absolute;left:calc(34px + var(--tree-depth) * 38px);top:0;bottom:50%;border-left:1px solid #50637d}", HTML)
        self.assertIn(".execution-tree-row.task-tree-has-next-sibling:before{bottom:0}", HTML)

    def test_ten_roots_return_a_narrow_server_side_closure(self):
        with tempfile.TemporaryDirectory() as directory:
            log_path = Path(directory) / "router.jsonl"
            records = []
            for root_index in range(30):
                root_turn = f"session-{root_index}:turn"
                records.append({"timestamp": f"2026-08-03T00:{root_index:02d}:00+00:00", "turn_id": root_turn,
                                "prompt_preview": f"root-{root_index}", "tier": "sol"})
                records.extend({"timestamp": f"2026-08-03T00:{root_index:02d}:{child:02d}+00:00",
                                "turn_id": f"session-{root_index}:sa-{child}", "parent_turn_id": root_turn,
                                "prompt_preview": f"[ASYNC DELEGATION COMPLETE {child}]", "is_internal_prompt": True,
                                "tier": "sol"} for child in range(1, 6))
            log_path.write_text("\n".join(json.dumps(record) for record in records), encoding="utf-8")
            original_path = Handler.log_path
            Handler.log_path = log_path
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{server.server_port}/api/entries?roots=10") as response:
                    payload = json.load(response)
            finally:
                server.shutdown(); server.server_close(); thread.join(timeout=2); Handler.log_path = original_path
        roots = [entry for entry in payload["entries"] if not entry.get("is_internal_prompt")]
        self.assertEqual([entry["prompt_preview"] for entry in roots], [f"root-{i}" for i in range(20, 30)])
        self.assertEqual(len(payload["entries"]), 60)
        self.assertEqual(payload["selected_root_count"], 10)
        self.assertEqual(payload["source_entry_count"], 180)

    def test_refresh_cycle_fetches_entries_and_agents_once_and_renders_once(self):
        self.assertIn("Promise.all([fetch(`/api/entries?roots=${$('last').value}`", HTML)
        self.assertIn("fetch('/api/agents'", HTML)
        cycle = self.javascript_function("refreshDashboard")
        self.assertEqual(cycle.count("render()"), 1)
        self.assertIn("agentActivity=activity", cycle)
        self.assertNotIn("renderAgents(", cycle)
        self.assertNotIn("async function load()", HTML)
        self.assertNotIn("async function loadAgents()", HTML)

    def test_polling_has_backpressure_and_keeps_old_rows_visible_while_refreshing(self):
        cycle = self.javascript_function("refreshDashboard")
        self.assertIn("if(refreshInFlight)return refreshInFlight", cycle)
        self.assertIn("finally", cycle)
        self.assertIn("refreshInFlight=null", cycle)
        self.assertNotIn("replaceChildren", cycle)
        self.assertIn("button.disabled=true", cycle)
        self.assertIn("button.textContent=t('status.refresh.short')", cycle)
        self.assertEqual(self.i18n('status.refresh.short'), ('Refresh…', 'Frissítés…'))
        self.assertIn("setInterval(()=>{if($('auto').checked)refreshDashboard()},3000)", HTML)

    def test_a_client_disconnect_does_not_escape_the_response_writer(self):
        from unittest.mock import MagicMock

        for error in (BrokenPipeError(32, "broken pipe"), ConnectionResetError(104, "reset")):
            with self.subTest(error=type(error).__name__):
                handler = object.__new__(Handler)
                handler.send_response = MagicMock()
                handler.send_header = MagicMock()
                handler.end_headers = MagicMock()
                handler.wfile = MagicMock()
                handler.wfile.write.side_effect = error

                handler._send(200, b"response", "application/json")

    def test_external_opus_child_card_has_badges_metrics_states_and_stable_run_disclosure(self):
        source = self.execution_source()
        self.assertIn("node.external", source)
        self.assertIn("badges.push(t('exec.external'))", source)
        self.assertIn("badges.push(t('exec.read.only'))", source)
        self.assertIn("badges.push(t('agents.requested.ro'))", source)
        self.assertEqual(self.i18n('exec.external'), ('EXTERNAL', 'KÜLSŐ'))
        self.assertEqual(self.i18n('exec.read.only'), ('READ-ONLY', 'READ-ONLY'))
        self.assertEqual(self.i18n('agents.requested.ro'), ('Requested READ-ONLY', 'KÉRT READ-ONLY'))
        self.assertIn("input_tokens", source)
        self.assertIn("cache_read_input_tokens", source)
        self.assertIn("total_cost_usd", source)
        self.assertIn("bridge_run_id", source)
        self.assertIn("running:t('state.running.short')", source)
        self.assertEqual(self.i18n('state.running.short'), ('RUNNING', 'FUT'))
        self.assertIn("'max-turn':'MAX-TURN'", source)
        self.assertIn("external_bridge_run_ids", HTML)
        self.assertIn("bridge_run_id", HTML)


class DashboardLanguageTests(unittest.TestCase):
    """The dictionary existing is not the same as the renderers using it.

    Every string the renderers emit used to be a Hungarian literal, so the
    language selector only ever translated the static shell.
    """

    def render_probe(self, language):
        helper = ModelRouterDashboardTests("test_lifecycle_description_uses_own_prompt_or_short_safe_fallback")
        source = helper.execution_source()
        probe = helper.i18n_runtime(language) + source + (
            "\nconsole.log(JSON.stringify({"
            "fallback:lifecycleDescription({}),"
            "external:t('exec.external'),"
            "running:t('state.running.short'),"
            "missing:t('no.such.key')"
            "}));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    def test_renderer_output_follows_the_selected_language(self):
        english, hungarian = self.render_probe("en"), self.render_probe("hu")
        self.assertEqual(english["fallback"], "Internal router step")
        self.assertEqual(hungarian["fallback"], "Belső router-lépés")
        self.assertEqual(english["external"], "EXTERNAL")
        self.assertEqual(hungarian["external"], "KÜLSŐ")
        self.assertEqual(english["running"], "RUNNING")
        self.assertEqual(hungarian["running"], "FUT")

    def test_unknown_key_degrades_to_the_key_itself(self):
        # A missing key must never render as blank — it has to stay findable.
        self.assertEqual(self.render_probe("hu")["missing"], "no.such.key")

    def test_switching_language_rerenders_the_dynamic_content(self):
        # applyLanguage only walks [data-i18n] nodes; the tables and agent
        # cards are built by the renderers and need an explicit repaint.
        handler = HTML[HTML.index("document.getElementById('language-select').addEventListener"):]
        handler = handler[:handler.index("});")]
        self.assertIn("applyLanguage()", handler)
        self.assertIn("render()", handler)
        self.assertIn("renderAgents(agentActivity)", handler)


if __name__ == "__main__":
    unittest.main()


class RouterModuleImportTests(unittest.TestCase):
    """A normal Hermes install has no ``model_router`` package anywhere on
    sys.path: the repo is ``hermes-model-router`` and the plugin directory is
    ``model-router`` (hyphenated). _router_module() must still find the router
    by loading it from its own script directory."""

    def test_router_module_imports_without_a_parent_named_model_router(self):
        import subprocess
        import sys
        import tempfile

        script_dir = str(Path(web_viewer.__file__).resolve().parent)
        probe = (
            "import importlib.util, sys\n"
            f"sys.path.insert(0, {script_dir!r})\n"
            f"spec = importlib.util.spec_from_file_location('web_viewer', {str(Path(web_viewer.__file__).resolve())!r})\n"
            "web_viewer = importlib.util.module_from_spec(spec)\n"
            "spec.loader.exec_module(web_viewer)\n"
            "router = web_viewer._router_module()\n"
            "assert router is not None, 'expected the router to import'\n"
            "assert hasattr(router, 'usage_guard'), 'expected the router module to carry usage_guard'\n"
            "print('OK')\n"
        )
        with tempfile.TemporaryDirectory() as cwd:
            result = subprocess.run(
                [sys.executable, "-c", probe],
                cwd=cwd,
                env={"HOME": cwd, "PATH": os.environ.get("PATH", "")},
                check=True, text=True, capture_output=True,
            )
        self.assertEqual(result.stdout.strip(), "OK")


class RouterStatusTests(unittest.TestCase):
    def test_router_status_degrades_to_empty_instead_of_failing(self):
        """The dashboard is a standalone script; it must keep serving the log
        even when the router package truly cannot be imported. _router_module()
        now has its own importlib fallback (see RouterModuleImportTests), so this
        drives the degrade path through that seam directly rather than via the
        old sys.modules trick, which the fallback would simply route around."""
        with patch.object(web_viewer, "_router_module", return_value=None):
            status = web_viewer._router_status()
        self.assertEqual(
            status, {"cooldowns": {}, "load": {}, "window_minutes": 0, "routable": []}
        )

    def test_only_a_routable_tier_is_offered_as_orchestrator(self):
        """A delegation-only target has no model entry in the router, so picking
        it raises a KeyError on the first routing decision. The card and the
        callability switch still apply to it -- only the orchestrator role does
        not."""
        status = web_viewer._router_status()
        self.assertNotIn("opus5", status["routable"])
        self.assertNotIn("sonnet5", status["routable"])
        self.assertIn("terra", status["routable"])
        renderer = HTML[HTML.index("function renderMainChain("):]
        self.assertIn("currentConfig.routable", renderer[:renderer.index('id="default-model-select"')])

    def test_router_status_reports_cooling_tiers_and_account_load(self):
        """Read through the router's own helpers rather than recomputed here, so
        the panel and the routing decision cannot disagree about availability."""
        status = web_viewer._router_status()
        self.assertIn("cooldowns", status)
        self.assertIn("load", status)
        self.assertIsInstance(status["cooldowns"], dict)
        self.assertIsInstance(status["load"], dict)
        for entry in status["cooldowns"].values():
            self.assertIn("seconds", entry)
            self.assertIn("reason", entry)

    def test_every_tier_the_router_knows_can_report_a_cooldown(self):
        """sonnet5 was missing from a hardcoded tuple here, so a cooling Sonnet
        reported nothing and the dashboard showed that account as merely idle —
        the one reading the operator most needs when Claude is the spare."""
        from unittest.mock import patch

        config = {
            "models": {"luna": "m", "terra": "m", "sol": "m", "qwen": "m"},
            "callable": {"luna": True, "terra": True, "sol": True,
                         "opus5": True, "sonnet5": True, "qwen": True},
            "usage_report": {"window_seconds": 3600},
        }
        with patch("model_router._load_config", return_value=config), \
             patch("model_router._read_cooldown_state",
                   return_value={"tiers": {"sonnet5": {"reason": "quota exhausted"}}}), \
             patch("model_router._tier_cooldown_remaining",
                   side_effect=lambda tier, cfg: 600.0 if tier == "sonnet5" else 0.0), \
             patch("model_router._recent_account_load", return_value={}):
            status = web_viewer._router_status()

        self.assertIn("sonnet5", status["cooldowns"])
        self.assertEqual(status["cooldowns"]["sonnet5"]["reason"], "quota exhausted")


class PreferenceSettingsTests(DashboardProbeMixin, unittest.TestCase):
    """The per-work-kind chain editor. The reordering logic runs in node, not in
    a Python re-implementation, so a bug in the shipped source fails here."""

    def _mutate(self, prefs, kind, index, act):
        """Run the real mutatePreference against a stub DOM and return the new prefs."""
        source = self.javascript_function("mutatePreference")
        probe = (
            "let currentConfig=" + json.dumps({"preferences": prefs}) + ";"
            "function renderSettings(){};function saveSettings(){};"
            + source
            + f"\nmutatePreference({json.dumps(kind)},{index},{json.dumps(act)});"
            "console.log(JSON.stringify(currentConfig.preferences));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    def test_moving_an_entry_up_reorders_the_chain(self):
        self.assertEqual(
            self._mutate({"design": ["sol", "opus5", "terra"]}, "design", 1, "up"),
            {"design": ["opus5", "sol", "terra"]},
        )

    def test_moving_the_first_entry_up_is_a_no_op(self):
        self.assertEqual(
            self._mutate({"design": ["sol", "opus5"]}, "design", 0, "up"),
            {"design": ["sol", "opus5"]},
        )

    def test_moving_the_last_entry_down_is_a_no_op(self):
        self.assertEqual(
            self._mutate({"design": ["sol", "opus5"]}, "design", 1, "down"),
            {"design": ["sol", "opus5"]},
        )

    def test_removing_the_last_entry_drops_the_kind_entirely(self):
        """An empty list is not "no preference" to the router — the absent key is."""
        self.assertEqual(self._mutate({"design": ["sol"]}, "design", 0, "del"), {})

    def test_removing_one_of_several_keeps_the_rest_in_order(self):
        self.assertEqual(
            self._mutate({"code": ["sol", "terra", "luna"]}, "code", 1, "del"),
            {"code": ["sol", "luna"]},
        )

    def _render(self, config, language="en"):
        source = self.javascript_function("chainChipAccount") + "\n" + self.javascript_function("renderPreferences")
        labels = {m: m.upper() for m in
                  ("luna", "spark", "terra", "sol", "opus5", "sonnet5", "qwen")}
        probe = (
            self.i18n_runtime(language)
            + "let html='';const box={set innerHTML(v){html=v;},appendChild(el){html+=el.outerHTML||el.textContent;},"
            "classList:{toggle(){}}};"
            "function $(id){return id==='pref-kinds'?box:null;}"
            "let currentConfig=" + json.dumps(config) + ";"
            "document={createElement:()=>({className:'',set innerHTML(v){this._h=v;},"
            "get outerHTML(){return this._h||'';},textContent:''})};"
            + source
            + f"\nrenderPreferences({json.dumps(labels)});console.log(html);"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return result.stdout

    def _chain_chip_account(self, model, config):
        source = self.javascript_function("chainChipAccount")
        probe = self.i18n_runtime() + "let currentConfig=" + json.dumps(config) + ";" + source \
            + f"\nconsole.log(JSON.stringify(chainChipAccount({json.dumps(model)})));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return json.loads(result.stdout)

    def test_chain_chip_account_carries_class_label_and_state_mark(self):
        config = {
            "tier_accounts": {"opus5": "anthropic"},
            "accounts": {"anthropic": {"label": "Claude", "state": "soft"}},
        }
        self.assertEqual(
            self._chain_chip_account("opus5", config),
            {"account": "anthropic", "label": "Claude", "mark": "soft limit"},
        )

    def test_chain_chip_account_has_no_mark_when_the_account_is_open(self):
        config = {
            "tier_accounts": {"terra": "openai-codex"},
            "accounts": {"openai-codex": {"label": "Codex", "state": "open"}},
        }
        self.assertEqual(
            self._chain_chip_account("terra", config),
            {"account": "openai-codex", "label": "Codex", "mark": ""},
        )

    def test_chain_chip_account_is_null_for_a_model_with_no_known_account(self):
        self.assertIsNone(self._chain_chip_account("terra", {"tier_accounts": {}, "accounts": {}}))

    def test_preference_chips_carry_the_account_class_label_and_closed_mark(self):
        html = self._render({
            "work_kinds": ["design"], "preferences": {"design": ["opus5"]},
            "routable": ["opus5"], "callable": {"opus5": True},
            "tier_accounts": {"opus5": "anthropic"},
            "accounts": {"anthropic": {"label": "Claude", "state": "closed"}},
        })
        chip_start = html.index('<span class="pref-chip')
        chip_end = html.index('</span>', html.rindex('data-act="del"', chip_start))
        chip = html[chip_start:chip_end]
        self.assertIn(' anthropic', chip.split('>')[0])
        self.assertIn('Claude', chip)
        english, _ = self.i18n('account.state.closed')
        self.assertIn(english, chip)

    def test_a_delegation_only_target_is_marked_apart_from_a_routed_one(self):
        """A purple chip means "handed to the conductor", not "routed here"."""
        html = self._render({
            "work_kinds": ["design"], "preferences": {"design": ["opus5", "sol"]},
            "routable": ["luna", "spark", "terra", "sol"],
            "callable": {"opus5": True, "sol": True},
        })
        self.assertIn("pref-chip external", html)
        self.assertIn("1.", html)

    def test_a_kind_without_a_preference_says_the_built_in_route_applies(self):
        html = self._render({
            "work_kinds": ["review"], "preferences": {},
            "routable": ["terra"], "callable": {"terra": True},
        })
        self.assertIn("pref-empty", html)

    def test_a_switched_off_model_is_not_offered(self):
        """Offering it would let you configure a chain entry that can never run."""
        html = self._render({
            "work_kinds": ["code"], "preferences": {},
            "routable": ["terra", "sol"], "callable": {"terra": True, "sol": False, "qwen": False},
        })
        self.assertIn('value="terra"', html)
        self.assertNotIn('value="sol"', html)

    def test_every_work_kind_has_a_label_in_both_languages(self):
        from model_router import WORK_KINDS

        for kind in WORK_KINDS:
            english, hungarian = self.i18n(f"kind.{kind}")
            self.assertTrue(english.strip(), kind)
            self.assertTrue(hungarian.strip(), kind)
            self.assertNotEqual(english, hungarian, f"kind.{kind} is untranslated")


class HermesFallbackChainTests(DashboardProbeMixin, unittest.TestCase):
    """The chains live in Hermes's config, not the router's — so the editor has to be
    careful with a file it does not own, and the UI has to say which file it writes."""

    OPTIONS = [
        {"key": "opus5", "provider": "anthropic", "model": "claude-opus-5-5"},
        {"key": "sonnet5", "provider": "anthropic", "model": "claude-sonnet-5-5"},
        {"key": "qwen", "provider": "qwen-token", "model": "qwen3.7-plus"},
    ]

    def test_a_route_outside_the_configured_targets_is_rejected(self):
        """Offering a route the installation lacks would configure a leaf that cannot run."""
        chain, error = web_viewer._clean_fallback_chain(
            [{"provider": "evil", "model": "x"}], self.OPTIONS)
        self.assertIsNone(chain)
        self.assertIn("Unknown route", error)

    def test_an_incomplete_entry_is_rejected(self):
        chain, error = web_viewer._clean_fallback_chain([{"provider": "anthropic"}], self.OPTIONS)
        self.assertIsNone(chain)
        self.assertIn("needs a provider and a model", error)

    def test_duplicates_collapse_and_order_is_kept(self):
        chain, error = web_viewer._clean_fallback_chain([
            {"provider": "qwen-token", "model": "qwen3.7-plus"},
            {"provider": "anthropic", "model": "claude-opus-5-5"},
            {"provider": "qwen-token", "model": "qwen3.7-plus"},
        ], self.OPTIONS)
        self.assertIsNone(error)
        self.assertEqual([e["model"] for e in chain], ["qwen3.7-plus", "claude-opus-5-5"])

    def test_an_empty_chain_is_preserved_not_dropped(self):
        """For a delegated worker [] means "no fallback" — not "inherit the parent's"."""
        chain, error = web_viewer._clean_fallback_chain([], self.OPTIONS)
        self.assertIsNone(error)
        self.assertEqual(chain, [])

    def test_an_absent_chain_is_left_alone(self):
        self.assertEqual(web_viewer._clean_fallback_chain(None, self.OPTIONS), (None, None))

    def test_a_write_keeps_a_restore_point(self):
        """This file carries providers, approvals and the command allowlist."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text("model:\n  default: gpt-5.6-terra\n", encoding="utf-8")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                web_viewer._write_hermes_config({"model": {"default": "changed"}})
            backups = list(Path(directory).glob("config.yaml.bak-router-*"))
            self.assertEqual(len(backups), 1)
            self.assertIn("gpt-5.6-terra", backups[0].read_text(encoding="utf-8"))
            self.assertIn("changed", target.read_text(encoding="utf-8"))

    def test_saving_one_chain_does_not_clear_the_other(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text(
                "delegation:\n  fallback_providers:\n  - provider: anthropic\n    model: claude-opus-5-5\n",
                encoding="utf-8")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target), \
                 patch.object(web_viewer, "_fallback_chain_options", return_value=self.OPTIONS):
                error = web_viewer._save_hermes_fallback(
                    {"orchestrator": [{"provider": "qwen-token", "model": "qwen3.7-plus"}]}, {})
            self.assertIsNone(error)
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["delegation"]["fallback_providers"][0]["model"], "claude-opus-5-5")
            self.assertEqual(written["fallback_providers"][0]["model"], "qwen3.7-plus")

    def test_the_primary_is_dropped_from_its_own_fallback_chain(self):
        """The page shows the main agent as one chain whose first entry is the
        model Hermes starts on; that model repeated as a fallback can never help."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text("model:\n  default: gpt-5.6-terra\n  provider: openai-codex\n", encoding="utf-8")
            options = self.OPTIONS + [{"key": "terra", "provider": "openai-codex", "model": "gpt-5.6-terra"}]
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target), \
                 patch.object(web_viewer, "_fallback_chain_options", return_value=options):
                error = web_viewer._save_hermes_fallback({"orchestrator": [
                    {"provider": "openai-codex", "model": "gpt-5.6-terra"},
                    {"provider": "qwen-token", "model": "qwen3.7-plus"}]}, {})
            self.assertIsNone(error)
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["fallback_providers"], [{"provider": "qwen-token", "model": "qwen3.7-plus"}])

    def test_the_worker_model_writes_the_hermes_delegation_block(self):
        import tempfile

        router_cfg = {"models": {"terra": "gpt-5.6-terra", "sol": "gpt-5.6-sol", "qwen": "qwen3.7-plus"},
                      "tier_providers": {"terra": "openai-codex", "sol": "openai-codex", "qwen": "qwen-token"}}
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text("delegation:\n  model: gpt-5.6-terra\n  provider: openai-codex\n"
                              "  max_iterations: 40\n", encoding="utf-8")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                self.assertIsNone(web_viewer._save_worker_model("sol", router_cfg))
                self.assertIn("cannot be", web_viewer._save_worker_model("qwen", router_cfg))
                status = web_viewer._worker_model_status(router_cfg)
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["delegation"]["model"], "gpt-5.6-sol")
            self.assertEqual(written["delegation"]["max_iterations"], 40, "the rest of the block survives")
            self.assertEqual(status["tier"], "sol")
            self.assertEqual(status["options"], ["terra", "sol"])

    def test_a_switched_off_route_is_dropped_from_both_chains(self):
        """Hermes fails over without consulting the router's switches, so Qwen
        switched off in the dashboard was still a live fallback (and a 403)."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text("model:\n  default: gpt-5.6-terra\n  provider: openai-codex\n", encoding="utf-8")
            options = [{"key": "qwen", "provider": "qwen-token", "model": "qwen3.7-plus"},
                       {"key": "sonnet5", "provider": "anthropic", "model": "claude-sonnet-5-5"}]
            qwen = {"provider": "qwen-token", "model": "qwen3.7-plus"}
            sonnet = {"provider": "anthropic", "model": "claude-sonnet-5-5"}
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target), \
                 patch.object(web_viewer, "_fallback_chain_options", return_value=options):
                error = web_viewer._save_hermes_fallback(
                    {"orchestrator": [sonnet, qwen], "children": [qwen]}, {"callable": {"qwen": False}})
            self.assertIsNone(error)
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["fallback_providers"], [sonnet])
            self.assertEqual(written["delegation"]["fallback_providers"], [])

    def test_an_unreadable_config_is_never_overwritten(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "absent.yaml"
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", missing), \
                 patch.object(web_viewer, "_fallback_chain_options", return_value=self.OPTIONS):
                error = web_viewer._save_hermes_fallback(
                    {"orchestrator": [{"provider": "qwen-token", "model": "qwen3.7-plus"}]}, {})
            self.assertIn("could not be read", error)
            self.assertFalse(missing.exists())

    def test_the_section_names_the_file_it_writes_in_both_languages(self):
        english, hungarian = self.i18n("settings.fb.file")
        for text in (english, hungarian):
            self.assertIn("~/.hermes/config.yaml", text)

    def test_options_include_every_router_model_and_a_claude_tier(self):
        """Restricting the picker to Hermes delegation targets alone rejected the
        operator's own orchestrator chain, which named this router's own account
        (openai-codex/gpt-6-sol) -- never a Hermes delegation target in the
        first place."""
        router_cfg = {
            "models": {"terra": "gpt-5.6-terra", "sol": "gpt-6-sol"},
            "tier_providers": {"terra": "openai-codex", "sol": "openai-codex"},
            "claude_delegation": {"tiers": {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}},
        }
        with patch.object(web_viewer, "_read_hermes_config", return_value={}):
            options = web_viewer._fallback_chain_options(router_cfg)
        self.assertIn({"key": "sol", "provider": "openai-codex", "model": "gpt-6-sol"}, options)
        self.assertIn({"key": "sonnet5", "provider": "anthropic", "model": "claude-sonnet-5-5"}, options)

    def test_options_still_include_the_hermes_delegation_targets(self):
        router_cfg = {"models": {}, "tier_providers": {}, "claude_delegation": {"tiers": {}}}
        with patch.object(web_viewer, "_read_hermes_config",
                          return_value={"delegation": {"targets": {
                              "qwen": {"provider": "qwen-token", "model": "qwen3.7-plus"}}}}):
            options = web_viewer._fallback_chain_options(router_cfg)
        self.assertIn({"key": "qwen", "provider": "qwen-token", "model": "qwen3.7-plus"}, options)

    def test_a_route_already_in_the_saved_chain_can_never_fail_a_save(self):
        """Measured live 2026-09-18: every settings save 400'd and reverted every
        toggle, because the operator's real orchestrator chain named a route
        (openai-codex/gpt-6-sol) the old picker never offered. Whatever is
        already saved must always be re-acceptable, even if the picker's own
        options do not (any longer, or yet) include it."""
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text(
                "fallback_providers:\n- provider: openai-codex\n  model: gpt-6-sol\n"
                "delegation:\n  targets:\n"
                "    opus5:\n      provider: anthropic\n      model: claude-opus-5-5\n"
                "    sonnet5:\n      provider: anthropic\n      model: claude-sonnet-5-5\n",
                encoding="utf-8")
            router_cfg = {"models": {}, "tier_providers": {}, "claude_delegation": {"tiers": {}}}
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_hermes_fallback(
                    {"orchestrator": [{"provider": "openai-codex", "model": "gpt-6-sol"}]}, router_cfg)
            self.assertIsNone(error)
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["fallback_providers"][0]["model"], "gpt-6-sol")

    def test_an_unknown_route_outside_the_saved_chain_is_still_rejected(self):
        import tempfile

        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text(
                "fallback_providers:\n- provider: openai-codex\n  model: gpt-5.6-sol\n"
                "delegation:\n  targets: {}\n",
                encoding="utf-8")
            router_cfg = {"models": {}, "tier_providers": {}, "claude_delegation": {"tiers": {}}}
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_hermes_fallback(
                    {"orchestrator": [{"provider": "evil", "model": "x"}]}, router_cfg)
            self.assertIsNotNone(error)
            self.assertIn("Unknown route", error)

    def test_the_operators_live_payload_saves_instead_of_reverting(self):
        """End-to-end reproduction of the live failure: a real POST /api/config
        with the operator's exact settings-page payload must now return 200 and
        actually write the callable change, instead of 400ing and leaving the
        page to reload the old config over every toggle."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "router_config.yaml"
            router_cfg = {
                "callable": {"terra": True, "sol": True},
                "default_model": "terra",
                "models": {"terra": "gpt-5.6-terra", "sol": "gpt-6-sol"},
                "tier_providers": {"terra": "openai-codex", "sol": "openai-codex"},
                "preferences": {},
                "claude_delegation": {"tiers": {"sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}},
            }
            with open(config_path, "w", encoding="utf-8") as f:
                web_viewer.yaml.dump(router_cfg, f)

            hermes_path = Path(directory) / "hermes-config.yaml"
            hermes_path.write_text(
                "fallback_providers:\n- provider: openai-codex\n  model: gpt-6-sol\n"
                "delegation:\n  targets:\n"
                "    opus5:\n      provider: anthropic\n      model: claude-opus-5-5\n"
                "    sonnet5:\n      provider: anthropic\n      model: claude-sonnet-5-5\n",
                encoding="utf-8")

            payload = json.dumps({
                "callable": {"terra": True, "sol": False},
                "default_model": "terra",
                "preferences": {},
                "hermes_fallback": {"orchestrator": [{"provider": "openai-codex", "model": "gpt-6-sol"}]},
                "usage_limits": {},
            }).encode("utf-8")

            with patch.object(web_viewer, "CONFIG_PATH", config_path), \
                 patch.object(web_viewer, "HERMES_CONFIG_PATH", hermes_path):
                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/config",
                        data=payload, method="POST",
                        headers={"Content-Type": "application/json"},
                    )
                    with urllib.request.urlopen(request) as response:
                        status = response.status
                        body = json.load(response)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)

            self.assertEqual(status, 200)
            self.assertTrue(body.get("success"))
            # Saves land in the git-ignored local file, over the shipped one.
            local = config_path.with_name("router_config.local.yaml")
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8"))
            self.assertEqual(written["callable"]["sol"], False)


class CooldownPillLayoutTests(DashboardProbeMixin, unittest.TestCase):
    """A long cooldown reason must stay inside its card.

    Observed 2026-09-09: "cooling down · 355m · model unavailable on this account"
    spilled out of the Spark card and pushed its switch onto the neighbouring one.
    The reason text grew when durable-unavailability cooldowns were added, and the
    pill was pinned to a single line.
    """

    def _rule(self, selector):
        import re

        match = re.search(r"(?<![\w\s])" + re.escape(selector) + r"\{([^}]*)\}", HTML)
        self.assertIsNotNone(match, f"{selector} has no rule")
        return match.group(1)

    def test_the_pill_may_wrap(self):
        rule = self._rule(".cooldown-pill")
        self.assertNotIn("white-space:nowrap", rule)
        self.assertIn("overflow-wrap:anywhere", rule)

    def test_the_pill_cannot_exceed_the_card(self):
        self.assertIn("max-width:100%", self._rule(".cooldown-pill"))

    def test_the_label_column_is_allowed_to_shrink(self):
        """Without min-width:0 a flex item never shrinks below its content, which is
        what pushed the switch out rather than wrapping the text."""
        self.assertIn("min-width:0", self._rule(".toggle-label"))

    def test_the_switch_keeps_its_size(self):
        self.assertIn("flex:0 0 44px", self._rule(".switch"))


class SettingsLabelTests(DashboardProbeMixin, unittest.TestCase):
    def test_the_default_model_says_what_only_it_controls(self):
        """It reads as redundant next to the preference chains unless it names the
        one thing a chain cannot change: the model Hermes itself starts on."""
        english, hungarian = self.i18n("settings.main.sub")
        self.assertIn("starts on", english)
        self.assertIn("indul", hungarian)

    def test_the_preference_chains_say_what_they_do_not_change(self):
        english, hungarian = self.i18n("settings.prefs.sub")
        self.assertIn("does not change the model Hermes starts on", english)
        self.assertIn("indulási modelljét nem", hungarian)


class DefaultModelSaveTests(unittest.TestCase):
    """Saving the Settings tab must not move the model Hermes starts on.

    The page posts ``default_model`` on every save -- a callable toggle, a
    preference chain, a fallback edit -- so an unrelated save used to rewrite
    Hermes's own ``model.default`` back onto this provider's tier. One click
    undid a Claude-parent setup, and the write left no restore point.
    """

    CONFIG = {
        "models": {"luna": "gpt-6-luna", "terra": "gpt-5.6-terra", "sol": "gpt-6-sol"},
        "callable": {"luna": True, "terra": True, "sol": False, "opus5": True},
        "fallbacks": {"sol": "terra"},
        "tier_providers": {
            "luna": "openai-codex", "terra": "openai-codex",
            "sol": "openai-codex", "opus5": "anthropic",
        },
        "default_model": "terra",
    }
    HERMES = "model:\n  default: claude-opus-5-5\n  provider: anthropic\n"
    HERMES_CODEX = "model:\n  default: gpt-5.6-terra\n  provider: openai-codex\n"

    def _config(self):
        import copy

        return copy.deepcopy(self.CONFIG)

    def _hermes_file(self, directory, content=None):
        target = Path(directory) / "config.yaml"
        target.write_text(self.HERMES if content is None else content, encoding="utf-8")
        return target

    def test_saving_without_changing_the_default_model_leaves_hermes_alone(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("terra", config)
            self.assertIsNone(error)
            self.assertEqual(target.read_text(encoding="utf-8"), self.HERMES)
            self.assertEqual(list(Path(directory).glob("config.yaml.bak-router-*")), [])

    def test_changing_the_default_model_writes_it_through_with_a_restore_point(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory, self.HERMES_CODEX)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("luna", config)
            self.assertIsNone(error)
            self.assertEqual(config["default_model"], "luna")
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["model"]["default"], "gpt-6-luna")
            self.assertEqual(written["model"]["provider"], "openai-codex")
            self.assertEqual(written["model"]["api_mode"], "codex_responses")
            backups = list(Path(directory).glob("config.yaml.bak-router-*"))
            self.assertEqual(len(backups), 1)
            self.assertIn("gpt-5.6-terra", backups[0].read_text(encoding="utf-8"))

    def test_a_disabled_tier_still_resolves_through_the_fallback_chain(self):
        config = self._config()
        config["default_model"] = "luna"
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory, self.HERMES_CODEX)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("sol", config)
            self.assertIsNone(error)
            self.assertEqual(config["default_model"], "terra")
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            self.assertEqual(written["model"]["default"], "gpt-5.6-terra")

    def test_a_parent_on_another_account_is_never_moved(self):
        """Measured live 2026-09-18: switching Qwen off moved default_model qwen->terra,
        and the save wrote gpt-5.6-terra into Hermes's model block -- the Opus parent
        was gone at the next Hermes start. The router's default tier is its own
        setting; a parent the router doesn't serve is set in Hermes's config only."""
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("luna", config)
            self.assertIsNone(error)
            self.assertEqual(config["default_model"], "luna")
            self.assertEqual(target.read_text(encoding="utf-8"), self.HERMES)
            self.assertEqual(list(Path(directory).glob("config.yaml.bak-router-*")), [])

    def test_a_chain_with_no_enabled_tier_is_refused(self):
        config = self._config()
        config["callable"]["terra"] = False
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("sol", config)
            self.assertIn("No enabled fallback", error)
            self.assertEqual(config["default_model"], "terra")
            self.assertEqual(target.read_text(encoding="utf-8"), self.HERMES)

    def test_a_delegation_target_is_refused_rather_than_blanking_the_model(self):
        """opus5 has no entry in `models`, so the old code wrote model.default: ''
        and the router's own _decision would raise KeyError on the tier."""
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                error = web_viewer._save_default_model("opus5", config)
            self.assertIsNotNone(error)
            self.assertIn("opus5", error)
            self.assertEqual(config["default_model"], "terra")
            self.assertEqual(target.read_text(encoding="utf-8"), self.HERMES)


class ConfigPathTests(unittest.TestCase):
    """The dashboard must read the config file the router actually loads.

    The path was hardcoded to ~/.hermes/plugins/model_router/, but `hermes
    plugins install` creates model-router (the manifest name), so on a normal
    install the dashboard read nothing and every save raised.
    """

    def test_the_dashboard_reads_the_file_beside_it(self):
        self.assertEqual(
            web_viewer.CONFIG_PATH,
            Path(web_viewer.__file__).resolve().parent / "router_config.yaml",
        )

    def test_the_dashboard_and_the_router_agree_on_one_file(self):
        import model_router

        self.assertEqual(web_viewer.CONFIG_PATH, model_router._CONFIG_PATH)


class HaikuDashboardTests(DashboardProbeMixin, unittest.TestCase):
    """Claude delegation adds a third Claude tier; the dashboard must show it
    wherever it shows the other two, or Haiku workers are counted nowhere."""

    def test_haiku_is_styled_like_sonnet5(self):
        import re

        css = "".join(re.findall(r"<style>(.*?)</style>", HTML, re.S))
        selectors = {
            tier: sorted(
                match.group(1).replace(tier, "<tier>")
                for match in re.finditer(r"([^{};]*\.%s[^{};]*)\{" % tier, css)
            )
            for tier in ("sonnet5", "haiku")
        }
        self.assertTrue(selectors["haiku"], "expected haiku to carry tier styling")
        self.assertEqual(selectors["haiku"], selectors["sonnet5"])

    def test_haiku_is_enumerated_everywhere_sonnet5_is(self):
        for fragment in ("--haiku:", 'class="card haiku"',
                         ".pill.haiku{color:var(--haiku)}", ".task-tree-marker.haiku{background:var(--haiku)"):
            with self.subTest(fragment=fragment):
                self.assertIn(fragment, HTML)
        effort = self.javascript_function("renderClaudeReasoningEffort") or ""
        self.assertIn("haiku", effort)
        self.assertIn("sonnet5", effort)

    def test_haiku_is_offered_by_the_tier_filter(self):
        # Since Task 7, the static #tier select only has the "all" option; the
        # per-tier <option>s (haiku included) come from tierFilterOptions(),
        # grouped by account.
        source = self.javascript_function("tierFilterOptions")
        probe = source + "\nconsole.log(tierFilterOptions([{account:'anthropic',label:'Claude',tiers:['haiku','sonnet5','opus5']}]));"
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        self.assertIn("<option>haiku</option>", result.stdout)

    def test_haiku_has_labels_in_both_languages(self):
        for key in ("card.haiku", "model.desc.haiku"):
            with self.subTest(key=key):
                english, hungarian = self.i18n(key)
                self.assertTrue(english and hungarian)

    def test_a_haiku_call_is_classified_as_haiku(self):
        source = self.execution_source()
        parent = {"children": [{"id": "external", "model": "claude-haiku-4-5-20251001",
                                "routed_calls": [{"tier": "haiku", "model": "claude-haiku-4-5-20251001",
                                                  "effort": "external"}], "children": []}]}
        probe = (source + "\nfunction sessionIdFromTurn(entry){return String(entry?.turn_id||'').split(':')[0]}\n"
                 "const scope=executionScope([{tier:'terra'}],[]," + json.dumps(parent) + ",'haiku');"
                 "console.log(JSON.stringify({kinds:scope.nodes.map(executionKind),"
                 "summary:executionSummary([scope.calls])}));")
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertEqual(observed["kinds"], ["haiku"])
        self.assertEqual(observed["summary"]["haiku"], 1)


class AccountsApiTests(unittest.TestCase):
    """The dashboard's per-account view: usage, guard limits, and delegation state,
    the same shape for every account the router can delegate to."""

    def _build_config(self, directory):
        state_path = Path(directory) / "usage-state.json"
        log_path = Path(directory) / "claude-delegation.jsonl"
        fetched_at = time.time() - 60
        state_path.write_text(json.dumps({
            "anthropic": {
                "weekly": 13, "session": 5,
                "weekly_resets_at": "2026-09-24T16:00:00+00:00",
                "session_resets_at": None,
                "fetched_at": fetched_at,
            },
        }), encoding="utf-8")
        log_path.write_text("\n".join([
            json.dumps({"event": "registration", "registered": True, "reason": ""}),
            json.dumps({"event": "delegate_claude", "session_id": "s1", "turn_id": "t1", "outcome": "ran"}),
            json.dumps({"event": "delegate_claude", "session_id": "s1", "turn_id": "t2", "outcome": "refused"}),
        ]), encoding="utf-8")
        config = {
            "callable": {
                "luna": True, "terra": True, "sol": True,
                "haiku": True, "sonnet5": True, "opus5": True,
                # qwen deliberately has no entry at all here (not even False):
                # a tier with a `callable` key, on or off, still gets a card and
                # a switch (see test_a_switched_off_tier_still_appears_with_its_switch);
                # only a tier the config never mentions is truly absent.
            },
            "tier_providers": {
                "luna": "openai-codex", "terra": "openai-codex", "sol": "openai-codex",
                "haiku": "anthropic", "sonnet5": "anthropic", "opus5": "anthropic",
                "qwen": "qwen-token",
            },
            "usage_guard": {
                "state_path": str(state_path),
                "accounts": {
                    "anthropic": {"soft_percent": 70, "hard_percent": 90, "step_down": {"opus5": "sonnet5"}},
                    "openai-codex": {"soft_percent": 70, "hard_percent": 90, "step_down": {"sol": "terra"}},
                },
            },
            "claude_delegation": {
                "enabled": True,
                "default_tier": "sonnet",
                "log_path": str(log_path),
            },
        }
        return config, state_path, log_path

    def setUp(self):
        import model_router

        self.model_router = model_router
        model_router.usage_guard._reset_cache()

    def tearDown(self):
        self.model_router.usage_guard._reset_cache()

    def test_accounts_status_covers_every_account_with_a_callable_tier(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            with patch.object(web_viewer, "_router_module", return_value=self.model_router):
                accounts = web_viewer._accounts_status(config)

        # Qwen has no callable tier, so it is absent entirely.
        self.assertEqual(set(accounts.keys()), {"openai-codex", "anthropic"})

        claude = accounts["anthropic"]
        self.assertEqual(claude["label"], "Claude")
        self.assertEqual(claude["state"], "open")
        self.assertEqual(claude["usage"]["weekly"], 13)
        self.assertTrue(50 <= claude["usage_age_seconds"] <= 120)
        self.assertIs(claude["guard"], True)
        self.assertEqual(claude["soft_percent"], 70)
        self.assertEqual(claude["delegation"], {
            "tool": "delegate_claude",
            "enabled": True,
            "registered": True,
            "restart_needed": False,
            "default_tier": "sonnet",
            "tiers": list(self.model_router.claude_delegation.TIERS),
        })

        codex = accounts["openai-codex"]
        self.assertIsNone(codex["usage"])
        self.assertEqual(codex["state"], "unknown")
        self.assertEqual(codex["delegation"], {"tool": "delegate_task", "always_on": True})

    def test_a_switched_off_tier_still_appears_with_its_switch(self):
        """The shipped config ships spark:false. Dropping it from `tiers` when it
        is off meant the switch that would turn it back on vanished with it."""
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            config["callable"]["spark"] = False
            config["tier_providers"]["spark"] = "openai-codex"
            with patch.object(web_viewer, "_router_module", return_value=self.model_router):
                accounts = web_viewer._accounts_status(config)
        self.assertIn("spark", accounts["openai-codex"]["tiers"])

    def test_an_account_with_every_tier_switched_off_still_gets_a_card(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            for tier in ("haiku", "sonnet5", "opus5"):
                config["callable"][tier] = False
            with patch.object(web_viewer, "_router_module", return_value=self.model_router):
                accounts = web_viewer._accounts_status(config)
        self.assertIn("anthropic", accounts)
        self.assertEqual(set(accounts["anthropic"]["tiers"]), {"haiku", "sonnet5", "opus5"})

    def test_accounts_status_serves_cache_seconds_for_staleness(self):
        """M10: the dashboard greys out a card at 2x cache_seconds, so that
        value has to travel with the account, not be assumed client-side."""
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            config["usage_guard"]["cache_seconds"] = 120
            with patch.object(web_viewer, "_router_module", return_value=self.model_router):
                accounts = web_viewer._accounts_status(config)
        self.assertEqual(accounts["anthropic"]["cache_seconds"], 120)
        self.assertEqual(accounts["openai-codex"]["cache_seconds"], 120)

    def test_accounts_status_never_fetches_usage(self):
        from unittest.mock import MagicMock

        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            anthropic_fetcher = MagicMock()
            codex_fetcher = MagicMock()
            with patch.dict(self.model_router.usage_guard.FETCHERS,
                             {"anthropic": anthropic_fetcher, "openai-codex": codex_fetcher}), \
                 patch.object(web_viewer, "_router_module", return_value=self.model_router):
                web_viewer._accounts_status(config)
            anthropic_fetcher.assert_not_called()
            codex_fetcher.assert_not_called()

    def test_save_usage_limits_updates_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)

            error = web_viewer._save_usage_limits(
                {"anthropic": {"soft_percent": 60, "hard_percent": 85}}, config
            )
            self.assertIsNone(error)
            self.assertEqual(config["usage_guard"]["accounts"]["anthropic"]["soft_percent"], 60)
            self.assertEqual(config["usage_guard"]["accounts"]["anthropic"]["hard_percent"], 85)

            self.assertIsNotNone(web_viewer._save_usage_limits(
                {"anthropic": {"soft_percent": 90, "hard_percent": 85}}, config))
            self.assertIsNotNone(web_viewer._save_usage_limits(
                {"anthropic": {"soft_percent": 10, "hard_percent": 150}}, config))
            self.assertIsNotNone(web_viewer._save_usage_limits(
                {"anthropic": {"soft_percent": 0, "hard_percent": 90}}, config))
            self.assertIsNotNone(web_viewer._save_usage_limits(
                {"nope": {"soft_percent": 10, "hard_percent": 90}}, config))

    def test_whole_percentages_are_saved_as_integers(self):
        config = {"usage_guard": {"accounts": {"anthropic": {"soft_percent": 70, "hard_percent": 90}}}}
        self.assertIsNone(web_viewer._save_usage_limits({"anthropic": {"soft_percent": 80, "hard_percent": 92.5}}, config))
        limits = config["usage_guard"]["accounts"]["anthropic"]
        self.assertEqual((repr(limits["soft_percent"]), repr(limits["hard_percent"])), ("80", "92.5"))

    def test_an_empty_usage_limits_payload_touches_nothing(self):
        """M9: setdefault()-ing usage_guard/accounts even for an empty payload
        added an empty `usage_guard: {accounts: {}}` block to a config that
        never had one."""
        config = {"callable": {}, "tier_providers": {}}
        error = web_viewer._save_usage_limits({}, config)
        self.assertIsNone(error)
        self.assertNotIn("usage_guard", config)

    def test_save_claude_delegation_updates_and_validates(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)

            before = config["claude_delegation"].get("enabled")
            error = web_viewer._save_claude_delegation(
                {"enabled": False, "default_tier": "haiku"}, config
            )
            self.assertIsNone(error)
            self.assertEqual(config["claude_delegation"].get("enabled"), before,
                             "the retired flag from an old tab is ignored, not written")
            self.assertEqual(config["claude_delegation"]["default_tier"], "haiku")

            self.assertIsNotNone(web_viewer._save_claude_delegation({"default_tier": "gpt"}, config))
            self.assertIsNone(web_viewer._save_claude_delegation({"enabled": "yes"}, config))

    def test_read_delegation_log_separates_audits_from_registration(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            audits, registration = web_viewer._read_delegation_log(config)
            self.assertEqual(len(audits), 2)
            self.assertEqual({a["outcome"] for a in audits}, {"ran", "refused"})
            self.assertTrue(all(a["session_id"] == "s1" for a in audits))
            self.assertEqual(registration, {"event": "registration", "registered": True, "reason": ""})

    def test_read_delegation_log_only_reads_the_tail(self):
        """The log is never rotated, so a full read every /api/entries poll would slow down
        forever. Only the last _DELEGATION_LOG_TAIL_BYTES bytes are ever loaded."""
        with tempfile.TemporaryDirectory() as directory:
            config, _, log_path = self._build_config(directory)
            registration = json.dumps({"event": "registration", "registered": True, "reason": ""})
            early_audits = [
                json.dumps({"event": "delegate_claude", "session_id": f"early-{i}",
                            "turn_id": f"t{i}", "outcome": "ran"})
                for i in range(50)
            ]
            late_audits = [
                json.dumps({"event": "delegate_claude", "session_id": f"late-{i}",
                            "turn_id": f"t{i}", "outcome": "ran"})
                for i in range(5)
            ]
            log_path.write_text("\n".join([registration] + early_audits + late_audits), encoding="utf-8")

            with patch.object(web_viewer, "_DELEGATION_LOG_TAIL_BYTES", 400):
                audits, tail_registration = web_viewer._read_delegation_log(config)

            # Only lines from the tail come back: the early lines (and the registration,
            # which precedes them) fall outside the 400-byte window.
            self.assertTrue(audits, "expected at least one audit line from the tail")
            session_ids = {a["session_id"] for a in audits}
            self.assertTrue(session_ids.issubset({f"late-{i}" for i in range(5)}))
            self.assertFalse(session_ids & {f"early-{i}" for i in range(50)})
            self.assertIsNone(tail_registration)
            # Every returned line parsed cleanly -- a seek into the middle of a line would
            # have produced a JSON error that is silently skipped, not a corrupt entry.
            for audit in audits:
                self.assertEqual(audit["event"], "delegate_claude")

    def test_read_delegation_log_skips_a_non_utf8_line(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, log_path = self._build_config(directory)
            good_line = json.dumps({"event": "delegate_claude", "session_id": "s2",
                                     "turn_id": "t9", "outcome": "ran"}).encode("utf-8")
            bad_line = b"\xff\xfe not valid utf-8 \x80\x81"
            log_path.write_bytes(good_line + b"\n" + bad_line + b"\n")

            audits, registration = web_viewer._read_delegation_log(config)

            self.assertEqual(len(audits), 1)
            self.assertEqual(audits[0]["session_id"], "s2")
            self.assertIsNone(registration)

    def test_registration_is_found_within_the_tail_regardless_of_the_audit_limit(self):
        """M7: the old code additionally windowed the already-tail-restricted
        lines down to `limit*2`, so a registration line older than that second
        window (but still well within the tail bytes) was missed even though
        it was right there in what got read."""
        with tempfile.TemporaryDirectory() as directory:
            config, _, log_path = self._build_config(directory)
            registration = json.dumps({"event": "registration", "registered": True, "reason": ""})
            audits = [
                json.dumps({"event": "delegate_claude", "session_id": f"s{i}",
                            "turn_id": f"t{i}", "outcome": "ran"})
                for i in range(10)
            ]
            log_path.write_text("\n".join([registration] + audits), encoding="utf-8")

            _audits, found = web_viewer._read_delegation_log(config, limit=2)

            self.assertEqual(found, {"event": "registration", "registered": True, "reason": ""})
            self.assertEqual(len(_audits), 2)

    def test_usage_refresh_endpoint_reads_through_the_shared_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _, _ = self._build_config(directory)
            config_path = Path(directory) / "router_config.yaml"
            with open(config_path, "w", encoding="utf-8") as f:
                web_viewer.yaml.dump(config, f)

            reading = self.model_router.usage_guard.Reading(
                weekly=20.0, session=4.0,
                weekly_resets_at="2026-09-25T00:00:00+00:00", session_resets_at=None,
                fetched_at=time.time(),
            )

            def fake_read(account, cfg, *, force=False):
                # Mimics the real read()'s cache side effect, without a network call.
                self.model_router.usage_guard._slot(account)["reading"] = reading
                return reading

            with patch.object(web_viewer, "CONFIG_PATH", config_path), \
                 patch.object(self.model_router.usage_guard, "read", side_effect=fake_read) as mocked_read:
                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/usage/refresh?account=anthropic",
                        method="POST",
                    )
                    with urllib.request.urlopen(request) as response:
                        status = response.status
                        payload = json.load(response)
                    self.assertEqual(status, 200)
                    self.assertEqual(payload["account"]["usage"]["weekly"], 20.0)
                    mocked_read.assert_called_once()
                    # The button is an explicit request, so it must not be
                    # throttled by the TTL that paces background reads.
                    self.assertTrue(mocked_read.call_args.kwargs.get("force"),
                                    "the refresh endpoint must force a live read")

                    with patch.object(self.model_router.usage_guard, "read", return_value=None), \
                         patch.object(self.model_router.usage_guard, "last_failure", return_value="boom"):
                        try:
                            urllib.request.urlopen(urllib.request.Request(
                                f"http://127.0.0.1:{server.server_port}/api/usage/refresh?account=anthropic",
                                method="POST"))
                            self.fail("a refresh that read nothing must not answer 200")
                        except urllib.error.HTTPError as exc:
                            self.assertEqual(exc.code, 502)
                            self.assertIn("boom", json.load(exc)["error"])

                    bad_request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/usage/refresh?account=nope",
                        method="POST",
                    )
                    try:
                        urllib.request.urlopen(bad_request)
                        self.fail("expected HTTPError for unknown account")
                    except urllib.error.HTTPError as exc:
                        self.assertEqual(exc.code, 400)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)


class AccountCardTests(DashboardProbeMixin, unittest.TestCase):
    """Settings aligns each model's switch, title and effort dropdown within
    the Models block, followed by account-level Limits, Delegation and Usage."""

    CODEX_INFO = {
        "label": "Codex",
        "tiers": ["luna", "spark", "terra", "sol"],
        "state": "unknown",
        "usage": None,
        "usage_age_seconds": None,
        "has_usage_source": False,
        "guard": True,
        "soft_percent": 70,
        "hard_percent": 90,
        "step_down": {"sol": "terra"},
        "delegation": {"tool": "delegate_task", "always_on": True},
    }

    CLAUDE_INFO = {
        "label": "Claude",
        "tiers": ["haiku", "opus5", "sonnet5"],
        "state": "open",
        "usage": {
            "weekly": 13, "session": 5,
            "weekly_resets_at": "2026-09-24T16:00:00+00:00",
            "session_resets_at": None,
        },
        "usage_age_seconds": 60,
        "has_usage_source": True,
        "guard": True,
        "soft_percent": 70,
        "hard_percent": 90,
        "step_down": {"opus5": "sonnet5"},
        "delegation": {
            "tool": "delegate_claude", "enabled": True, "registered": True,
            "restart_needed": False, "default_tier": "sonnet",
            "tiers": ["haiku", "sonnet", "opus"],
        },
    }

    GROK_INFO = {
        "label": "Grok",
        "tiers": ["grok"],
        "state": "unknown",
        "usage": None,
        "usage_age_seconds": None,
        "has_usage_source": False,
        "guard": False,
        "soft_percent": None,
        "hard_percent": None,
        "step_down": {},
        "delegation": {"tool": "delegate_task", "always_on": True},
    }

    NO_USAGE_SOURCE_INFO = {
        "label": "Qwen",
        "tiers": ["qwen"],
        "state": "unknown",
        "usage": None,
        "usage_age_seconds": None,
        "has_usage_source": False,
        "guard": False,
        "soft_percent": None,
        "hard_percent": None,
        "step_down": {},
        "delegation": {"tool": "delegate_task", "always_on": True},
    }

    def _account_functions(self):
        return "const ROUTER_EFFORT_TIERS=['luna','spark','terra','sol','grok'];\n" + "\n".join(self.javascript_function(name)
                          for name in ("ageText", "resetText", "usageRow", "shortModelName",
                                       "escapeHtml", "renderEffort", "renderClaudeReasoningEffort", "usageError", "accountCard"))

    def _run(self, script):
        # 'status.locale' must resolve to a real BCP-47 tag: toLocaleString throws
        # on the stub's usual echo-the-key behaviour.
        probe = "const t=k=>k==='status.locale'?'en-US':k;\n" + self._account_functions() + "\n" + script
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return result.stdout.strip()

    def _card(self, account, info, extra_config=None):
        config = {"callable": {}, "cooldowns": {}, "load": {"openai-codex": 3, "anthropic": 1},
                   "window_minutes": 60, "accounts": {account: info}}
        if extra_config:
            config.update(extra_config)
        script = (f"let currentConfig={json.dumps(config)};"
                  f"console.log(accountCard({json.dumps(account)},currentConfig.accounts[{json.dumps(account)}]));")
        return self._run(script)

    def test_markup_has_the_new_containers_not_the_old_ones(self):
        start = HTML.index('id="settings-panel"')
        end = HTML.index("</section>", start)
        panel = HTML[start:end]
        self.assertIn('id="account-cards"', panel)
        self.assertNotIn('id="callable-toggles"', HTML)
        self.assertNotIn('id="account-load"', HTML)

    def test_every_card_has_the_same_row_structure(self):
        import re

        codex_card = self._card("openai-codex", self.CODEX_INFO)
        claude_card = self._card("anthropic", self.CLAUDE_INFO)
        # Models | Limits | Delegation; Usage and its load count are full width.
        expected = ["models", "limits", "delegation", "usage"]
        self.assertEqual(re.findall(r'class="account-row ([\w-]+)"', codex_card), expected)
        self.assertEqual(re.findall(r'class="account-row ([\w-]+)"', claude_card), expected)
        self.assertNotIn("account.effort", codex_card + claude_card)
        for card in (codex_card, claude_card):
            usage_row = card[card.index('class="account-row usage"'):]
            self.assertIn("account.load.calls", usage_row)
        claude_footer = claude_card[claude_card.index('class="usage-age"'):]
        self.assertLess(claude_footer.index("data-refresh-usage"), claude_footer.index("account.load.calls"))

    def test_effort_controls_follow_account_models_and_do_not_appear_in_qwen(self):
        codex = self._card("openai-codex", self.CODEX_INFO, {"effort": {"luna": "low", "terra": "high", "sol": "xhigh"}})
        grok = self._card("xai-oauth", self.GROK_INFO, {
            "callable": {"grok": False}, "effort": {"grok": "xhigh"},
        })
        claude = self._card("anthropic", self.CLAUDE_INFO, {
            "claude_reasoning_effort": {"available": True, "levels": {"sonnet": "low", "opus": "high"}},
        })
        qwen = self._card("qwen-token", self.NO_USAGE_SOURCE_INFO)
        for model in self.CODEX_INFO["tiers"]:
            self.assertIn(f'data-effort="{model}"', codex)
        self.assertLess(codex.index('data-effort="luna"'), codex.index('data-effort="terra"'))
        self.assertIn('<option value="high" selected>', codex)
        self.assertIn('data-effort="grok"', grok)
        grok_switch = grok[grok.index('class="model-switch disabled"'):grok.index('</div>', grok.index('class="model-switch disabled"'))]
        self.assertIn('data-effort="grok"', grok_switch)
        self.assertIn('<option value="xhigh" selected>', grok_switch)
        # A switched-off model's effort cannot be changed until it is switched back on.
        self.assertRegex(grok_switch, r'<select data-effort="grok" disabled>')
        self.assertNotIn('model-list no-effort', grok)
        self.assertIn('data-claude-effort="sonnet"', claude)
        self.assertIn('data-claude-effort="opus"', claude)
        self.assertNotIn('data-claude-effort="haiku"', claude)
        self.assertIn("settings.claude_effort.haiku_no_reasoning", claude)
        self.assertNotIn('data-effort=', qwen)
        self.assertNotIn('data-claude-effort=', qwen)
        self.assertNotIn('<select', qwen)
        self.assertNotIn('account.effort', qwen)

    def test_claude_effort_dropdown_selects_default_when_the_tier_is_not_pinned(self):
        card = self._card("anthropic", self.CLAUDE_INFO, {
            "claude_reasoning_effort": {
                "available": True,
                "levels": {"sonnet": "medium", "opus": "high"},
                "defaults": {"sonnet": "medium", "opus": "medium"},
                "pinned": {"sonnet": False, "opus": True},
            },
        })
        sonnet = card[card.index('data-claude-effort="sonnet"'):card.index('</select>', card.index('data-claude-effort="sonnet"'))]
        opus = card[card.index('data-claude-effort="opus"'):card.index('</select>', card.index('data-claude-effort="opus"'))]
        self.assertIn('<option value="" selected>settings.claude_effort.default', sonnet)
        self.assertIn('<option value="">settings.claude_effort.default', opus)
        self.assertNotIn('<option value="" selected>settings.claude_effort.default', opus)

    def test_each_effort_control_shares_its_model_row(self):
        import re

        codex = self._card("openai-codex", self.CODEX_INFO, {
            "callable": {},
            "cooldowns": {"luna": {"seconds": 120, "reason": "3 failures within 60s"}},
            "effort": {"luna": "low", "terra": "high", "sol": "xhigh"},
        })
        claude = self._card("anthropic", self.CLAUDE_INFO, {
            "claude_reasoning_effort": {"available": True, "levels": {"sonnet": "low", "opus": "high"}},
        })

        def switch_for(card, model):
            match = re.search(
                r'<div class="model-switch[^\"]*">(?:(?!</div>).)*'
                + rf'data-model="{model}"(?:(?!</div>).)*</div>',
                card,
                re.DOTALL,
            )
            self.assertIsNotNone(match, f"{model} has no model switch")
            return match.group(0)

        for model in self.CODEX_INFO["tiers"]:
            switch = switch_for(codex, model)
            self.assertIn(f'data-effort="{model}"', switch)
        self.assertIn("3 failures within 60s", switch_for(codex, "luna"))
        for model, effort in (("sonnet5", "sonnet"), ("opus5", "opus")):
            self.assertIn(f'data-claude-effort="{effort}"', switch_for(claude, model))

    def test_a_switched_off_models_effort_dropdown_is_disabled(self):
        import re

        def select_for(card, model):
            switch = re.search(r'<div class="model-switch[^"]*">(?:(?!</div>).)*data-model="'
                               + model + r'"(?:(?!</div>).)*</div>', card, re.DOTALL)
            self.assertIsNotNone(switch, f"{model} has no model switch")
            found = re.search(r'<select[^>]*>', switch.group(0))
            self.assertIsNotNone(found, f"{model} has no effort select")
            return found.group(0)

        codex = self._card("openai-codex", self.CODEX_INFO, {
            "callable": {"terra": False}, "effort": {"luna": "low", "terra": "high"},
        })
        self.assertRegex(select_for(codex, "terra"), r'^<select data-effort="terra" disabled>$')
        self.assertEqual(select_for(codex, "luna"), '<select data-effort="luna">')

        claude = self._card("anthropic", self.CLAUDE_INFO, {
            "callable": {"opus5": False},
            "claude_reasoning_effort": {"available": True, "levels": {"sonnet": "low", "opus": "high"}},
        })
        self.assertRegex(select_for(claude, "opus5"), r'^<select data-claude-effort="opus" disabled>$')
        self.assertEqual(select_for(claude, "sonnet5"), '<select data-claude-effort="sonnet">')

        # Switched off and unavailable at once: one disabled attribute, the reason still shown.
        both = self._card("anthropic", self.CLAUDE_INFO, {
            "callable": {"sonnet5": False},
            "claude_reasoning_effort": {"available": False, "reason": "No module named 'x'", "levels": {}},
        })
        sonnet = select_for(both, "sonnet5")
        self.assertEqual(sonnet.count(" disabled"), 1)
        self.assertIn("title=", sonnet)

    def test_haiku_has_a_disabled_unsaved_placeholder(self):
        import re
        card = self._card("anthropic", self.CLAUDE_INFO)
        switch = re.search(r'<div class="model-switch[^\"]*">(?:(?!</div>).)*data-model="haiku"(?:(?!</div>).)*</div>', card, re.DOTALL)
        self.assertIsNotNone(switch)
        self.assertRegex(switch.group(0), r'<select class="effort-none" disabled><option>settings.claude_effort.haiku_no_reasoning</option></select>')
        self.assertNotIn('data-claude-effort=', switch.group(0))

    def test_dropdowns_share_a_fixed_width_css_rule(self):
        import re
        self.assertRegex(HTML, r'\.account-card \.model-switch select\{[^}]*width:190px;[^}]*\}')
        self.assertIn('grid-template-columns:44px max-content 190px', HTML)
        # Descenders ('g' in 'no reasoning allowed') need an explicit line box and a taller row.
        self.assertRegex(HTML, r'\.account-card \.model-switch select\{[^}]*line-height:20px;[^}]*padding:7px 8px[^}]*\}')
        self.assertRegex(HTML, r'\.account-card \.model-switch\{[^}]*min-height:50px[^}]*\}')
        # Haiku's placeholder is greyed out beyond the ordinary disabled look.
        self.assertRegex(HTML, r'\.account-card \.model-switch select\.effort-none\{[^}]*opacity:[^}]*\}')
        self.assertIn('grid-template-columns:subgrid', HTML)
        self.assertIn('.model-list.no-effort{grid-template-columns:44px max-content}', HTML)
        self.assertIn('.model-switch .cooldown-pill{grid-column:1/-1', HTML)

    def test_unavailable_claude_effort_stays_visible_but_disabled_in_account_card(self):
        card = self._card("anthropic", self.CLAUDE_INFO, {
            "claude_reasoning_effort": {"available": False, "reason": "seam moved", "levels": {"sonnet": "medium", "opus": "medium"}},
        })
        self.assertIn('data-claude-effort="sonnet" disabled', card)
        self.assertIn('data-claude-effort="opus" disabled title="seam moved"', card)
        self.assertEqual(card.count('title="seam moved"'), 2)
        self.assertNotIn('>seam moved<', card)

    def test_unavailable_claude_effort_reason_is_html_escaped(self):
        card = self._card("anthropic", self.CLAUDE_INFO, {
            "claude_reasoning_effort": {"available": False, "reason": "<img src=x onerror=alert(1)>", "levels": {}},
        })
        self.assertEqual(card.count('title="&lt;img src=x onerror=alert(1)&gt;"'), 2)
        self.assertNotIn("<img src=x onerror=alert(1)>", card)
        self.assertNotIn('>&lt;img src=x onerror=alert(1)&gt;<', card)

    def test_standalone_effort_sections_are_removed(self):
        self.assertNotIn('id="effort-settings"', HTML)
        self.assertNotIn('id="claude-effort-settings"', HTML)

    def test_model_switches_drop_the_repeated_vendor_prefix(self):
        script = ("console.log(JSON.stringify(['GPT-5.6 Luna','GPT-5.3 Spark','Claude Opus 5.5',"
                  "'Claude Haiku 4.5','Qwen 3.7 Plus'].map(shortModelName)));")
        self.assertEqual(json.loads(self._run("let currentConfig={};" + script)),
                         ["Luna 5.6", "Spark 5.3", "Opus 5.5", "Haiku 4.5", "Qwen 3.7 Plus"])

    def test_claude_card_specifics(self):
        registered = self._card("anthropic", self.CLAUDE_INFO)
        # The Workflow switch owns delegate_claude being on; the card only reports it.
        self.assertNotIn("data-account-toggle", registered)
        self.assertIn("account.delegation.via", registered)
        # The default tier moved to the Workers section (renderWorkers).
        self.assertNotIn('<select data-default-tier', registered)
        self.assertIn("account.delegation.live", registered)

        restart_info = dict(self.CLAUDE_INFO,
                             delegation=dict(self.CLAUDE_INFO["delegation"], restart_needed=True))
        restarting = self._card("anthropic", restart_info)
        self.assertIn("account.delegation.restart", restarting)

    def test_the_live_and_restart_badge_lives_in_the_card_header_not_the_delegation_row(self):
        """M7: it used to sit inside the Delegation row; the card header (next
        to the state badge) is where the other at-a-glance status lives."""
        registered = self._card("anthropic", self.CLAUDE_INFO)
        head_start = registered.index('class="account-head"')
        head_end = registered.index('</div>', head_start)
        head = registered[head_start:head_end]
        self.assertIn("account.delegation.live", head)

        delegation_row_start = registered.index('class="account-row delegation"')
        delegation_row_end = registered.index('</div></div>', delegation_row_start)
        delegation_row = registered[delegation_row_start:delegation_row_end]
        self.assertNotIn("account.delegation.live", delegation_row)

        restart_info = dict(self.CLAUDE_INFO,
                             delegation=dict(self.CLAUDE_INFO["delegation"], restart_needed=True))
        restarting = self._card("anthropic", restart_info)
        restart_head_start = restarting.index('class="account-head"')
        restart_head_end = restarting.index('</div>', restart_head_start)
        self.assertIn("account.delegation.restart", restarting[restart_head_start:restart_head_end])

    def test_codex_card_specifics(self):
        codex_card = self._card("openai-codex", self.CODEX_INFO)
        self.assertIn("account.delegation.always", codex_card)
        self.assertNotIn("data-account-toggle", codex_card)

    def test_an_account_without_a_usage_source_shows_the_none_message_and_disabled_limits(self):
        card = self._card("qwen-token", self.NO_USAGE_SOURCE_INFO)
        self.assertIn("account.usage.none", card)
        self.assertIn('data-limit="soft" data-account="qwen-token" value="" disabled', card)
        self.assertIn('data-limit="hard" data-account="qwen-token" value="" disabled', card)

    def test_cooldown_pill_shows_its_reason(self):
        """M10: the old UI showed why a tier is cooling, not just for how long."""
        config = {
            "callable": {}, "cooldowns": {"luna": {"seconds": 120, "reason": "3 failures within 60s"}},
            "load": {}, "window_minutes": 60, "accounts": {"openai-codex": self.CODEX_INFO},
        }
        script = (f"let currentConfig={json.dumps(config)};"
                  f"console.log(accountCard('openai-codex',currentConfig.accounts['openai-codex']));")
        out = self._run(script)
        self.assertIn("3 failures within 60s", out)

    def test_cooldown_pill_escapes_its_reason(self):
        """C-M4: the reason is untrusted (upstream failure text) and must go through
        escapeHtml like the other reason strings in this file (state.reason, usage
        errors), not be interpolated raw into the pill."""
        config = {
            "callable": {}, "cooldowns": {"luna": {"seconds": 120, "reason": "<img src=x onerror=alert(1)>"}},
            "load": {}, "window_minutes": 60, "accounts": {"openai-codex": self.CODEX_INFO},
        }
        script = (f"let currentConfig={json.dumps(config)};"
                  f"console.log(accountCard('openai-codex',currentConfig.accounts['openai-codex']));")
        out = self._run(script)
        self.assertIn("&lt;img src=x onerror=alert(1)&gt;", out)
        self.assertNotIn("<img src=x onerror=alert(1)>", out)

    def test_a_card_greys_out_at_twice_the_cache_seconds_not_a_fixed_ten_minutes(self):
        """M10: staleness used to be a hardcoded 600s; it now follows 2x
        whatever cache_seconds the accounts payload actually served."""
        fresh_info = dict(self.CLAUDE_INFO, usage_age_seconds=250, cache_seconds=200)
        stale_info = dict(self.CLAUDE_INFO, usage_age_seconds=250, cache_seconds=100)
        self.assertNotIn(' stale', self._card("anthropic", fresh_info))
        self.assertIn(' stale', self._card("anthropic", stale_info))

    def test_usage_row_places_soft_and_hard_ticks(self):
        script = "console.log(usageRow('account.usage.week',55,null,70,90));"
        row = self._run(script)
        self.assertIn('style="left:70%"', row)
        self.assertIn('style="left:90%"', row)

    def test_every_i18n_key_used_by_the_new_code_exists_in_both_languages(self):
        for key in [
            "settings.accounts.heading", "account.state.open", "account.state.soft",
            "account.state.closed", "account.state.unknown", "account.models",
            "account.usage", "account.usage.week", "account.usage.session",
            "account.usage.resets", "account.usage.age", "account.usage.none",
            "account.usage.refresh", "account.limits", "account.limits.soft",
            "account.limits.hard", "account.limits.stepdown", "account.delegation",
            "account.delegation.via", "account.delegation.always",
            "account.delegation.default_tier", "account.delegation.live",
            "account.delegation.restart", "account.load", "account.load.calls",
            "settings.main.heading", "settings.main.sub", "settings.main.primary",
            "settings.main.external", "settings.workers.heading",
            "settings.workers.sub", "settings.workers.codex", "settings.workers.codex.desc",
            "settings.workers.claude", "settings.workers.claude.desc",
            "settings.workers.fallback", "settings.workers.fallback.desc",
        ]:
            self.i18n(key)

    def test_save_payload_includes_usage_limits_and_claude_delegation(self):
        source = self.javascript_function("saveSettings")
        listener = HTML[HTML.index("document.getElementById('account-cards').addEventListener('change'"):HTML.index("document.getElementById('balance-settings')")]
        self.assertIn("usage_limits:", source)
        self.assertIn("claude_delegation:", source)
        self.assertIn("currentConfig.effort=Object.assign({},currentConfig.effort,{[el.dataset.effort]:el.value})", listener)

    def test_every_model_control_is_the_same_slider_toggle_as_the_delegation_switch(self):
        """Live check finding: a model on/off switch must look like the Claude
        delegation switch -- a slider, not a bare checkbox -- while keeping the
        data-model attribute the change listener relies on."""
        import re

        card = self._card("anthropic", self.CLAUDE_INFO)
        for tier in self.CLAUDE_INFO["tiers"]:
            match = re.search(
                r'<label class="switch"><input type="checkbox" data-model="' + tier
                + r'"[^>]*><span class="slider"></span></label>',
                card,
            )
            self.assertIsNotNone(match, f"{tier} is not rendered as a switch/slider control")


class AccountGroupTests(DashboardProbeMixin, unittest.TestCase):
    """The main view groups the per-model count cards by account, with a compact
    weekly usage bar per group, and the #tier filter mirrors the same grouping."""

    TIER_ACCOUNTS = {
        "luna": "openai-codex", "spark": "openai-codex", "terra": "openai-codex",
        "sol": "openai-codex", "haiku": "anthropic", "sonnet5": "anthropic",
        "opus5": "anthropic", "qwen": "qwen-token",
    }
    ACCOUNTS = {
        "openai-codex": {"label": "Codex"},
        "anthropic": {"label": "Claude"},
        "qwen-token": {"label": "Qwen"},
    }

    def _group_source(self):
        start = HTML.index("const ACCOUNT_ORDER=")
        end = HTML.index("function accountCard(account,info)")
        return HTML[start:end]

    def _run(self, script, language="en"):
        probe = self.i18n_runtime(language) + "\n" + self.javascript_function("resetText") + "\n" \
            + self.javascript_function("ageText") + "\n" + self.javascript_function("usageRow") + "\n" \
            + self._group_source() + "\n" + script
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return result.stdout.strip()

    def test_accounts_are_ordered_codex_then_claude_then_others(self):
        out = self._run(
            "console.log(JSON.stringify(accountGroupsFor("
            + json.dumps(self.TIER_ACCOUNTS) + "," + json.dumps(self.ACCOUNTS) + ")));"
        )
        groups = json.loads(out)
        self.assertEqual(
            [(g["account"], g["tiers"]) for g in groups],
            [
                ("openai-codex", ["luna", "spark", "terra", "sol"]),
                ("anthropic", ["haiku", "sonnet5", "opus5"]),
                ("qwen-token", ["qwen"]),
            ],
        )

    def test_an_account_with_no_configured_tiers_is_omitted(self):
        accounts = dict(self.ACCOUNTS, **{"unused-account": {"label": "Unused"}})
        out = self._run(
            "console.log(JSON.stringify(accountGroupsFor("
            + json.dumps(self.TIER_ACCOUNTS) + "," + json.dumps(accounts) + ")));"
        )
        groups = json.loads(out)
        self.assertNotIn("unused-account", [g["account"] for g in groups])

    def test_a_switched_off_tier_and_an_account_left_empty_leave_the_overview(self):
        callable_ = {"qwen": False, "spark": False}
        out = self._run(
            "console.log(JSON.stringify(accountGroupsFor("
            + json.dumps(self.TIER_ACCOUNTS) + "," + json.dumps(self.ACCOUNTS) + ","
            + json.dumps(callable_) + ")));"
        )
        groups = {g["account"]: g["tiers"] for g in json.loads(out)}
        self.assertNotIn("qwen-token", groups)
        self.assertEqual(groups["openai-codex"], ["luna", "terra", "sol"])

    def test_the_overview_hides_switched_off_cards_instead_of_removing_them(self):
        """render() writes every tier's count by element id, so a removed card
        would break the page; hidden keeps it ready for switching back on."""
        source = self.javascript_function("groupModelCards")
        self.assertIn("card.hidden=callable[tier]===false", source)
        self.assertIn("box.hidden=!shown.has(box.dataset.account)", source)

    def test_claude_group_is_filtered_to_tiers_present_in_tier_accounts(self):
        tier_accounts = dict(self.TIER_ACCOUNTS)
        del tier_accounts["opus5"]
        out = self._run(
            "console.log(JSON.stringify(accountGroupsFor("
            + json.dumps(tier_accounts) + "," + json.dumps(self.ACCOUNTS) + ")));"
        )
        groups = {g["account"]: g["tiers"] for g in json.loads(out)}
        self.assertEqual(groups["anthropic"], ["haiku", "sonnet5"])

    def test_compact_usage_shows_the_weekly_bar_with_ticks_and_state(self):
        info = {
            "state": "soft", "has_usage_source": True, "soft_percent": 70, "hard_percent": 90,
            "usage": {"weekly": 75, "weekly_resets_at": None, "session": 12},
            "usage_age_seconds": 90,
        }
        out = self._run(f"console.log(compactUsage({json.dumps(info)}));")
        self.assertIn('account-usage soft', out)
        self.assertIn('style="left:70%"', out)
        self.assertIn('style="left:90%"', out)
        self.assertIn('75%', out)

    def test_compact_usage_color_follows_state_not_weekly_percent(self):
        """M10: an account can be `closed` on its session window while its
        weekly percent alone would still read green; the compact bar must
        show the account's actual state, not recompute a color from weekly."""
        info = {
            "state": "closed", "has_usage_source": True, "soft_percent": 70, "hard_percent": 90,
            "usage": {"weekly": 10, "weekly_resets_at": None, "session": 95},
            "usage_age_seconds": 30,
        }
        out = self._run(f"console.log(compactUsage({json.dumps(info)}));")
        self.assertIn("background:#ff6b7a", out)

    def test_compact_usage_with_no_usage_source_shows_the_none_message(self):
        out = self._run(f"console.log(compactUsage({json.dumps({'has_usage_source': False})}));")
        english, _ = self.i18n("main.usage.none")
        self.assertIn(english, out)

    def test_tier_filter_options_are_grouped_by_account_label(self):
        out = self._run(
            "console.log(tierFilterOptions(accountGroupsFor("
            + json.dumps(self.TIER_ACCOUNTS) + "," + json.dumps(self.ACCOUNTS) + ")));"
        )
        self.assertIn('<optgroup label="Codex">', out)
        self.assertIn('<optgroup label="Claude">', out)
        self.assertIn('<optgroup label="Qwen">', out)
        codex_start = out.index('<optgroup label="Codex">')
        claude_start = out.index('<optgroup label="Claude">')
        qwen_start = out.index('<optgroup label="Qwen">')
        self.assertLess(claude_start, qwen_start, "Qwen must come after Claude")
        codex_block = out[codex_start:claude_start]
        claude_block = out[claude_start:qwen_start]
        qwen_block = out[qwen_start:]
        for tier in ("luna", "spark", "terra", "sol"):
            with self.subTest(tier=tier):
                self.assertIn(f'<option>{tier}</option>', codex_block)
        for tier in ("sonnet5", "haiku", "opus5"):
            with self.subTest(tier=tier):
                self.assertIn(f'<option>{tier}</option>', claude_block)
        self.assertIn('<option>qwen</option>', qwen_block)

    def test_static_markup_still_has_every_tier_count_id(self):
        cards_start = HTML.index('class="cards"')
        cards_end = HTML.index('id="account-groups"', cards_start)
        cards_html = HTML[cards_start:cards_end]
        for tier in ("luna", "spark", "terra", "sol", "opus5", "sonnet5", "haiku", "qwen"):
            with self.subTest(tier=tier):
                self.assertIn(f'id="{tier}"', cards_html)

    def test_static_tier_select_has_only_the_all_option(self):
        start = HTML.index('<select id="tier">')
        end = HTML.index('</select>', start)
        select_html = HTML[start:end]
        self.assertNotIn('<option>', select_html)
        self.assertIn('data-i18n="router.tier.all"', select_html)

    def test_main_usage_none_exists_in_both_languages(self):
        english, hungarian = self.i18n("main.usage.none")
        self.assertEqual(english, "no usage data")
        self.assertEqual(hungarian, "nincs használati adat")


class DelegationChipTests(DashboardProbeMixin, unittest.TestCase):
    """Every prompt row gets one chip per delegated worker: Codex children come
    from routed calls already in the router log, Claude children come from the
    claude-delegation audit lines that /api/entries returns as `delegations`."""

    ACCOUNTS_STATE = {
        "accounts": {},
        "tier_accounts": {
            "terra": "openai-codex", "sol": "openai-codex",
            "haiku": "anthropic", "sonnet5": "anthropic", "opus5": "anthropic",
        },
    }

    DOM_SHIM = (
        "global.document={createElement(tag){return {tag,className:'',textContent:'',"
        "title:'',children:[],append(...els){this.children.push(...els)}};}};"
    )

    def _source(self):
        names = [
            "executionOwnCalls", "executionRawCall", "executionCalls", "executionKind",
            "sessionIdFromTurn", "tierAccount", "accountLabel", "runForTurnId", "assignDelegations",
            "chip", "stepDownHoverText", "delegationChips",
        ]
        return "\n".join(self.javascript_function(name) for name in names)

    def _run(self, accounts_state, script):
        probe = (
            self.DOM_SHIM
            + f"let accountsState={json.dumps(accounts_state)};"
            + self._source()
            + "\n" + script
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        return result.stdout.strip()

    def test_cli_refusal_audit_reaches_dashboard_chip_without_a_fake_worker(self):
        import time
        from model_router import _maybe_run_opus5, usage_guard
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'claude-audit.jsonl'
            cfg = {'enabled': True, 'callable': {'sonnet5': True},
                   'coding_agent': {'delegated_review': {'enabled': True}},
                   'claude_delegation': {'log_path': str(path)},
                   'usage_guard': {'accounts': {'anthropic': {'soft_percent': 70,
                       'hard_percent': 90}}}}
            request = {'messages': [{'role': 'user', 'content': '[sonnet-review] Review parser'}]}
            with patch('model_router._verified_delegated_claude_review',
                       return_value=(Path(directory), 'sonnet')), \
                 patch('model_router.usage_guard.read',
                       return_value=usage_guard.Reading(95, 10, None, None, time.time())), \
                 patch('model_router._run_opus5_bridge') as bridge:
                self.assertIsNone(_maybe_run_opus5(request, cfg, platform='subagent',
                                   api_mode='codex_responses', turn_id='parent:sa-1'))
            bridge.assert_not_called()
            audits, _ = web_viewer._read_delegation_log(cfg)
        self.assertEqual(len(audits), 1)
        self.assertEqual(audits[0]['event'], 'bridge_claude')
        self.assertEqual(audits[0]['session_id'], 'parent')
        self.assertIn('weekly usage 95%', audits[0]['message'])
        out = self._run(self.ACCOUNTS_STATE,
                        'const box=delegationChips({scope:{nodes:[]}},' + json.dumps(audits)
                        + ');console.log(JSON.stringify(box.children.map(c=>({text:c.textContent,title:c.title}))));')
        self.assertEqual(json.loads(out), [{'text':'Claude: none ✕',
                                            'title': audits[0]['message']}])

    def test_router_jsonl_and_bridge_lifecycle_render_both_step_downs(self):
        import sqlite3
        from agent_activity import load_agent_activity
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "state.db"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE async_delegations (delegation_id TEXT, origin_session TEXT, parent_session_id TEXT, state TEXT, dispatched_at REAL, completed_at REAL, updated_at REAL, task_json TEXT, result_json TEXT)")
                conn.execute("CREATE TABLE sessions (id TEXT, parent_session_id TEXT, started_at REAL, ended_at REAL, model TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER, session_id TEXT, role TEXT, content TEXT, tool_name TEXT, timestamp REAL)")
                conn.execute("INSERT INTO sessions VALUES ('child','parent',100,NULL,'gpt-terra')")
                conn.execute("INSERT INTO messages VALUES (1,'parent','user','Review parser',NULL,90)")
                conn.execute("INSERT INTO messages VALUES (2,'child','user','Inspect parser',NULL,100)")
                conn.execute("INSERT INTO async_delegations VALUES (?,?,?,?,?,?,?,?,?)",
                             ('deleg','parent','parent','completed',99,110,110,
                              json.dumps({'goal':'Inspect parser'}),json.dumps({'api_calls':1})))
            router = Path(directory) / "router.jsonl"
            router.write_text(json.dumps({'turn_id':'child:sa-0:t','tier':'terra',
                                          'model':'gpt-terra','effort':'medium',
                                          'reason':'long work; usage soft limit: sol→terra (weekly 72%)'})+'\n')
            lifecycle = Path(directory) / "bridge.jsonl"
            lifecycle.write_text('\n'.join(json.dumps(event) for event in [
                {'bridge_run_id':'bridge','event':'started','state':'running','timestamp':101,
                 'parent_session_id':'parent','requested_tier':'opus','effective_tier':'sonnet',
                 'requested_model':'claude-sonnet-5-5','review':True,'requested_read_only':True},
                {'bridge_run_id':'bridge','event':'terminal','state':'success','timestamp':111,
                 'parent_session_id':'parent','requested_tier':'opus','effective_tier':'sonnet',
                 'canonical_model':'claude-sonnet-5-5','review':True,'requested_read_only':True,
                 'adjusted':'opus5→sonnet5 (weekly usage 75%)'},
            ])+'\n')
            activity = load_agent_activity(db, now=112, router_log_path=router,
                                           bridge_lifecycle_path=lifecycle)
        parent = next(p for p in activity['parents'] if p['session_id']=='parent')
        self.assertEqual(len(parent['children']),2)
        self.assertEqual(activity['external_bridge_run_ids'], ['bridge'])
        run = {'scope': {'nodes': parent['children']}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>({text:c.textContent,title:c.title}))));"
        ))
        chips = json.loads(out)
        self.assertEqual(chips, [
            {'text':'Codex: terra ↓','title':'sol→terra, weekly 72%'},
            {'text':'Claude: sonnet ↓','title':'opus5→sonnet5, weekly 75%'},
        ])

    def test_assignment_by_session_and_time(self):
        """Two runs in the same session get the audit that landed after them but
        before the next one; an audit in an unrelated session goes nowhere."""
        runs = [
            {"first": {"turn_id": "s1:a", "timestamp": "2026-01-01T10:00:00Z"}},
            {"first": {"turn_id": "s1:b", "timestamp": "2026-01-01T10:05:00Z"}},
        ]
        audits = [
            {"session_id": "s1", "timestamp": "2026-01-01T10:02:00Z", "tag": "first"},
            {"session_id": "s1", "timestamp": "2026-01-01T10:06:00Z", "tag": "second"},
            {"session_id": "s2", "timestamp": "2026-01-01T10:03:00Z", "tag": "stray"},
        ]
        out = self._run(self.ACCOUNTS_STATE, (
            "const map=assignDelegations(" + json.dumps(runs) + "," + json.dumps(audits) + ");"
            "console.log(JSON.stringify([...map.entries()].map(([k,v])=>[k,v.map(a=>a.tag)])));"
        ))
        self.assertEqual(json.loads(out), [[0, ["first"]], [1, ["second"]]])

    def test_assignment_prefers_an_exact_turn_id_match_over_session_and_time(self):
        """M6: the router log's own turn_id is 'session:turn[:suffix]'; an audit
        carrying a turn_id is assigned to whichever run's rawEntries actually
        contains that turn (or a sub-call under it), even when the naive
        session+time rule would have picked the other run."""
        runs = [
            {"first": {"turn_id": "s1:1", "timestamp": "2026-01-01T10:00:00Z"},
             "rawEntries": [{"turn_id": "s1:1"}, {"turn_id": "s1:1:sub"}]},
            {"first": {"turn_id": "s1:2", "timestamp": "2026-01-01T10:05:00Z"},
             "rawEntries": [{"turn_id": "s1:2"}]},
        ]
        # Timestamp alone would land this on run 1 (it lands after run 1's own
        # first timestamp), but the turn_id belongs to run 0.
        audits = [{"session_id": "s1", "turn_id": "s1:1:sub", "timestamp": "2026-01-01T10:06:00Z", "tag": "byturn"}]
        out = self._run(self.ACCOUNTS_STATE, (
            "const map=assignDelegations(" + json.dumps(runs) + "," + json.dumps(audits) + ");"
            "console.log(JSON.stringify([...map.entries()].map(([k,v])=>[k,v.map(a=>a.tag)])));"
        ))
        self.assertEqual(json.loads(out), [[0, ["byturn"]]])

    def test_assignment_falls_back_to_session_and_time_when_no_run_has_the_turn_id(self):
        runs = [{"first": {"turn_id": "s1:a", "timestamp": "2026-01-01T10:00:00Z"}, "rawEntries": []}]
        audits = [{"session_id": "s1", "turn_id": "s1:unrelated", "timestamp": "2026-01-01T10:05:00Z", "tag": "fallback"}]
        out = self._run(self.ACCOUNTS_STATE, (
            "const map=assignDelegations(" + json.dumps(runs) + "," + json.dumps(audits) + ");"
            "console.log(JSON.stringify([...map.entries()].map(([k,v])=>[k,v.map(a=>a.tag)])));"
        ))
        self.assertEqual(json.loads(out), [[0, ["fallback"]]])

    def test_chips_for_both_accounts(self):
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [{"tier": "terra", "model": "terra"}], "children": []},
            {"kind": None, "routed_calls": [{"tier": "haiku", "model": "haiku"}], "children": []},
        ]}}
        audits = [
            {"tier_used": "haiku", "outcome": "ran"},
            {"tier_requested": "opus", "tier_used": "sonnet", "outcome": "lowered",
             "adjusted": "opus→sonnet (weekly usage 74%)"},
            {"tier_requested": "sonnet", "tier_used": "sonnet", "outcome": "refused",
             "message": "Claude delegation closed: …"},
        ]
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + "," + json.dumps(audits) + ");"
            "console.log(JSON.stringify(box.children.map(c=>({text:c.textContent,title:c.title}))));"
        ))
        chips = json.loads(out)
        self.assertEqual([c["text"] for c in chips],
                          ["Codex: terra", "Claude: haiku", "Claude: sonnet ↓", "Claude: sonnet ✕"])
        # M5: one hover format for both accounts -- "<from>→<to>, weekly N%".
        self.assertEqual(chips[2]["title"], "opus→sonnet, weekly 74%")
        # Refused/error chips keep their original message untouched.
        self.assertIn("Claude delegation closed", chips[3]["title"])

    def test_codex_step_down_chip_detects_the_appended_reason_clause(self):
        """Controller ruling: the router now appends the usage-limit clause onto
        the original reason rather than starting the reason with it, so the
        marker must be detected with a regex search, not startsWith."""
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [
                {"tier": "terra", "reason": "long work; usage soft limit: sol→terra (weekly 72%)"}
            ], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>({text:c.textContent,title:c.title}))));"
        ))
        self.assertEqual(json.loads(out), [{
            "text": "Codex: terra ↓",
            "title": "sol→terra, weekly 72%",
        }])

    def test_codex_hard_limit_reason_also_marks_a_step_down_chip(self):
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [
                {"tier": "terra", "reason": "usage hard limit: sol→terra (weekly 91%)"}
            ], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>({text:c.textContent,title:c.title}))));"
        ))
        self.assertEqual(json.loads(out), [{
            "text": "Codex: terra ↓",
            "title": "sol→terra, weekly 91%",
        }])

    def test_codex_hard_limit_session_window_reason_reformats_too(self):
        """M10 makes the router name the session window when that is what
        actually triggered the hard limit; the same hover formatter must
        handle that clause too."""
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [
                {"tier": "terra", "reason": "usage hard limit: sol→terra (session 95%)"}
            ], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>c.title)));"
        ))
        self.assertEqual(json.loads(out), ["sol→terra, session 95%"])

    def test_a_skipped_step_down_never_marks_the_chip(self):
        """I3: the router also writes this clause when the target itself is
        unavailable ('...skipped (terra unavailable)'); that must never be
        read as a real step-down."""
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [
                {"tier": "sol", "reason": "usage soft limit: sol→terra skipped (terra unavailable)"}
            ], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>c.textContent)));"
        ))
        self.assertEqual(json.loads(out), ["Codex: sol"])

    def test_claude_fallback_chip_with_no_audit_shows_the_tier_name_not_the_target(self):
        """No audits means the entry has to be read off the routed node itself,
        whose tier field is a router target name (sonnet5/opus5); the chip must
        still say the short tier name (sonnet/opus) the Settings tier selector
        and the audit log both use."""
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [{"tier": "sonnet5"}], "children": []},
            {"kind": None, "routed_calls": [{"tier": "opus5"}], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>c.textContent)));"
        ))
        self.assertEqual(json.loads(out), ["Claude: sonnet", "Claude: opus"])

    def test_haiku_chip_with_no_audit_log(self):
        run = {"scope": {"nodes": [
            {"kind": None, "routed_calls": [{"tier": "haiku"}], "children": []},
        ]}}
        out = self._run(self.ACCOUNTS_STATE, (
            "const box=delegationChips(" + json.dumps(run) + ",[]);"
            "console.log(JSON.stringify(box.children.map(c=>c.textContent)));"
        ))
        self.assertEqual(json.loads(out), ["Claude: haiku"])

    def test_delegation_chips_are_appended_to_the_routes_cell_not_the_header(self):
        """.router-run-header is a CSS grid with a fixed number of column tracks
        (redefined at the 1120px and 700px breakpoints, plus .no-details) — an
        extra header child auto-places onto a stray grid cell instead of
        flowing inline. The chips must attach to the existing router-run-routes
        cell instead, after its route pills."""
        renderer = HTML[HTML.rindex("function render(){"):]
        self.assertIn("const delegationMap=assignDelegations(runData,delegations)", renderer)
        routes_start = renderer.index(
            "const routes=document.createElement('span');routes.className='router-run-routes';"
        )
        header_append = renderer.index(
            "header.append(dateEl,timeEl,prompt,stateEl,total,routes,workers);"
        )
        between = renderer[routes_start:header_append]
        self.assertIn("routes.append(delegationChips(", between)
        self.assertNotIn("header.append(delegationChips(", renderer)

    def test_delegation_chips_are_a_descendant_of_the_routes_cell_at_runtime(self):
        """Node probe over the live renderer's actual header-building statements
        (not a reimplementation): builds the same header/routes elements the
        real code builds and asserts the chips box lands inside routes.children,
        never directly in header.children."""
        renderer = HTML[HTML.rindex("function render(){"):]
        start = renderer.index("const routes=document.createElement")
        end = renderer.index("if(hasDetails)header.addEventListener")
        segment = renderer[start:end]
        probe = (
            self.DOM_SHIM
            + "const header={children:[],append(...els){this.children.push(...els)}};"
            + "const dateEl='dateEl',timeEl='timeEl',prompt='prompt',stateEl='stateEl',total='total';"
            + "const t=k=>k;const accountingCalls=[];const workerCalls=0;"
            + "function appendRoutePills(el,calls){el.append('PILL')}"
            + "function delegationChips(record,audits){const b=document.createElement('span');b.className='delegation-chips';return b}"
            + "const record={id:'r'},delegationMap=new Map([[0,['audit']]]),runIndex=0;"
            + segment
            + "console.log(JSON.stringify({"
            + "routesHasChip:routes.children.some(c=>c&&c.className==='delegation-chips'),"
            + "headerHasChip:header.children.some(c=>c&&c.className==='delegation-chips'),"
            + "headerHasRoutes:header.children.includes(routes)}));"
        )
        result = subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True)
        observed = json.loads(result.stdout)
        self.assertTrue(observed["routesHasChip"])
        self.assertFalse(observed["headerHasChip"])
        self.assertTrue(observed["headerHasRoutes"])


class RetiredWorkflowSwitchTests(DashboardProbeMixin, unittest.TestCase):
    """The Codex only / Codex + Claude switch is gone: the Claude model switches decide.

    A save still keeps the file's comments and flow lists, never writes
    ``workflow`` or ``claude_delegation.enabled``, and ignores a ``workflow`` an
    old open tab still posts.
    """

    ROUTER_YAML = (
        "# Router settings -- this comment must survive a dashboard save.\n"
        "enabled: true\n"
        "callable:\n"
        "  terra: true\n"
        "  opus5: true\n"
        "preferences:\n"
        "  review: [sonnet5, opus5, terra]  # Claude-tuned chain\n"
        "claude_delegation:\n"
        "  default_tier: sonnet\n"
    )

    def test_the_workflow_helpers_are_gone(self):
        for name in ("WORKFLOWS", "_workflow_name", "_save_workflow"):
            self.assertFalse(hasattr(web_viewer, name), name)

    def _serve(self, directory, calls):
        # Saves go to router_config.local.yaml; an empty shipped file under it keeps
        # every key an override, so the whole ROUTER_YAML round-trips there.
        config_path = Path(directory) / "router_config.yaml"
        config_path.write_text("{}\n", encoding="utf-8")
        local_path = config_path.with_name("router_config.local.yaml")
        local_path.write_text(self.ROUTER_YAML, encoding="utf-8")
        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=server.serve_forever, daemon=True)
        thread.start()
        results = []
        try:
            with patch.object(web_viewer, "CONFIG_PATH", config_path):
                for method, body in calls:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/config",
                        data=None if body is None else json.dumps(body).encode("utf-8"),
                        headers={"Content-Type": "application/json"}, method=method)
                    try:
                        with urllib.request.urlopen(request) as response:
                            results.append((response.status, json.load(response)))
                    except urllib.error.HTTPError as error:
                        results.append((error.code, json.load(error)))
        finally:
            server.shutdown(); server.server_close(); thread.join(timeout=2)
        return local_path.read_text(encoding="utf-8"), results

    def test_a_stale_workflow_in_a_post_is_ignored_and_the_files_comments_survive(self):
        with tempfile.TemporaryDirectory() as directory:
            text, results = self._serve(directory, [
                ("POST", {"workflow": "codex", "callable": {"terra": True, "opus5": False}}), ("GET", None)])
        self.assertEqual(results[0][0], 200)
        self.assertTrue(results[0][1]["success"])
        self.assertNotIn("workflow", results[1][1])
        self.assertIn("# Router settings -- this comment must survive a dashboard save.", text)
        self.assertIn("# Claude-tuned chain", text)
        self.assertNotIn("workflow", text)
        self.assertIn("opus5: false", text)
        self.assertIn("review: [sonnet5, opus5, terra]", text)

    def test_an_unknown_workflow_is_not_an_error_either(self):
        with tempfile.TemporaryDirectory() as directory:
            text, results = self._serve(directory, [("POST", {"workflow": "gemini"})])
        self.assertEqual(results[0][0], 200)
        self.assertNotIn("workflow", text)

    def test_a_full_page_save_keeps_the_comments_and_flow_lists(self):
        """Measured live 2026-09-19: the page posts callable and preferences on every
        save, and replacing those blocks wholesale dropped every comment attached to
        them -- including the ones above the next key -- and turned [a, b] into
        block lists. They are updated in place instead."""
        yaml_text = (
            "callable:\n"
            "  terra: true\n"
            "  sol: true\n"
            "  sonnet5: true\n"
            "  # opus note: on since 2026-09-09\n"
            "  opus5: true\n"
            "  qwen: true\n"
            "# Preferred models per kind of work\n"
            "preferences:\n"
            "  design:    [sol, opus5]\n"
            "  review:    [sonnet5, opus5, terra]\n"
            "#  chat:      [luna, spark]\n"
            "\n"
            "claude_delegation:\n"
            "  default_tier: sonnet\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            self.ROUTER_YAML, original = yaml_text, self.ROUTER_YAML
            try:
                text, results = self._serve(directory, [("POST", {
                    "callable": {"terra": True, "sol": True, "sonnet5": True, "opus5": True, "qwen": False},
                    "preferences": {"design": ["sol", "opus5"], "review": ["opus5", "terra"]},
                })])
            finally:
                self.ROUTER_YAML = original
        self.assertEqual(results[0][0], 200)
        for comment in ("# opus note: on since 2026-09-09", "# Preferred models per kind of work",
                        "#  chat:      [luna, spark]"):
            self.assertIn(comment, text)
        self.assertIn("qwen: false", text)
        self.assertRegex(text, r"design: +\[sol, opus5\]")
        self.assertRegex(text, r"review: +\[opus5, terra\]")

    def test_a_stale_delegation_flag_is_never_written(self):
        with tempfile.TemporaryDirectory() as directory:
            text, _ = self._serve(directory, [("POST", {
                "claude_delegation": {"enabled": False, "default_tier": "haiku"}})])
        import yaml
        self.assertEqual(yaml.safe_load(text)["claude_delegation"], {"default_tier": "haiku"})

    def _status(self, claude_on, registered):
        import model_router

        with tempfile.TemporaryDirectory() as directory:
            config, _, log_path = AccountsApiTests()._build_config(directory)
            log_path.write_text(json.dumps({"event": "registration", "registered": registered}), encoding="utf-8")
            for tier in ("opus5", "sonnet5", "haiku"):
                if tier in config["callable"]:
                    config["callable"][tier] = claude_on
            model_router.usage_guard._reset_cache()
            try:
                return web_viewer._accounts_status(config)["anthropic"]["delegation"]
            finally:
                model_router.usage_guard._reset_cache()

    def test_claude_switched_off_shows_delegation_off_and_never_asks_for_a_restart(self):
        delegation = self._status(False, registered=True)
        self.assertEqual((delegation["enabled"], delegation["restart_needed"]), (False, False))
        self.assertNotIn("workflow", delegation)

    def test_claude_switched_on_without_a_registered_tool_asks_for_a_restart(self):
        delegation = self._status(True, registered=False)
        self.assertEqual((delegation["enabled"], delegation["restart_needed"]), (True, True))

    def test_settings_has_no_workflow_section_and_balance_follows_the_accounts(self):
        start = HTML.index('id="settings-panel"')
        panel = HTML[start:HTML.index("</section>", start)]
        self.assertNotIn("workflow", panel.casefold())
        self.assertLess(panel.index('id="account-cards"'), panel.index('id="balance-section"'))
        self.assertIn('data-i18n="settings.balance.heading"', panel)
        for name in ("workflowControl",):
            self.assertNotIn(f"function {name}", HTML)
        self.assertNotIn("data-workflow", HTML)
        self.assertNotIn("getElementById('workflow-switch')", HTML)

    def test_the_claude_card_reports_delegation_off_from_the_switches(self):
        card_tests = AccountCardTests()
        off = dict(AccountCardTests.CLAUDE_INFO, delegation=dict(
            AccountCardTests.CLAUDE_INFO["delegation"], enabled=False))
        card = card_tests._card("anthropic", off)
        self.assertNotIn("data-account-toggle", card)
        self.assertIn("account.delegation.off", card)
        self.assertNotIn("account.delegation.live", card, "registered but switched off is not live")
        self.assertIn("data-default-tier", self.javascript_function("renderWorkers"))
        self.assertNotIn("Workflow", self.i18n("account.delegation.off")[0])
        self.assertNotIn("Munkafolyamat", self.i18n("account.delegation.off")[1])

    def test_saving_posts_neither_the_workflow_nor_the_delegation_flag(self):
        source = self.javascript_function("saveSettings")
        self.assertNotIn("workflow", source)
        self.assertIn("claude_delegation:(currentConfig.accounts||{}).anthropic?{default_tier:", source)
        self.assertNotIn("delegation.enabled", source)

    def test_the_preferences_no_longer_pause_for_a_workflow(self):
        source = self.javascript_function("renderPreferences")
        self.assertNotIn("settings.prefs.paused", source)
        self.assertNotIn("workflow", source)
        self.assertNotIn("pref-paused", HTML)

    def test_no_workflow_i18n_key_is_left_in_either_language(self):
        import re
        self.assertEqual(re.findall(r"'settings\.workflow\.[^']*'", HTML), [])
        self.assertNotIn("'settings.prefs.paused'", HTML)
        for key in ("settings.balance.heading", "account.delegation.off", "settings.sub"):
            english, hungarian = self.i18n(key)
            self.assertNotIn("Workflow", english)
            self.assertNotIn("Munkafolyamat", hungarian)
        self.assertEqual(self.i18n("settings.balance.heading"), ("Load balancing", "Terheléselosztás"))


class HermesParentGuardTests(unittest.TestCase):
    """Terra's review of 992d706.

    1. Whether Hermes's parent is the router's own was decided by model name alone,
       so the same model name served by a different provider counted as the router's.
       Name and provider must both match a router tier.
    2. The guard and the write read ~/.hermes/config.yaml separately, so a change
       between the two could be overwritten. One read per save, and a write refuses
       when the file changed since that read.
    """

    CONFIG = {
        "models": {"terra": "gpt-5.6-terra", "luna": "gpt-6-luna", "qwen": "qwen3.7-plus"},
        "callable": {"terra": True, "luna": True, "qwen": True},
        "tier_providers": {"terra": "openai-codex", "luna": "openai-codex", "qwen": "qwen-token"},
        "default_model": "terra",
    }

    def _config(self):
        import copy

        return copy.deepcopy(self.CONFIG)

    def _hermes(self, directory, default, provider):
        target = Path(directory) / "config.yaml"
        target.write_text(f"model:\n  default: {default}\n  provider: {provider}\n", encoding="utf-8")
        return target

    def test_a_router_model_name_on_another_provider_is_not_the_routers_parent(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes(directory, "gpt-5.6-terra", "openrouter")
            before = target.read_text(encoding="utf-8")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                self.assertIsNone(web_viewer._save_default_model("luna", config))
            self.assertEqual(target.read_text(encoding="utf-8"), before)
        self.assertEqual(config["default_model"], "luna")

    def test_a_parent_on_its_own_tiers_provider_still_follows_the_default(self):
        """A Qwen parent on qwen-token is one the router put there (default_model: qwen
        writes exactly that), so it still follows the router's default tier."""
        config = self._config()
        config["default_model"] = "qwen"
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes(directory, "qwen3.7-plus", "qwen-token")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                self.assertIsNone(web_viewer._save_default_model("terra", config))
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
        self.assertEqual((written["model"]["default"], written["model"]["provider"]),
                         ("gpt-5.6-terra", "openai-codex"))

    def test_a_parent_with_no_provider_is_judged_by_its_name(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = Path(directory) / "config.yaml"
            target.write_text("model:\n  default: gpt-5.6-terra\n", encoding="utf-8")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                self.assertIsNone(web_viewer._save_default_model("luna", config))
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
        self.assertEqual(written["model"]["default"], "gpt-6-luna")

    def test_a_default_model_save_reads_the_hermes_config_once(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes(directory, "gpt-5.6-terra", "openai-codex")
            real = web_viewer._read_hermes_config
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target), \
                 patch.object(web_viewer, "_read_hermes_config", side_effect=real) as reads:
                self.assertIsNone(web_viewer._save_default_model("luna", config))
        self.assertEqual(reads.call_count, 1)

    def _changing_read(self, target, change):
        real = web_viewer._read_hermes_config

        def read():
            snapshot = real()
            target.write_text(change, encoding="utf-8")
            import os
            os.utime(target, ns=(target.stat().st_atime_ns, target.stat().st_mtime_ns + 1_000_000))
            return snapshot
        return read

    def test_a_hermes_config_changed_mid_save_is_not_overwritten(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes(directory, "gpt-5.6-terra", "openai-codex")
            concurrent = "model:\n  default: claude-opus-5-5\n  provider: anthropic\n"
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target), \
                 patch.object(web_viewer, "_read_hermes_config", side_effect=self._changing_read(target, concurrent)):
                error = web_viewer._save_default_model("luna", config)
            self.assertIn("changed", error)
            self.assertEqual(target.read_text(encoding="utf-8"), concurrent)
            self.assertEqual(list(Path(directory).glob("config.yaml.bak-router-*")), [])
        self.assertEqual(config["default_model"], "terra", "a refused save stores nothing")

    def test_a_fallback_chain_save_refuses_the_same_way(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes(directory, "claude-opus-5-5", "anthropic")
            concurrent = "model:\n  default: claude-opus-5-5\n  provider: anthropic\nfallback_providers: []\n"
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target), \
                 patch.object(web_viewer, "_read_hermes_config", side_effect=self._changing_read(target, concurrent)), \
                 patch.object(web_viewer, "_hermes_chain", return_value=[]):
                error = web_viewer._save_hermes_fallback({"orchestrator": []}, self._config())
            self.assertIn("changed", error)
            self.assertEqual(target.read_text(encoding="utf-8"), concurrent)

    def test_a_failed_hermes_write_fails_the_save_and_stores_nothing(self):
        """Codex review of 992d706: the write error was swallowed, the router saved the
        new default and answered success while Hermes stayed on the old parent."""
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes(directory, "gpt-5.6-terra", "openai-codex")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target), \
                 patch.object(web_viewer, "_write_hermes_config", side_effect=PermissionError("read-only")):
                error = web_viewer._save_default_model("luna", config)
        self.assertIn("read-only", error)
        self.assertEqual(config["default_model"], "terra")


class BalanceSwitchTests(DashboardProbeMixin, unittest.TestCase):
    """The load-balancing on/off switch, inside the Workflow block, Claude delegation only."""

    def test_save_balance_writes_only_the_enabled_flag(self):
        config = {"usage_guard": {"balance": {"enabled": True, "busy_percent": 55, "margin_percent": 30}}}
        self.assertIsNone(web_viewer._save_balance({"enabled": False}, config))
        self.assertEqual(config["usage_guard"]["balance"], {"enabled": False, "busy_percent": 55, "margin_percent": 30})

    def test_save_balance_creates_the_block_when_absent(self):
        config = {"usage_guard": {"accounts": {}}}
        self.assertIsNone(web_viewer._save_balance({"enabled": True}, config))
        self.assertEqual(config["usage_guard"]["balance"], {"enabled": True})

    def test_save_balance_refuses_a_non_boolean(self):
        config = {}
        self.assertIn("true or false", web_viewer._save_balance({"enabled": "yes"}, config))
        self.assertEqual(config, {})

    def test_save_balance_stores_the_thresholds_as_the_file_spells_them(self):
        config = {"usage_guard": {"balance": {"enabled": True}}}
        self.assertIsNone(web_viewer._save_balance(
            {"enabled": True, "busy_percent": 25, "margin_percent": 12.5}, config))
        balance = config["usage_guard"]["balance"]
        self.assertEqual((repr(balance["busy_percent"]), repr(balance["margin_percent"])), ("25", "12.5"))

    def test_save_balance_refuses_thresholds_out_of_range(self):
        for busy, margin in ((-1, 10), (101, 10), (20, 0), (20, 101), ("x", 10)):
            config = {"usage_guard": {"balance": {"enabled": True}}}
            self.assertIsNotNone(web_viewer._save_balance(
                {"enabled": True, "busy_percent": busy, "margin_percent": margin}, config), (busy, margin))
            self.assertEqual(config["usage_guard"]["balance"], {"enabled": True})

    def test_the_api_serves_the_balance_settings(self):
        self.assertEqual(web_viewer._balance_status({"usage_guard": {"balance": {"enabled": True}}}),
                         {"enabled": True, "busy_percent": 20.0, "margin_percent": 10.0, "window": "5-hour"})
        self.assertFalse(web_viewer._balance_status({})["enabled"])

    def _control(self, balance):
        script = (self.javascript_function("balanceControl")
                  + f"\nconsole.log(balanceControl({json.dumps(balance)}));")
        return subprocess.run(["node", "-e", self.i18n_runtime() + "\n" + script],
                              check=True, text=True, capture_output=True).stdout

    def _shown(self, accounts, callable_):
        script = (self.javascript_function("balanceAccounts")
                  + f"\nconsole.log(balanceAccounts({json.dumps(accounts)},{json.dumps(callable_)}));")
        return int(subprocess.run(["node", "-e", script], check=True, text=True, capture_output=True).stdout)

    def test_the_switch_shows_with_editable_thresholds(self):
        out = self._control({"enabled": True, "busy_percent": 20, "margin_percent": 10, "window": "5-hour"})
        self.assertRegex(out, r'<label class="switch"><input type="checkbox" data-balance-toggle checked>')
        self.assertIn('data-balance-field="busy_percent" value="20"', out)
        self.assertIn('data-balance-field="margin_percent" value="10"', out)
        self.assertIn("5-hour", out)

    def test_each_threshold_sits_on_one_line_with_its_label_and_unit(self):
        """The global `label{display:grid}` stacked "from", the box and "%" vertically,
        so the two fields sat at different heights (seen on the dashboard)."""
        self.assertIn(".balance-row .balance-field{display:inline-flex;align-items:center", HTML)

    def test_the_switch_is_off_when_balancing_is_off(self):
        out = self._control({"enabled": False, "busy_percent": 20, "margin_percent": 10, "window": "5-hour"})
        self.assertIn("data-balance-toggle", out)
        self.assertNotIn("data-balance-toggle checked", out)

    def test_the_section_needs_two_accounts_with_a_model_switched_on(self):
        accounts = {"openai-codex": {"tiers": ["terra", "sol"]}, "anthropic": {"tiers": ["opus5", "haiku"]},
                    "xai-oauth": {"tiers": ["grok"]}}
        self.assertEqual(self._shown(accounts, {"grok": False, "opus5": False, "haiku": False}), 1)
        self.assertEqual(self._shown(accounts, {"grok": False, "opus5": False, "haiku": True}), 2)
        self.assertEqual(self._shown(accounts, {"grok": True, "opus5": False, "haiku": False}), 2)
        source = self.javascript_function("renderBalance")
        self.assertIn(">=2", source)
        self.assertIn("section.hidden=!shown", source)

    def test_saving_posts_the_balance_flag_and_thresholds(self):
        source = self.javascript_function("saveSettings")
        self.assertIn("balance:", source)
        self.assertIn("busy_percent:", source)
        self.assertIn("margin_percent:", source)

    def test_its_i18n_keys_exist_in_both_languages(self):
        for key in ("settings.balance.heading", "settings.balance.label", "settings.balance.desc", "settings.balance.busy",
                    "settings.balance.margin", "settings.balance.window.5-hour", "settings.balance.window.tighter"):
            self.i18n(key)


class RefreshButtonUsageTests(DashboardProbeMixin, unittest.TestCase):
    """The toolbar Refresh button only ever re-read /api/entries and /api/agents,
    so the usage % bars -- drawn by loadAccounts() off /api/config -- never moved
    on a click. /api/config serves guard.cached(), so a re-render alone is not
    enough either: a click has to force the live read that /api/usage/refresh does.
    """

    def test_a_dashboard_refresh_redraws_the_usage_bars(self):
        self.assertIn("loadAccounts()", self.javascript_function("refreshDashboard"))

    def test_a_click_forces_a_live_usage_read_first(self):
        source = self.javascript_function("manualRefresh")
        self.assertIn("refreshAllUsage()", source)
        self.assertIn("refreshDashboard()", source)

    def test_the_live_read_posts_to_the_usage_refresh_endpoint(self):
        source = self.javascript_function("refreshAllUsage")
        self.assertIn("/api/usage/refresh?account=", source)
        self.assertIn("method:'POST'", source)
        # Accounts with no usage source have nothing to read; asking 400s.
        self.assertIn("has_usage_source", source)

    def test_the_click_is_wired_to_the_manual_path_and_the_timer_is_not(self):
        wiring = HTML[HTML.index("$('refresh').addEventListener"):]
        wiring = wiring[:wiring.index("\n")]
        self.assertIn("manualRefresh", wiring)
        timer = HTML[HTML.index("if($('auto').checked)"):]
        self.assertNotIn("manualRefresh", timer[:timer.index("\n")])

    def test_the_handlers_never_hand_their_event_to_refreshDashboard(self):
        """addEventListener passes an Event, which is truthy -- a bare function
        reference would make any future argument silently arrive as the event."""
        for wiring in ("$('last').addEventListener('change',", "$('refresh').addEventListener('click',"):
            line = HTML[HTML.index(wiring) + len(wiring):]
            self.assertTrue(line.startswith("()=>"), f"{wiring} passes the event through")


class ReasoningEffortSettingsTests(DashboardProbeMixin, unittest.TestCase):
    """The dashboard owns the four plain routed tiers, not route-marker or situational effort keys."""

    CONFIG = {
        "models": {
            "luna": "gpt-6-luna", "spark": "gpt-5.3-codex-spark",
            "terra": "gpt-5.6-terra", "sol": "gpt-6-sol", "grok": "grok-4.7",
        },
        "callable": {"luna": True, "spark": True, "terra": True, "sol": True, "grok": False},
        "tier_providers": {"luna": "openai-codex", "spark": "openai-codex", "terra": "openai-codex", "sol": "openai-codex", "grok": "xai-oauth"},
        "effort": {
            "luna": "low", "spark": "medium", "terra": "medium", "sol": "medium", "grok": "medium",
            "opus5": "external", "sol_long": "medium", "explicit_sol": "medium",
            "explicit_sol_xhigh": "medium", "explicit_luna_xhigh": "high",
            "explicit_spark_xhigh": "high", "explicit_terra_xhigh": "high",
        },
    }

    def _write_config(self, directory):
        target = Path(directory) / "router_config.yaml"
        with open(target, "w", encoding="utf-8") as f:
            web_viewer.yaml.dump(self.CONFIG, f)
        return target

    def _request(self, config_path, method, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        # A save that moves the parent writes the Hermes config: keep it beside the
        # router config. Unpatched, a run wrote a Terra stub over the real
        # ~/.hermes/config.yaml whenever HERMES_HOME pointed there.
        with patch.object(web_viewer, "CONFIG_PATH", config_path), \
             patch.object(web_viewer, "HERMES_CONFIG_PATH", config_path.with_name("hermes-config.yaml")), \
             patch.object(web_viewer, "_router_module", return_value=None):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/api/config", data=data, method=method,
                    headers={"Content-Type": "application/json"} if data is not None else {},
                )
                try:
                    with urllib.request.urlopen(request) as response:
                        return response.status, json.load(response)
                except urllib.error.HTTPError as error:
                    return error.code, json.load(error)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_get_serves_only_the_dashboard_managed_effort_keys(self):
        with tempfile.TemporaryDirectory() as directory:
            status, payload = self._request(self._write_config(directory), "GET")
        self.assertEqual(status, 200)
        self.assertEqual(payload["effort"], {"luna": "low", "spark": "medium", "terra": "medium", "sol": "medium", "grok": "medium"})
        self.assertNotIn("opus5", payload["effort"])

    def test_a_valid_effort_save_writes_only_the_local_delta_and_preserves_the_shipped_file(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            shipped_before = config_path.read_bytes()
            status, body = self._request(config_path, "POST", {"effort": {"grok": " HIGH "}})
            local = config_path.with_name("router_config.local.yaml")
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8"))
            self.assertEqual(config_path.read_bytes(), shipped_before)
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])
        self.assertIn("revision", body)
        self.assertEqual(written, {"effort": {"grok": "high"}})

    def test_an_invalid_grok_effort_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            shipped_before = config_path.read_bytes()
            status, body = self._request(config_path, "POST", {"effort": {"grok": "external"}})
            self.assertFalse(config_path.with_name("router_config.local.yaml").exists())
            self.assertEqual(config_path.read_bytes(), shipped_before)
        self.assertEqual(status, 400)
        self.assertIn("effort.grok must be one of: low, medium, high, xhigh", body["error"])

    def test_a_full_page_save_does_not_pin_absent_grok_at_its_effective_default(self):
        """Older shipped configs lack ``effort.grok`` while the frontend posts all
        visible levels. An unrelated switch must not turn the effective medium
        default into a permanent local override."""
        older = json.loads(json.dumps(self.CONFIG))
        for key in ("models", "callable", "tier_providers", "effort"):
            older[key].pop("grok", None)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "router_config.yaml"
            config_path.write_text(web_viewer.yaml.safe_dump(older), encoding="utf-8")
            payload = {
                "callable": {**older["callable"], "luna": False},
                "default_model": "terra",
                "effort": {"luna": "low", "spark": "medium", "terra": "medium", "sol": "medium", "grok": "medium"},
                "claude_reasoning_effort": {},
                "preferences": {},
                "hermes_fallback": {},
                "usage_limits": {},
            }
            with patch.object(web_viewer, "_read_hermes_snapshot", return_value=(None, {
                "model": {"default": "gpt-5.6-terra", "provider": "openai-codex"},
            })), patch.object(web_viewer, "_hermes_chain", return_value=[]):
                real_hermes = web_viewer.HERMES_CONFIG_PATH
                before = real_hermes.read_bytes() if real_hermes.exists() else None
                status, body = self._request(config_path, "POST", payload)
                after = real_hermes.read_bytes() if real_hermes.exists() else None
            self.assertEqual(after, before, "the save wrote the real Hermes config")
            local = config_path.with_name("router_config.local.yaml")
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8")) if local.exists() else {}
        self.assertEqual(status, 200, body)
        self.assertTrue(body["success"])
        self.assertEqual(written.get("callable"), {"luna": False})
        self.assertNotIn("grok", written.get("effort", {}))

    def test_a_non_object_effort_payload_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            shipped_before = config_path.read_bytes()
            status, body = self._request(config_path, "POST", {"effort": ["high"]})
            local = config_path.with_name("router_config.local.yaml")
            self.assertFalse(local.exists())
            self.assertEqual(config_path.read_bytes(), shipped_before)
        self.assertEqual(status, 400)
        self.assertIn("object", body["error"])

    def test_unmanaged_effort_keys_are_refused_without_writing(self):
        for key in ("opus5", "qwen", "haiku", "sonnet5", "sol_long", "explicit_sol", "explicit_spark_xhigh"):
            with self.subTest(key=key), tempfile.TemporaryDirectory() as directory:
                config_path = self._write_config(directory)
                shipped_before = config_path.read_bytes()
                status, body = self._request(config_path, "POST", {"effort": {key: "high"}})
                self.assertEqual(status, 400)
                self.assertIn(key, body["error"])
                self.assertFalse(config_path.with_name("router_config.local.yaml").exists())
                self.assertEqual(config_path.read_bytes(), shipped_before)

    def test_non_string_and_blank_effort_values_are_refused_without_writing(self):
        for value in (True, "   "):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                config_path = self._write_config(directory)
                shipped_before = config_path.read_bytes()
                status, body = self._request(config_path, "POST", {"effort": {"terra": value}})
                self.assertEqual(status, 400)
                self.assertIn("terra", body["error"])
                self.assertFalse(config_path.with_name("router_config.local.yaml").exists())
                self.assertEqual(config_path.read_bytes(), shipped_before)

    def test_external_and_unknown_effort_values_are_refused_without_writing(self):
        for value in ("external", "max"):
            with self.subTest(value=value), tempfile.TemporaryDirectory() as directory:
                config_path = self._write_config(directory)
                shipped_before = config_path.read_bytes()
                status, body = self._request(config_path, "POST", {"effort": {"terra": value}})
                self.assertEqual(status, 400)
                self.assertIn("low, medium, high, xhigh", body["error"])
                self.assertFalse(config_path.with_name("router_config.local.yaml").exists())
                self.assertEqual(config_path.read_bytes(), shipped_before)

    def test_an_unrelated_existing_local_override_survives_an_effort_save(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            local = config_path.with_name("router_config.local.yaml")
            local.write_text("fallbacks:\n  sol: terra\n", encoding="utf-8")
            status, _ = self._request(config_path, "POST", {"effort": {"spark": "xhigh"}})
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8"))
        self.assertEqual(status, 200)
        self.assertEqual(written, {"fallbacks": {"sol": "terra"}, "effort": {"spark": "xhigh"}})

    def test_the_reasoning_effort_control_has_all_of_its_i18n_keys_in_both_languages(self):
        for key in ("settings.effort.low", "settings.effort.medium", "settings.effort.high", "settings.effort.xhigh"):
            self.i18n(key)

    def test_the_control_offers_exactly_the_four_shared_effort_levels(self):
        source = self.javascript_function("renderEffort") or ""
        self.assertIn("const ROUTER_EFFORT_TIERS=['luna','spark','terra','sol','grok']", HTML)
        self.assertIn("tiers.filter(tier=>ROUTER_EFFORT_TIERS.includes(tier))", source)
        self.assertNotIn("opus5", source)
        self.assertNotIn("sol_long", source)
        for level in ("low", "medium", "high", "xhigh"):
            self.assertEqual(self.i18n("settings.effort." + level), (level, level))

    def test_a_later_invalid_effort_entry_leaves_earlier_entries_unchanged(self):
        config = {"effort": {"luna": "low", "terra": "medium"}}
        before = json.loads(json.dumps(config))
        error = web_viewer._save_effort({"luna": "high", "opus5": "external"}, config)
        self.assertIsNotNone(error)
        self.assertIn("opus5", error)
        self.assertEqual(config, before)

    def test_get_never_returns_null_for_a_managed_effort_tier_missing_from_config(self):
        """C-M3: with the shipped ``effort.grok`` key removed, GET must still report a
        string (the effective default), not null; a full-page save posting that value
        back must succeed and write no grok key."""
        older = json.loads(json.dumps(self.CONFIG))
        for key in ("models", "callable", "tier_providers", "effort"):
            older[key].pop("grok", None)
        with tempfile.TemporaryDirectory() as directory:
            config_path = Path(directory) / "router_config.yaml"
            config_path.write_text(web_viewer.yaml.safe_dump(older), encoding="utf-8")
            status, payload = self._request(config_path, "GET")
            self.assertEqual(status, 200)
            self.assertIn("grok", payload["effort"])
            self.assertIsNotNone(payload["effort"]["grok"])
            self.assertIsInstance(payload["effort"]["grok"], str)

            full_payload = {
                "callable": older["callable"],
                "default_model": "terra",
                "effort": dict(payload["effort"]),
                "claude_reasoning_effort": {},
                "preferences": {},
                "hermes_fallback": {},
                "usage_limits": {},
            }
            with patch.object(web_viewer, "_read_hermes_snapshot", return_value=(None, {
                "model": {"default": "gpt-5.6-terra", "provider": "openai-codex"},
            })), patch.object(web_viewer, "_hermes_chain", return_value=[]):
                status, body = self._request(config_path, "POST", full_payload)
            self.assertEqual(status, 200, body)
            self.assertTrue(body["success"])
            local = config_path.with_name("router_config.local.yaml")
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8")) if local.exists() else {}
            self.assertNotIn("grok", written.get("effort", {}))

    def test_save_effort_ignores_a_none_value_instead_of_erroring(self):
        """C-M3: ``_save_effort`` must skip a posted ``None`` for a managed tier (the
        shape GET can produce for an unmanaged/absent value) rather than rejecting the
        whole payload."""
        config = {"effort": {"luna": "low", "terra": "medium"}}
        before = json.loads(json.dumps(config))
        error = web_viewer._save_effort({"grok": None, "terra": "high"}, config)
        self.assertIsNone(error)
        self.assertEqual(config["effort"]["terra"], "high")
        self.assertNotIn("grok", config["effort"])
        self.assertEqual(config["effort"]["luna"], before["effort"]["luna"])

    def test_effort_status_never_returns_null_even_without_the_router_module(self):
        """C2-N1: _DEFAULT_MANAGED_EFFORT must carry every managed tier's shipped
        default, so _effort_status still promises never-null when the router
        module itself is not importable (not just when a tier is merely absent
        from an importable router's config)."""
        config = {"effort": {"luna": "low"}}
        with patch.object(web_viewer, "_router_module", return_value=None):
            status = web_viewer._effort_status(config)
        for tier in web_viewer.MANAGED_EFFORT_TIERS:
            with self.subTest(tier=tier):
                self.assertIsNotNone(status[tier])
                self.assertIsInstance(status[tier], str)


class FullPageFilterTests(unittest.TestCase):
    def test_controls_update_visible_cards_and_counts_together(self):
        import shutil
        if not shutil.which("node"):
            self.fail("Full-page dashboard test requires Node.js")
        probe = subprocess.run(["node", "-e", "require.resolve('jsdom')"], capture_output=True)
        if probe.returncode:
            self.fail("Full-page dashboard test requires jsdom; set NODE_PATH to its node_modules directory")
        script = r"""
const {JSDOM,VirtualConsole}=require('jsdom');
const assert=require('node:assert/strict');
const html=require('node:fs').readFileSync(0,'utf8');
const errors=[]; const vc=new VirtualConsole(); vc.on('jsdomError',e=>errors.push(e.message));
const dom=new JSDOM(html,{url:'http://localhost/',runScripts:'dangerously',virtualConsole:vc,
 beforeParse(w){w.fetch=()=>new Promise(()=>{});w.setInterval=()=>0;}});
const w=dom.window,d=w.document;
w.eval(`entries=[
 {timestamp:'2026-09-25T10:00:00Z',turn_id:'alpha:1',tier:'terra',model:'gpt-terra',prompt_preview:'alpha task'},
 {timestamp:'2026-09-25T10:01:00Z',turn_id:'beta:1',tier:'sol',model:'gpt-sol',prompt_preview:'beta task'}
];agentActivity={parents:[],active_turns:[]};render();`);
d.getElementById('tier').innerHTML='<option value=""></option><option value="sol">Sol</option>';
const cards=()=>d.querySelectorAll('#runs .router-run').length;
assert.equal(cards(),2);
d.getElementById('search').value='alpha';
d.getElementById('search').dispatchEvent(new w.Event('input'));
assert.equal(cards(),1);assert.match(d.getElementById('runs').textContent,/alpha task/);
assert.equal(d.getElementById('total').textContent,'1');
d.getElementById('search').value='';d.getElementById('tier').value='sol';
d.getElementById('tier').dispatchEvent(new w.Event('input'));
assert.equal(cards(),1);assert.match(d.getElementById('runs').textContent,/beta task/);
d.getElementById('grouped').checked=false;
d.getElementById('grouped').dispatchEvent(new w.Event('input'));
assert.equal(cards(),1);assert.equal(d.getElementById('total').textContent,'1');
assert.deepEqual(errors,[]);dom.window.close();
"""
        result = subprocess.run(["node", "-e", script], input=HTML, text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class SettingsSaveQueueTests(DashboardProbeMixin, unittest.TestCase):
    def test_worker_model_joins_the_two_file_settings_transaction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            router = root / 'router_config.yaml'
            hermes = root / 'config.yaml'
            router.write_text('default_model: terra\nprovider: openai-codex\nmodels:\n  terra: gpt-terra\n  sol: gpt-sol\ntier_providers:\n  terra: openai-codex\n  sol: openai-codex\ncallable:\n  terra: true\n  sol: true\n', encoding='utf-8')
            original = 'model:\n  provider: openai-codex\n  default: gpt-terra\ndelegation:\n  provider: openai-codex\n  model: gpt-terra\n'
            hermes.write_text(original, encoding='utf-8')
            local = root / 'router_config.local.yaml'
            with patch.object(web_viewer, 'CONFIG_PATH', router), patch.object(web_viewer, 'HERMES_CONFIG_PATH', hermes):
                status, result = web_viewer._save_config_payload({
                    'revision': web_viewer._config_revision(), 'worker_model': 'sol',
                    'callable': {'terra': True, 'sol': False}})
                self.assertEqual(status, 200, result)
                self.assertEqual(web_viewer.yaml.safe_load(hermes.read_text())['delegation']['model'], 'gpt-sol')
                self.assertIn('sol: false', local.read_text())
                self.assertEqual(result['revision'], web_viewer._config_revision())
                hermes.write_text(original, encoding='utf-8')
                local.unlink()
                original_write = web_viewer._atomic_write
                def fail_local(path, content):
                    if path == local:
                        raise OSError('local write failed')
                    return original_write(path, content)
                with patch.object(web_viewer, '_atomic_write', side_effect=fail_local), \
                     self.assertRaisesRegex(OSError, 'local write failed'):
                    web_viewer._save_config_payload({'worker_model': 'sol',
                        'callable': {'terra': True, 'sol': False}})
                self.assertEqual(hermes.read_text(), original)
                self.assertFalse(local.exists())

    def test_failed_save_drops_queued_edits_without_hiding_error_then_reload_recovers(self):
        script = r"""
const assert=require('node:assert/strict');
let currentConfig={revision:'r0',callable:{terra:true},accounts:{}};
let settingsSaveQueue=Promise.resolve(),settingsPending=0,settingsSaveFailed=false,settingsLoadGeneration=0;
const status={textContent:'',style:{}};
const $=()=>status,t=k=>k,requests=[],releases=[];
const fetch=(_url,options)=>{requests.push(JSON.parse(options.body));return new Promise(resolve=>releases.push(resolve))};
const renderSettings=()=>{};
""" + self.javascript_function('saveSettings') + 'async ' + self.javascript_function('loadSettings').replace(
    "const response=await fetch('/api/config',{cache:'no-store'});", "const response={ok:true,json:async()=>({revision:'r2',callable:{terra:false},accounts:{}})};") + r"""
(async()=>{
 const first=saveSettings();currentConfig.callable.terra=false;const second=saveSettings();
 await Promise.resolve();assert.equal(requests.length,1);
 releases[0]({ok:false,status:409,json:async()=>({error:'Settings changed'})});
 await first;const error=status.textContent;await second;
 assert.match(error,/Settings changed/);assert.equal(status.textContent,error);
 assert.equal(requests.length,1);assert.equal(settingsPending,0);assert.equal(settingsSaveFailed,true);
 await loadSettings();assert.equal(settingsSaveFailed,false);
 const third=saveSettings();await Promise.resolve();assert.equal(requests.length,2);
 assert.equal(requests[1].revision,'r2');
 releases[1]({ok:true,json:async()=>({success:true,revision:'r3'})});await third;
 assert.equal(currentConfig.revision,'r3');
})().catch(e=>{console.error(e);process.exit(1)});
"""
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


    def test_second_page_stale_revision_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'router_config.yaml'
            hermes = root / 'config.yaml'
            config.write_text('callable:\n  terra: true\n', encoding='utf-8')
            hermes.write_text('model: gpt-terra\n', encoding='utf-8')
            with patch.object(web_viewer, 'CONFIG_PATH', config), patch.object(web_viewer, 'HERMES_CONFIG_PATH', hermes):
                stale = web_viewer._config_revision()
                status, first = web_viewer._save_config_payload({'revision': stale, 'callable': {'terra': False}})
                self.assertEqual(status, 200)
                written = config.with_name('router_config.local.yaml').read_bytes()
                status, second = web_viewer._save_config_payload({'revision': stale, 'callable': {'terra': True}})
                self.assertEqual(status, 409)
                self.assertIn('Reload', second['error'])
                self.assertEqual(config.with_name('router_config.local.yaml').read_bytes(), written)
                self.assertNotEqual(first['revision'], stale)

    def test_external_hermes_write_after_first_file_preserves_external_change(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'router_config.yaml'
            hermes = root / 'config.yaml'
            config.write_text('default_model: terra\nmodels:\n  terra: gpt-terra\n  sol: gpt-sol\ncallable:\n  terra: true\n  sol: true\n', encoding='utf-8')
            hermes.write_text('provider: openai-codex\nmodel: gpt-terra\n', encoding='utf-8')
            original_write = web_viewer._atomic_write
            def competing_write(path, content):
                if path == config.with_name('router_config.local.yaml'):
                    hermes.write_text('provider: anthropic\nmodel: claude-sonnet-5-5\n', encoding='utf-8')
                    raise OSError('router disk write failed')
                return original_write(path, content)
            def change_default(_requested, _config, *, hermes, persist):
                hermes['model'] = 'gpt-sol'
                return None
            with patch.object(web_viewer, 'CONFIG_PATH', config), patch.object(web_viewer, 'HERMES_CONFIG_PATH', hermes), \
                 patch.object(web_viewer, '_atomic_write', side_effect=competing_write), \
                 patch.object(web_viewer, '_save_default_model', side_effect=change_default):
                with self.assertRaisesRegex(RuntimeError, 'automatic rollback refused'):
                    web_viewer._save_config_payload({'default_model': 'sol', 'callable': {'terra': False, 'sol': True}})
            self.assertIn('claude-sonnet-5-5', hermes.read_text(encoding='utf-8'))
            self.assertFalse(config.with_name('router_config.local.yaml').exists())

    def test_delayed_get_does_not_replace_a_newer_local_edit(self):
        script = r"""
const assert=require('node:assert/strict');
let currentConfig={revision:'r0',callable:{terra:true},accounts:{}};
let settingsSaveQueue=Promise.resolve(),settingsPending=0,settingsSaveFailed=false,settingsLoadGeneration=0;
const status={textContent:'',style:{}};
const $=()=>status,t=k=>k;let releaseGet,releaseSave,renders=0;
const renderSettings=()=>{renders++};
const fetch=(_url,options)=>options?.method==='POST'?new Promise(resolve=>releaseSave=resolve):
  new Promise(resolve=>releaseGet=resolve);
""" + self.javascript_function('saveSettings') + 'async ' + self.javascript_function('loadSettings') + r"""
(async()=>{
 const stale=loadSettings();currentConfig.callable.terra=false;const save=saveSettings();
 await Promise.resolve();
 releaseGet({ok:true,json:async()=>({revision:'old',callable:{terra:true}})});
 await stale;assert.equal(currentConfig.callable.terra,false);assert.equal(currentConfig.revision,'r0');
 assert.equal(renders,0);
 releaseSave({ok:true,json:async()=>({success:true,revision:'r1'})});await save;
 assert.equal(currentConfig.revision,'r1');
})().catch(e=>{console.error(e);process.exit(1)});
"""
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_rapid_edits_send_serial_snapshots_using_latest_revision(self):
        script = r"""
const assert=require('node:assert/strict');
let currentConfig={revision:'r0',callable:{terra:true},accounts:{}};
let settingsSaveQueue=Promise.resolve(),settingsPending=0,settingsSaveFailed=false,settingsLoadGeneration=0;
const $=()=>({style:{}}),t=k=>k,requests=[],releases=[];
const fetch=(_url,options)=>{requests.push(JSON.parse(options.body));return new Promise(resolve=>releases.push(resolve))};
""" + self.javascript_function('saveSettings') + r"""
(async()=>{
 const first=saveSettings();
 currentConfig.callable.terra=false;
 const second=saveSettings();
 await Promise.resolve();
 assert.equal(requests.length,1);
 assert.equal(requests[0].callable.terra,true);
 releases[0]({ok:true,json:async()=>({success:true,revision:'r1'})});
 await first;await Promise.resolve();
 assert.equal(requests.length,2);
 assert.equal(requests[1].revision,'r1');
 assert.equal(requests[1].callable.terra,false);
 releases[1]({ok:true,json:async()=>({success:true,revision:'r2'})});
 await second;
 assert.equal(currentConfig.revision,'r2');assert.equal(settingsPending,0);
})().catch(e=>{console.error(e);process.exit(1)});
"""
        result = subprocess.run(['node', '-e', script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)


class ClaudeReasoningEffortSettingsTests(DashboardProbeMixin, unittest.TestCase):
    """The dashboard edits only the two effective Claude tiers; Haiku is explicitly unsupported."""

    CONFIG = {
        "models": {"luna": "gpt-6-luna"},
        "callable": {"luna": True},
        "claude_delegation": {"enabled": True, "default_tier": "sonnet"},
    }

    def setUp(self):
        self._reset_claude_reasoning_tracebacks()
        self.addCleanup(self._reset_claude_reasoning_tracebacks)

    @staticmethod
    def _reset_claude_reasoning_tracebacks():
        lock = getattr(web_viewer, "_CLAUDE_REASONING_TRACEBACK_LOCK", None)
        if lock is None:
            web_viewer._CLAUDE_REASONING_TRACEBACKS.clear()
        else:
            with lock:
                web_viewer._CLAUDE_REASONING_TRACEBACKS.clear()

    def _write_config(self, directory):
        target = Path(directory) / "router_config.yaml"
        with open(target, "w", encoding="utf-8") as f:
            web_viewer.yaml.dump(self.CONFIG, f)
        return target

    def _request(self, config_path, method, payload=None):
        data = None if payload is None else json.dumps(payload).encode("utf-8")
        # A save that moves the parent writes the Hermes config: keep it beside the
        # router config. Unpatched, a run wrote a Terra stub over the real
        # ~/.hermes/config.yaml whenever HERMES_HOME pointed there.
        with patch.object(web_viewer, "CONFIG_PATH", config_path), \
             patch.object(web_viewer, "HERMES_CONFIG_PATH", config_path.with_name("hermes-config.yaml")), \
             patch.object(web_viewer, "_router_module", return_value=None):
            server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
            thread = threading.Thread(target=server.serve_forever, daemon=True)
            thread.start()
            try:
                request = urllib.request.Request(
                    f"http://127.0.0.1:{server.server_port}/api/config", data=data, method=method,
                    headers={"Content-Type": "application/json"} if data is not None else {},
                )
                try:
                    with urllib.request.urlopen(request) as response:
                        return response.status, json.load(response)
                except urllib.error.HTTPError as error:
                    return error.code, json.load(error)
            finally:
                server.shutdown()
                server.server_close()
                thread.join(timeout=2)

    def test_get_exposes_only_sonnet_and_opus_with_haiku_marked_unsupported(self):
        with tempfile.TemporaryDirectory() as directory:
            status, payload = self._request(self._write_config(directory), "GET")
        self.assertEqual(status, 200)
        effort = payload["claude_reasoning_effort"]
        self.assertEqual(effort["levels"], {"sonnet": "medium", "opus": "medium"})
        self.assertEqual(effort["defaults"], {"sonnet": "medium", "opus": "medium"})
        self.assertEqual(effort["pinned"], {"sonnet": False, "opus": False})
        self.assertFalse(effort["haiku_supported"])
        self.assertNotIn("haiku", effort["levels"])

    def test_missing_router_status_includes_pinned_tiers_and_defaults(self):
        config = {
            "claude_delegation": {"reasoning_effort": {"sonnet": "high"}},
        }
        with patch.object(web_viewer, "_claude_delegation_module", return_value=None):
            status = web_viewer._claude_reasoning_status(config)
        self.assertEqual(status["defaults"], {"sonnet": "medium", "opus": "medium"})
        self.assertEqual(status["pinned"], {"sonnet": True, "opus": False})

    def test_fallback_status_keeps_pinned_effort_levels(self):
        config = {"claude_delegation": {"reasoning_effort": {"opus": "high"}}}
        with patch.object(web_viewer, "_claude_delegation_module", return_value=None):
            missing = web_viewer._claude_reasoning_status(config)

        class BrokenDelegation:
            @staticmethod
            def reasoning_effort_config(_config):
                raise RuntimeError("reasoning configuration unavailable")

        with patch.object(web_viewer, "_claude_delegation_module", return_value=BrokenDelegation), \
             contextlib.redirect_stderr(io.StringIO()):
            broken = web_viewer._claude_reasoning_status(config)

        for status in (missing, broken):
            self.assertEqual(status["levels"], {"sonnet": "medium", "opus": "high"})
            self.assertEqual(status["pinned"], {"sonnet": False, "opus": True})

    def test_a_fallen_back_status_full_page_save_preserves_a_pinned_effort(self):
        """An unavailable effort control must not turn its pinned value into the default on an unrelated save."""
        class BrokenDelegation:
            @staticmethod
            def reasoning_effort_config(_config):
                raise RuntimeError("reasoning configuration unavailable")

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            config = web_viewer.yaml.safe_load(config_path.read_text(encoding="utf-8"))
            config["models"]["terra"] = "gpt-5.6-terra"
            config["callable"]["terra"] = True
            config["default_model"] = "terra"
            config["effort"] = {tier: "medium" for tier in ("luna", "spark", "terra", "sol", "grok")}
            with config_path.open("w", encoding="utf-8") as handle:
                web_viewer.yaml.dump(config, handle)
            config_path.with_name("hermes-config.yaml").write_text(
                "model:\n  default: gpt-5.6-terra\n  provider: openai-codex\n",
                encoding="utf-8",
            )
            local = config_path.with_name("router_config.local.yaml")
            local.write_text("claude_delegation:\n  reasoning_effort:\n    opus: high\n", encoding="utf-8")
            with patch.object(web_viewer, "_claude_delegation_module", return_value=BrokenDelegation), \
                 contextlib.redirect_stderr(io.StringIO()):
                status, page = self._request(config_path, "GET")
                self.assertEqual(status, 200, page)
                payload = {
                    "callable": page["callable"],
                    "default_model": page["default_model"],
                    "effort": page["effort"],
                    "claude_reasoning_effort": page["claude_reasoning_effort"]["levels"],
                    "preferences": page["preferences"],
                    "hermes_fallback": {},
                    "usage_limits": {
                        account: {"soft_percent": info["soft_percent"], "hard_percent": info["hard_percent"]}
                        for account, info in page["accounts"].items() if info["guard"]
                    },
                    "revision": page["revision"],
                }
                if page.get("balance"):
                    payload["balance"] = {key: page["balance"][key] for key in ("enabled", "busy_percent", "margin_percent")}
                if page["accounts"].get("anthropic"):
                    payload["claude_delegation"] = {
                        "default_tier": page["accounts"]["anthropic"]["delegation"]["default_tier"]
                    }
                payload["callable"] = {**payload["callable"], "luna": False}
                status, body = self._request(config_path, "POST", payload)
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8"))
        self.assertEqual(status, 200, body)
        self.assertTrue(body["success"])
        self.assertEqual(written["claude_delegation"]["reasoning_effort"]["opus"], "high")

    def test_saveSettings_omits_unavailable_claude_effort_controls(self):
        script = r"""
const assert=require('node:assert/strict');
let currentConfig={revision:'r0',callable:{luna:true},accounts:{},default_model:'luna',effort:{luna:'medium'},claude_reasoning_effort:{available:false,levels:{sonnet:'medium',opus:'high'}},preferences:{},hermes_fallback:{}};
let settingsSaveQueue=Promise.resolve(),settingsPending=0,settingsSaveFailed=false,settingsLoadGeneration=0;
const status={textContent:'',style:{}};const $=()=>status,t=k=>k,requests=[];
const fetch=(_url,options)=>{requests.push(JSON.parse(options.body));return Promise.resolve({ok:true,json:async()=>({success:true,revision:'r1'})})};
""" + self.javascript_function("saveSettings") + r"""
(async()=>{await saveSettings();assert.equal(Object.hasOwn(requests[0],'claude_reasoning_effort'),false)})().catch(e=>{console.error(e);process.exit(1)});
"""
        result = subprocess.run(["node", "-e", script], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_claude_reasoning_status_fallback_derives_haiku_support_from_the_delegation_module(self):
        # These direct status probes never save, so CONFIG_PATH/HERMES_CONFIG_PATH need no patch.
        class BrokenDelegation:
            EDITABLE_REASONING_TIERS = ("sonnet", "opus", "haiku")

            @staticmethod
            def reasoning_effort_config(config):
                raise RuntimeError("reasoning configuration unavailable")

        with patch.object(web_viewer, "_claude_delegation_module", return_value=BrokenDelegation), \
             contextlib.redirect_stderr(io.StringIO()):
            status = web_viewer._claude_reasoning_status(self.CONFIG)

        self.assertFalse(status["available"])
        self.assertTrue(status["haiku_supported"])

    def test_claude_reasoning_status_fallback_prints_each_broken_module_traceback_once(self):
        class BrokenDelegation:
            @staticmethod
            def reasoning_effort_config(config):
                raise RuntimeError("reasoning configuration unavailable")

        stderr = io.StringIO()
        with patch.object(web_viewer, "_claude_delegation_module", return_value=BrokenDelegation), \
             contextlib.redirect_stderr(stderr):
            web_viewer._claude_reasoning_status(self.CONFIG)
            web_viewer._claude_reasoning_status(self.CONFIG)

        self.assertEqual(stderr.getvalue().count("RuntimeError: reasoning configuration unavailable"), 1)

    def test_claude_reasoning_traceback_record_is_locked_and_stops_at_64_signatures(self):
        class BrokenDelegation:
            calls = 0

            @classmethod
            def reasoning_effort_config(cls, config):
                cls.calls += 1
                raise RuntimeError(f"reasoning configuration unavailable {cls.calls}")

        self.assertIsInstance(web_viewer._CLAUDE_REASONING_TRACEBACK_LOCK, type(threading.Lock()))
        stderr = io.StringIO()
        with patch.object(web_viewer, "_claude_delegation_module", return_value=BrokenDelegation), \
             contextlib.redirect_stderr(stderr):
            for _ in range(65):
                web_viewer._claude_reasoning_status(self.CONFIG)

        self.assertEqual(len(web_viewer._CLAUDE_REASONING_TRACEBACKS), 64)
        self.assertEqual(stderr.getvalue().count("RuntimeError: reasoning configuration unavailable"), 64)

    def test_a_valid_claude_effort_save_writes_only_the_local_delta(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            shipped_before = config_path.read_bytes()
            status, body = self._request(config_path, "POST", {"claude_reasoning_effort": {"opus": " HIGH "}})
            local = config_path.with_name("router_config.local.yaml")
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8"))
            self.assertEqual(config_path.read_bytes(), shipped_before)
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])
        self.assertIn("revision", body)
        self.assertEqual(written, {"claude_delegation": {"reasoning_effort": {"opus": "high"}}})

    def test_invalid_claude_effort_payloads_are_refused_without_writing(self):
        cases = [
            {"claude_reasoning_effort": ["high"]},
            {"claude_reasoning_effort": {"haiku": "high"}},
            {"claude_reasoning_effort": {"opus5": "high"}},
            {"claude_reasoning_effort": {"sol_long": "high"}},
            {"claude_reasoning_effort": {"opus": True}},
            {"claude_reasoning_effort": {"opus": "   "}},
            {"claude_reasoning_effort": {"opus": "external"}},
            {"claude_reasoning_effort": {"opus": "max"}},
        ]
        for payload in cases:
            with self.subTest(payload=payload), tempfile.TemporaryDirectory() as directory:
                config_path = self._write_config(directory)
                shipped_before = config_path.read_bytes()
                status, body = self._request(config_path, "POST", payload)
                self.assertEqual(status, 400)
                self.assertIn("error", body)
                self.assertFalse(config_path.with_name("router_config.local.yaml").exists())
                self.assertEqual(config_path.read_bytes(), shipped_before)

    def test_a_later_invalid_claude_effort_entry_leaves_earlier_entries_unchanged(self):
        payload = {"claude_reasoning_effort": {"sonnet": "high", "opus": "bogus"}}
        config = json.loads(json.dumps(self.CONFIG))
        before = json.loads(json.dumps(config))
        error = web_viewer._save_claude_reasoning_effort(payload["claude_reasoning_effort"], config)
        self.assertIsNotNone(error)
        self.assertIn("opus", error)
        self.assertEqual(config, before)

        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            shipped_before = config_path.read_bytes()
            status, body = self._request(config_path, "POST", payload)
            self.assertEqual(status, 400)
            self.assertIn("opus", str(body.get("error", "")))
            self.assertFalse(config_path.with_name("router_config.local.yaml").exists())
            self.assertEqual(config_path.read_bytes(), shipped_before)

    def test_a_reset_mixed_with_an_invalid_claude_effort_value_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            local = config_path.with_name("router_config.local.yaml")
            local.write_text(
                "claude_delegation:\n  reasoning_effort:\n    sonnet: high\n    opus: low\n",
                encoding="utf-8",
            )
            before = local.read_bytes()
            status, body = self._request(
                config_path,
                "POST",
                {"claude_reasoning_effort": {"sonnet": "", "opus": "bogus"}},
            )
            self.assertEqual(local.read_bytes(), before)
        self.assertEqual(status, 400)
        self.assertIn("opus", str(body.get("error", "")))

    def test_a_full_defaults_save_with_an_unrelated_change_writes_no_claude_block(self):
        """The exact bug from the final review: saveSettings() always posts the
        full (currently-default) levels alongside any unrelated setting change.
        That must not pin today's defaults into the local file forever."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            status, body = self._request(
                config_path, "POST",
                {"claude_reasoning_effort": {"sonnet": "medium", "opus": "medium"}, "callable": {"luna": False}},
            )
            local = config_path.with_name("router_config.local.yaml")
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8")) if local.exists() else {}
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])
        self.assertEqual(written.get("callable"), {"luna": False})
        self.assertNotIn("claude_delegation", written)

    def test_resetting_a_pinned_opus_effort_to_default_removes_only_that_local_key(self):
        """The full frontend payload uses an empty tier value to unpin its override."""
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            local = config_path.with_name("router_config.local.yaml")
            local.write_text(
                "claude_delegation:\n  default_tier: opus\n  reasoning_effort:\n    sonnet: low\n    opus: high\n",
                encoding="utf-8",
            )
            status, payload = self._request(config_path, "GET")
            self.assertEqual(status, 200)
            self.assertEqual(payload["claude_reasoning_effort"]["pinned"], {"sonnet": True, "opus": True})
            self.assertEqual(payload["claude_reasoning_effort"]["defaults"], {"sonnet": "medium", "opus": "medium"})

            status, body = self._request(
                config_path, "POST",
                {"claude_reasoning_effort": {"sonnet": None, "opus": ""}},
            )
            written = web_viewer.yaml.safe_load(local.read_text(encoding="utf-8"))
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])
        self.assertEqual(written, {"claude_delegation": {"default_tier": "opus"}})

    def test_an_unavailable_bridge_save_of_unchanged_levels_writes_nothing(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            with patch.object(web_viewer, "_claude_delegation_module", return_value=None):
                status, body = self._request(
                    config_path, "POST",
                    {"claude_reasoning_effort": {"sonnet": "medium", "opus": "medium"}},
                )
            local = config_path.with_name("router_config.local.yaml")
            self.assertFalse(local.exists())
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])

    def test_an_empty_claude_effort_payload_is_a_noop_and_creates_no_block(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            status, body = self._request(config_path, "POST", {"claude_reasoning_effort": {}})
            local = config_path.with_name("router_config.local.yaml")
            self.assertFalse(local.exists())
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])

    def test_a_claude_effort_save_preserves_unrelated_local_content_and_comments(self):
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            local = config_path.with_name("router_config.local.yaml")
            local.write_text(
                "# preserve this local note\nclaude_delegation:\n  default_tier: opus\n",
                encoding="utf-8",
            )
            status, body = self._request(config_path, "POST", {"claude_reasoning_effort": {"sonnet": "low"}})
            text = local.read_text(encoding="utf-8")
            written = web_viewer.yaml.safe_load(text)
        self.assertEqual(status, 200)
        self.assertTrue(body["success"])
        self.assertIn("# preserve this local note", text)
        self.assertEqual(
            written,
            {"claude_delegation": {"default_tier": "opus", "reasoning_effort": {"sonnet": "low"}}},
        )

    def test_renderClaudeReasoningEffort_renders_sonnet_and_opus_selects_only(self):
        source = self.javascript_function("renderClaudeReasoningEffort")
        self.assertIn("data-claude-effort=", source)
        self.assertIn("sonnet5:'sonnet',opus5:'opus'", source)
        self.assertNotIn('data-claude-effort="haiku"', source)

    def test_renderClaudeReasoningEffort_shows_haiku_as_a_disabled_placeholder(self):
        source = self.javascript_function("renderClaudeReasoningEffort")
        self.assertIn("settings.claude_effort.haiku_no_reasoning", source)
        self.assertIn('<select class="effort-none" disabled><option>', source)

    def test_renderClaudeReasoningEffort_offers_exactly_the_four_shared_levels(self):
        source = self.javascript_function("renderClaudeReasoningEffort")
        self.assertIn("['low','medium','high','xhigh']", source)

    def test_renderClaudeReasoningEffort_disables_selects_and_tooltips_reason_when_unavailable(self):
        source = self.javascript_function("renderClaudeReasoningEffort")
        self.assertIn("available", source)
        self.assertIn("escapeHtml(state.reason", source)
        self.assertIn("title=", source)
        self.assertIn("disabled", source)

    def test_the_claude_reasoning_effort_control_has_all_of_its_i18n_keys_in_both_languages(self):
        self.assertEqual(self.i18n("settings.claude_effort.haiku_no_reasoning"),
                         ("no reasoning allowed", "nincs gondolkodás"))
        self.assertEqual(self.i18n("settings.claude_effort.default"),
                         ("Default ({level})", "Alapértelmezett ({level})"))

    def test_get_reports_available_when_uninstalled_but_the_host_seam_is_compatible(self):
        """A standalone dashboard process never calls install_reasoning_bridge() itself.

        Availability must mean "the host seam is compatible", not "this
        process installed the wrapper" -- otherwise a separate dashboard
        process permanently disables the Sonnet/Opus selects. This test uses
        the real router module (not the None-patched fallback) so the real
        reasoning_bridge_status()/reasoning_bridge_compatibility() path runs.
        """
        from model_router import claude_delegation

        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)
        claude_delegation._reset_reasoning_bridge_for_tests()
        self.assertFalse(claude_delegation._REASONING_BRIDGE_INSTALLED)
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            data = None
            with patch.object(web_viewer, "CONFIG_PATH", config_path):
                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/config", data=data, method="GET",
                    )
                    with urllib.request.urlopen(request) as response:
                        status, payload = response.status, json.load(response)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
        self.assertEqual(status, 200)
        self.assertTrue(payload["claude_reasoning_effort"]["available"])
        self.assertEqual(payload["claude_reasoning_effort"]["reason"], "")

    def test_get_reports_unavailable_with_a_reason_when_the_probe_is_incompatible(self):
        """When the host seam itself is incompatible, the dashboard must still refuse.

        Patches reasoning_bridge_compatibility() (what reasoning_bridge_status()
        falls back to when uninstalled) directly, rather than sys.modules, so
        this test is independent of the real host's actual compatibility.
        """
        from model_router import claude_delegation

        self.addCleanup(claude_delegation._reset_reasoning_bridge_for_tests)
        claude_delegation._reset_reasoning_bridge_for_tests()
        with tempfile.TemporaryDirectory() as directory:
            config_path = self._write_config(directory)
            with patch.object(
                claude_delegation, "reasoning_bridge_compatibility",
                return_value=(False, "the host seam has moved"),
            ), patch.object(web_viewer, "CONFIG_PATH", config_path):
                server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
                thread = threading.Thread(target=server.serve_forever, daemon=True)
                thread.start()
                try:
                    request = urllib.request.Request(
                        f"http://127.0.0.1:{server.server_port}/api/config", method="GET",
                    )
                    with urllib.request.urlopen(request) as response:
                        status, payload = response.status, json.load(response)
                finally:
                    server.shutdown()
                    server.server_close()
                    thread.join(timeout=2)
        self.assertEqual(status, 200)
        self.assertFalse(payload["claude_reasoning_effort"]["available"])
        self.assertEqual(payload["claude_reasoning_effort"]["reason"], "the host seam has moved")
        # The renderer disables both selects whenever available is false --
        # already covered by test_renderClaudeReasoningEffort_disables_selects_and_shows_reason_when_unavailable,
        # which asserts on the JS source directly rather than re-deriving a DOM here.



class MainParentTests(DashboardProbeMixin, unittest.TestCase):
    """The main agent's first entry can start Hermes on a switched-on Claude model.

    The picker used to offer only the router's own tiers, so a Claude parent could
    only be set by hand in ~/.hermes/config.yaml.
    """

    CONFIG = {
        "models": {"terra": "gpt-5.6-terra", "sol": "gpt-6-sol", "qwen": "qwen3.7-plus"},
        "tier_providers": {"terra": "openai-codex", "sol": "openai-codex", "qwen": "qwen-token",
                           "opus5": "anthropic", "sonnet5": "anthropic", "haiku": "anthropic"},
        "callable": {"terra": True, "sol": True, "qwen": False,
                     "opus5": True, "sonnet5": True, "haiku": False},
        "claude_delegation": {"tiers": {"haiku": "claude-haiku-4-5-20251001",
                                        "sonnet": "claude-sonnet-5-5", "opus": "claude-opus-5-5"}},
        "default_model": "terra",
    }
    HERMES = ("model:\n  default: gpt-5.6-terra\n  provider: openai-codex\n  api_mode: codex_responses\n"
              "agent:\n  max_turns: 150\nfallback_providers:\n- provider: anthropic\n  model: claude-opus-5-5\n")

    def _config(self):
        return json.loads(json.dumps(self.CONFIG))

    def _hermes_file(self, directory, content=None):
        target = Path(directory) / "config.yaml"
        target.write_text(self.HERMES if content is None else content, encoding="utf-8")
        return target

    def test_only_switched_on_claude_models_are_offered(self):
        options = web_viewer._claude_parent_options(self._config())
        self.assertEqual(options, [
            {"key": "sonnet5", "provider": "anthropic", "model": "claude-sonnet-5-5"},
            {"key": "opus5", "provider": "anthropic", "model": "claude-opus-5-5"},
        ])

    def test_picking_opus_moves_hermes_onto_anthropic_and_keeps_the_rest(self):
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                self.assertIsNone(web_viewer._save_main_parent("opus5", config))
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
            backups = list(Path(directory).glob("config.yaml.bak-router-*"))
        self.assertEqual(written["model"], {"default": "claude-opus-5-5", "provider": "anthropic"})
        self.assertEqual(written["agent"], {"max_turns": 150})
        self.assertEqual(len(backups), 1)
        self.assertEqual(config["default_model"], "terra", "a Claude parent leaves the router's route alone")

    def test_picking_a_router_tier_from_a_claude_parent_moves_it_back(self):
        """_save_default_model never moves a parent on another account; an explicit pick must."""
        config = self._config()
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory, "model:\n  default: claude-opus-5-5\n  provider: anthropic\n")
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                self.assertIsNone(web_viewer._save_main_parent("sol", config))
            written = web_viewer.yaml.safe_load(target.read_text(encoding="utf-8"))
        self.assertEqual((written["model"]["default"], written["model"]["provider"], written["model"]["api_mode"]),
                         ("gpt-6-sol", "openai-codex", "codex_responses"))
        self.assertEqual(config["default_model"], "sol")

    def test_a_switched_off_or_unknown_model_is_refused_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            target = self._hermes_file(directory)
            with patch.object(web_viewer, "HERMES_CONFIG_PATH", target):
                for key in ("haiku", "qwen", "nonsense", ""):
                    self.assertIsNotNone(web_viewer._save_main_parent(key, self._config()), key)
            self.assertEqual(target.read_text(encoding="utf-8"), self.HERMES)
            self.assertEqual(list(Path(directory).glob("config.yaml.bak-router-*")), [])

    def test_the_payload_saves_the_parent_and_drops_it_from_its_own_fallbacks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            shipped = root / "router_config.yaml"
            shipped.write_text(web_viewer.yaml.safe_dump(self.CONFIG), encoding="utf-8")
            hermes = self._hermes_file(directory)
            with patch.object(web_viewer, "CONFIG_PATH", shipped), \
                 patch.object(web_viewer, "HERMES_CONFIG_PATH", hermes):
                status, body = web_viewer._save_config_payload({
                    "default_model": "terra", "main_parent": "opus5",
                    "hermes_fallback": {"orchestrator": [
                        {"provider": "anthropic", "model": "claude-opus-5-5"},
                        {"provider": "openai-codex", "model": "gpt-6-sol"}]},
                })
            written = web_viewer.yaml.safe_load(hermes.read_text(encoding="utf-8"))
        self.assertEqual(status, 200, body)
        self.assertEqual(written["model"]["default"], "claude-opus-5-5")
        self.assertEqual(written["fallback_providers"], [{"provider": "openai-codex", "model": "gpt-6-sol"}])

    def _render(self, config):
        source = "\n".join(self.javascript_function(name) for name in
                           ("escapeHtml", "fallbackOptionLabel", "fallbackRouteOff", "fallbackChips",
                            "fallbackPicker", "chainRow", "mainPrimary", "renderMainChain"))
        labels = {m: m.upper() for m in ("terra", "sol", "qwen", "opus5", "sonnet5", "haiku")}
        probe = (self.i18n_runtime()
                 + "const box={innerHTML:''};function $(id){return id==='main-chain'?box:null}"
                 + "let currentConfig=" + json.dumps(config) + ";" + source
                 + "\nrenderMainChain(" + json.dumps(labels) + ",['terra','sol','qwen','opus5','sonnet5','haiku']);"
                 + "console.log(box.innerHTML);")
        return subprocess.run(["node", "-e", probe], check=True, text=True, capture_output=True).stdout

    def _page_config(self, parent):
        return {"default_model": "terra", "routable": ["qwen", "sol", "terra"],
                "callable": self.CONFIG["callable"],
                "parent_options": web_viewer._claude_parent_options(self._config()),
                "hermes_parent": parent, "fallback_options": [], "hermes_fallback": {}}

    def test_the_picker_offers_claude_and_selects_a_claude_parent(self):
        html = self._render(self._page_config(
            {"provider": "anthropic", "model": "claude-opus-5-5", "router_model": False}))
        select = html[html.index('<select id="default-model-select">'):html.index("</select>")]
        self.assertIn('<option value="opus5" selected>OPUS5</option>', select)
        self.assertIn('value="sonnet5"', select)
        self.assertNotIn('value="haiku"', select, "switched off")
        self.assertNotIn('value="qwen"', select, "switched off")
        self.assertIn('value="terra"', select)
        english, _ = self.i18n("settings.main.claude")
        self.assertIn(english.split("{tier}")[0].replace("\\'", "'"), html.replace("&#39;", "'"))

    def test_a_router_parent_selects_its_tier(self):
        html = self._render(self._page_config(
            {"provider": "openai-codex", "model": "gpt-5.6-terra", "router_model": True}))
        self.assertIn('<option value="terra" selected>', html)
        self.assertIn('value="opus5"', html)

    def test_a_pick_is_posted_once_as_main_parent(self):
        handler = HTML[HTML.index("if(el.id==='default-model-select')"):]
        self.assertIn("currentConfig.main_parent=el.value", handler[:300])
        save = self.javascript_function("saveSettings")
        self.assertIn("main_parent:currentConfig.main_parent||undefined", save)
        self.assertLess(save.index("main_parent:"), save.index("delete currentConfig.main_parent"))
