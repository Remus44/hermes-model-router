import json
import sqlite3
import tempfile
import unittest
from pathlib import Path

from view_log import _annotate_internal_prompts, _attach_lifecycle_provenance, load_entries


class ViewLogProvenanceTests(unittest.TestCase):
    def test_completion_gets_human_label_and_redacted_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            db = Path(directory) / "state.db"
            with sqlite3.connect(db) as conn:
                conn.execute("CREATE TABLE async_delegations (delegation_id TEXT PRIMARY KEY, parent_session_id TEXT, dispatched_at REAL, task_json TEXT)")
                conn.execute("CREATE TABLE messages (id INTEGER PRIMARY KEY, session_id TEXT, role TEXT, timestamp REAL, content TEXT)")
                conn.execute("INSERT INTO messages VALUES (7, 'parent', 'user', 10, 'Eredeti feladat: secret=hide-me')")
                conn.execute(
                    "INSERT INTO async_delegations VALUES (?, ?, ?, ?)",
                    ("deleg_test", "parent", 20, json.dumps({"goal": "Részfeladat token=hide-me"})),
                )
            entries = [{"turn_id": "completion:turn", "prompt_preview": "[ASYNC DELEGATION COMPLETE — deleg_test] raw result"}]
            _attach_lifecycle_provenance(entries, db)
            entry = entries[0]
            self.assertEqual(entry["event_kind"], "async_delegation_completion")
            self.assertEqual(entry["origin_message_id"], 7)
            self.assertIn("Delegált feladat befejezési eseménye", entry["lifecycle_prompt"])
            self.assertNotIn("hide-me", entry["origin_preview"])

    def test_typed_completion_stays_an_internal_lifecycle_node_after_raw_preview_is_removed(self):
        entries = [
            {"turn_id": "root:one", "prompt_preview": "Valódi fő feladat"},
            {
                "turn_id": "root:two",
                "event_kind": "async_delegation_completion",
                "prompt_preview": "Delegált feladat befejezési eseménye",
            },
        ]
        _annotate_internal_prompts(entries)
        self.assertTrue(entries[1]["is_internal_prompt"])
        self.assertEqual(entries[1]["parent_turn_id"], "root:one")

    def test_repeated_pricing_completion_counts_as_one_routing_decision(self):
        """The pricing-copy completion turn logged the same delegation eight times."""
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "router.jsonl"
            rows = [{
                "timestamp": "2026-09-30T06:39:51+00:00",
                "turn_id": "parent:pricing",
                "api_call_count": 1,
                "tier": "terra",
                "model": "gpt-5.6-terra",
                "prompt_preview": "mivel a beauty szalon nincs engedve a prodon a araknal felrevezeto",
            }]
            rows.extend({
                "timestamp": f"2026-09-30T07:02:{index:02d}+00:00",
                "turn_id": "parent:completion",
                "api_call_count": index,
                "tier": "terra",
                "model": "gpt-5.6-terra",
                "event_kind": "async_delegation_completion",
                "delegation_id": "deleg_pricing",
                "prompt_preview": "Delegált feladat befejezési eseménye",
            } for index in range(1, 9))
            rows.append({
                "timestamp": "2026-09-30T06:41:54+00:00",
                "turn_id": "worker:pricing",
                "api_call_count": 1,
                "tier": "grok",
                "model": "grok-4.7",
                "prompt_preview": "Correct the pricing copy",
            })
            path.write_text("".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8")
            accounted = load_entries(path)
        completions = [row for row in accounted if row.get("delegation_id") == "deleg_pricing"]
        self.assertEqual(len(completions), 1)
        self.assertEqual(completions[0]["api_call_count"], 8)
        self.assertEqual([row["tier"] for row in accounted if row.get("tier") == "grok"], ["grok"])


if __name__ == "__main__":
    unittest.main()