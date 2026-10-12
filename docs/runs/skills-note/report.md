# Skills-index note injected-context fix

## Commit

Pending final commit on `fix/skills-note-injected-context`.

## Change

`_without_host_injected_context()` now also removes a complete host skills-index delta only when it begins at an appended-context boundary with the host's exact literal signature:

`[System: The skills available in this conversation changed after its system prompt was built.`

The matcher accepts LF/CRLF, permits the note before or after the existing signature-checked host blocks, and preserves look-alikes that lack the exact boundary/signature. Raw prompt text remains the source for logs and for the task passed to the Claude CLI bridge.

## `_last_user_text_and_index` audit

| Site | Reads user text for a decision? | Result |
| --- | --- | --- |
| `__init__._request_has_image_attachment` | No: finds the current item and structural/image payload | Raw content retained. |
| `__init__._strip_historical_image_attachments` | No: finds the current item before rewriting earlier image payloads | Raw content retained. |
| `__init__._prompt_preview` | Log/preview only | Raw text retained. |
| `__init__._quota_redispatch_instruction` | Yes: parses delegation outcome | Stripped. |
| `__init__.classify_request` | Yes: eligibility/veto decision | Stripped. |
| `__init__._classify_request` | Yes: routing gates, labels, thresholds | Stripped. |
| `__init__._append_user_instruction` | No: selects the latest item to append router text | Raw request retained. |
| `__init__._orchestration_skip_reason` | Yes: tool mention, length, lifecycle/preflight gates | Stripped. |
| `__init__._shadow_eligible` | Yes: image-text and eligibility gate | Stripped. |
| `__init__._routing_note` | Yes: eligibility/lifecycle gate before advice | Stripped. |
| `__init__._route_llm_request` external redispatch check | Yes: delegation-outcome gate | Stripped. |
| `__init__._route_llm_request` root-session label check | Yes | Stripped. |
| `__init__._route_llm_request` Spark-worker read-only gate | Yes | Stripped. |
| `__init__._durable_fallback_model` | Yes: design fallback guard | Stripped. |
| `__init__._transient_fallback_model` | Yes: design fallback guard | Stripped. |
| `__init__._maybe_run_opus5` | Yes: bridge admission/coding classifier | Uses stripped `routing_text`; raw `text` remains the CLI task. |
| `triage.skip_reason` | Yes: lifecycle, tool mention, conversational/action gate | Stripped before every text decision. |

Additional non-`_last_user_text_and_index` readers `_lifecycle_event_kind` and `_completion_delegation_id` now consume the stripped view before deciding from the latest-user preview. No calls were found in `worker_admission.py`, `claude_delegation.py`, or `execution_adapters.py`.

## RED evidence

Detached worktree: `/home/remus/.hermes/cache/scratch/skills-note-red` at `61df486`, with only `test_injected_context.py` changed. Command:

```text
bash ~/.hermes/skills/hermes-model-router-dev/scripts/run_tests.sh --tree /home/remus/.hermes/cache/scratch/skills-note-red module model_router.test_injected_context
```

Result: 17 tests, 8 expected assertion failures. The new failures showed Luna changing to Terra, the design note changing to Sol, and a host `delegate_task` mention being treated as explicit delegation. The detached worktree was removed immediately after the run.

Swap proof: temporarily restricting the matcher to the `Now available` variant made `test_skills_note_variants_and_positions_are_stripped` fail for the `accurate again` variant; the restriction was restored before verification.

## Tests added

- `test_skills_note_keeps_luna_and_spark_child_routes_and_effort`
- `test_skills_note_variants_and_positions_are_stripped`
- `test_skills_note_lookalikes_remain_operator_text`
- `test_skills_note_is_absent_from_design_label_and_length_decisions`
- `test_skills_note_is_absent_from_tool_mention_and_triage_decisions`
- `test_execution_bridge_uses_stripped_text_for_workspace_and_length_but_sends_raw_task` now includes a skills-index note and asserts the CLI still receives the raw task.
- `test_host_skills_delta_note_does_not_change_luna_read_only_classification` updates the prior regression expectation.

## Verification

- Targeted injected-context module: 17 tests passed.
- Required focused modules (`test_injected_context`, `test_model_router`, `test_triage`, `test_f09_effort_labels`, `test_evidence_goal_verbs`): 236 tests passed.
- Full suite: 1374 tests passed, zero failures (`run_suite.sh --full`).

## Issues

None.
