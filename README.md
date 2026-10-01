# Model Router — Hermes Agent Plugin

Routing and delegation for Hermes Agent. It keeps the user-facing conversation on one durable parent model, lets that parent's plan choose which model runs each delegated worker, and records every decision in a privacy-safe audit log.

The point of choosing per worker is that the models sit on **different accounts**: Codex (Luna/Spark/Terra/Sol), a Qwen token plan, a SuperGrok subscription (Grok 4.7), and a Claude subscription (Opus 5.5/Sonnet 5.5/Haiku 4.5). Spreading independent work across them spends separate quotas in parallel instead of draining one.

## Install

```bash
hermes plugins install Gdeepwell/hermes-model-router
```

Then enable it:

```bash
hermes plugins enable model-router
```

## Features

### Automatic Routing

The tier name is what you write in `model:` and in `router_config.yaml`. The
account is the point of the table: spreading independent work across the three
is what keeps a single quota from carrying everything.

| Tier | Model | Account | Purpose |
|------|-------|---------|---------|
| `terra` | GPT-5.6 Terra | Codex | Durable default parent and conductor |
| `luna` | GPT-6 Luna | Codex | Simple tasks, short answers |
| `spark` | GPT-5.3 Codex-Spark | Codex | Read-only code analysis, bounded subtasks |
| `sol` | GPT-6.1 Sol | Codex | Complex, security-sensitive, design |
| `qwen` | Qwen 3.7 Plus | Qwen token plan | Delegation target only |
| `grok` | Grok 4.7 | SuperGrok subscription (`xai-oauth`) | Delegation target, heavy peer of Terra/Opus/Sonnet; ships switched off |
| `opus5` | Claude Opus 5.5 | Claude subscription | Delegation target for hard or consequential work (see below) |
| `sonnet5` | Claude Sonnet 5.5 | Claude subscription | Delegation target, the everyday Claude worker (see below) |
| `haiku` | Claude Haiku 4.5 | Claude subscription | Quick lookups and exploration; reached only through `delegate_claude` |

`qwen`, `grok`, `opus5`, `sonnet5` and `haiku` are delegation targets rather than routable
tiers: the middleware cannot move a call across providers, so they are reached by a
plan choosing them, not by the router switching to them mid-turn. `qwen`, `grok`,
`opus5` and `sonnet5` are chosen with `model:` on `delegate_task`; the three Claude
targets are also reached with `delegate_claude(tier=...)`, and `haiku` only that
way, since it has no `delegate_task` target. Claude, like Grok, needs its own
subscription, so its three models **ship switched off**; switching one on is what
makes Claude available (see [Switching Claude on](#switching-claude-on) and
[Claude targets](#claude-targets)).
Any tier in `models` can hold the orchestrator role, including one on another
account — that choice is made at spawn time, where the provider is still open.

### Grok (SuperGrok)

A SuperGrok subscription reaches Grok through Hermes's own `xai-oauth` provider
(the Codex Responses API at `api.x.ai`, billed to the subscription, no API key;
X Premium+ does not include this access). The tier ships **switched off**. To
enable it:

1. `hermes auth add xai-oauth --type oauth --no-browser` — adds the login to the
   credential pool without touching the main agent's model (`hermes model` would
   switch the parent onto Grok).
2. Add the delegation target to `~/.hermes/config.yaml`:
   ```yaml
   delegation:
     targets:
       grok:
         provider: xai-oauth
         model: grok-4.7
   ```
3. Switch `grok` on in the dashboard (or `callable.grok: true` in
   `router_config.local.yaml`).

Grok is then a `delegate_task(model="grok")` target, a preference-chain entry, a
fallback for the main agent or the workers, and a `heavy` peer next to Terra,
Opus and Sonnet. It takes the same `reasoning.effort` levels as the Codex tiers
(`low` to `xhigh`, default `medium`). A Grok parent is written with
`api_mode: codex_responses`.

xAI publishes no *official* usage endpoint, but `hermes usage`/`/usage` and the
dashboard's account card read the same undocumented billing endpoint the Grok
CLI itself uses (`https://cli-chat-proxy.grok.com/v1/billing?format=credits`,
sent your stored `xai-oauth` bearer, no API key, no cookie). SuperGrok has a
single **weekly** pool shared across API/Chat/Build/Voice — no 5-hour session
window — so only the weekly figure shows and `usage_guard`'s 5-hour/tighter
comparisons fall back to weekly alone for this account. This is an unofficial
endpoint xAI can change or remove without notice; a failed read just leaves the
card blank and the ordinary cooldown still benches the tier on a quota refusal.
To apply the same soft/hard delegation limits Claude and Codex get, add it
under `usage_guard.accounts` in `router_config.local.yaml`:

```yaml
usage_guard:
  accounts:
    xai-oauth:
      soft_percent: 70
      hard_percent: 90
      step_down: {grok: terra}
```

### Substitution groups

`fallbacks` rewrites the model on a call whose provider and credentials are
already fixed, so it can only ever move work between tiers on one account.
Moving it between *accounts* has to happen where the provider is still being
chosen — when the plan picks a target. So peers are expressed to the conductor
rather than applied behind it:

```yaml
peer_groups:
  heavy: [terra, opus5, qwen, sonnet5]
  light: [luna, spark, haiku]
```

The contract names the groups, and an unavailable target is annotated with its
live replacement — `opus5 [unavailable for another 15 min; use qwen instead]` —
instead of vanishing from the list. Dropping it said only that it was gone;
naming the replacement is what turns one account's exhaustion into work
continuing somewhere else.

**"Comparable in strength" is quoted to the conductor verbatim, so a wrong
grouping is an instruction to misroute.** `sonnet5` sat in the light group until
1.9.5, next to a tier bounded at 700 characters and `low` effort. The conductor
duly substituted Luna for it whenever the Codex account looked loaded, and
"stabilize, correct, test and commit the dirty foundation" ran on Luna twelve
times. Group by what a target can actually carry, not by what it costs.

Substitution is for capacity, not permission. A `[spark]` leaf must still be
read-only wherever it runs, and design work still belongs to Sol — so a cooling
Sol is shown as unavailable rather than hidden, because waiting for it is a
legitimate answer and rerouting the work is not.

### Stable Parent Policy

The user-facing conversation stays on one durable parent model. Classifier results are worker recommendations, not silent parent switches. Sol, Spark, Qwen and Opus never become the apparent responder merely because a prompt mentions UI, CSS, design, or long text.

### Privacy-First Audit Logging

- 240-character bounded prompt preview
- Redacted sensitive data (tokens, URLs with credentials, etc.)
- Parent/child correlation IDs
- Token and cost metrics when supplied by the runtime
- No raw prompt or response bodies stored in logs

### Bounded Delegation

- Concurrency, spawn depth, and iteration budgets come from Hermes’s `delegation` settings and are shown in Settings.
- A forced conductor requires `max_spawn_depth >= 2` and `orchestrator_enabled: true`. With one level, the parent delegates workers directly and keeps integration ownership.
- The router does not raise host limits.
- Handoff capsule required for every worker task

On current Hermes, forced planning uses one `tasks` entry and puts the routing
contract in that entry's `context`. Legacy single-goal hosts remain supported.
The router advertises a `model` parameter only when the host exposes it; otherwise
Codex labels choose models within the configured provider and Claude workers use
`delegate_claude`.

#### Small-task budget

A short, clear task must not pay for a planner plus a tree of workers. The
`task_budget` guard applies to a single-task `delegate_task` or `delegate_claude`
dispatch whose goal is at most `low_risk_max_chars` (default 1200) and is not a
design request or a consequential action (deploy, migration, security, credential,
password, payment, database, ssh, sudo). The word "production" alone is not a
risk signal. Such a task gets:

- one worker per **root user turn** (`max_routing_decisions`, default 1). The
  allowance belongs to the root turn, so a child's own spawns draw on the same
  budget rather than a fresh one;
- no recursive delegation (`max_depth`, default 1);
- one further worker only when the call carries `escalation_evidence` naming a
  failed test or a concrete blocker, and never more than that one escalation.

```yaml
task_budget:
  enabled: true
  low_risk_max_chars: 1200
  max_routing_decisions: 1
  max_depth: 1
  require_evidence_for_second_worker: true
```

Blocked dispatches return `Nothing was spawned.` Complex and consequential work keeps
the normal host limits. The lifecycle log deduplicates replays by event, phase, plan,
turn and child session, and the dashboard counts an async-delegation completion once
per `delegation_id` instead of once per replayed row.

### Claude targets

Claude is reached like any other delegation target — `model: "opus5"` or
`model: "sonnet5"`, running on the native Anthropic Messages API. The child is
an ordinary worker with the usual tools, so it can implement rather than merely
read, and it runs in the background alongside the plan's other leaves. It draws
on a different subscription from every other target, which is the point.

```yaml
delegation:
  targets:
    opus5:
      provider: anthropic
      model: claude-opus-5-5
    sonnet5:
      provider: anthropic
      model: claude-sonnet-5-5
```


**These targets work** once switched on (they ship off only because they need a
Claude subscription; see [Switching Claude on](#switching-claude-on)). They
shipped switched off for months before that because every call returned:

```
HTTP 400 invalid_request_error
Third-party apps now draw from your extra usage, not your plan limits.
Add more at claude.ai/settings/usage and keep going.
```

That reads like a policy prohibition. It was not one, and it was not an account
limit either — the earlier diagnosis here ruled out the credential, request
size, tools, delegation, and the account's extra-usage setting, then wrongly
concluded the Hermes version was identical to upstream and could be excluded.

**The version was the whole difference.** Proved by running the same
`auth.json` and `HERMES_HOME` against two trees: `HTTP 400` on the old one,
`OK` on current upstream. The old tree authenticated with Hermes's own OAuth
app — a `manual:hermes_pkce` token in the credential pool — which the API
correctly classifies as a third-party app. Current upstream instead borrows the
Claude Code login (`agent/anthropic_credentials.py` plus the credential pool's
`_seed_from_singletons`), which is the same grant Claude Code itself uses.

After migrating to the current tree on 2026-09-09, the same account answers on
both paths — a direct `hermes -z --provider anthropic --model claude-opus-5`,
and the delegated credential path that `_task_credentials` resolves per child.
If you are on an older Hermes and see the 400, update before you buy credit.


The native targets above are Hermes workers, not Claude Code processes. A normal
`model: "opus5"` / `model: "sonnet5"` target is a Hermes child on the Anthropic
provider; `delegate_claude` reaches the same kind of
child by passing a pinned `provider: anthropic` / model route to Hermes's own
`delegate_task`. Hermes resolves that child's credentials and supplies its usual
worker lifecycle and tool set. That is why these are the normal choice for
parallel delegated work, and why Haiku exists only there.

Meanwhile the CLI bridge below invokes `claude -p` in the selected repository:
that is a Claude Code process, not a native provider child. Its authentication,
tools and project behaviour belong to the installed, authenticated CLI; this
plugin only supplies the prompt, working directory and bounded tool flags. A
`[sonnet-review]` or `[opus-review]` is therefore a read-only replacement for
one call, not an agent. For those delegated labels, the bridge resolves the
repository from an absolute path in the goal that lies inside a Git work tree
(its top level is used), then configured aliases, then the child request's
`WORKSPACE PATH:` system-prompt block, then `default_repo`; a nonexistent
configured path is skipped. The same bridge can take the explicit `[opus]` / `[opus5]`
coding override when `coding_agent.enabled` admits it; that is the bounded
single-call coding path, configured by `coding_agent.default_repo`, aliases and
its CLI limits, not a `delegate_claude` worker. Choose this path when the work
specifically requires Claude Code project context or CLI-managed capabilities
(for example a project's `CLAUDE.md`, skills, plugins, MCP servers or Claude Code
tools); the native target does not start the CLI. This repository has not
measured which of those features a particular Claude Code installation loads
from `cwd`, so verify that setup in the target project rather than treating this
paragraph as a capability probe.

The router does not *route* these children — `route_llm_request` returns `None`
for a model outside its own tier map, so nothing here rewrites them — but it does
record them. Invisible to the router had meant invisible to the operator: a
Claude worker produced no card, no count and no line in the per-account load, so
the one account whose usage most needed watching was the one nothing reported on.
They now appear as `opus5`, `sonnet5` and `haiku` alongside the other tiers, and
their `callable` switches govern whether the conductor is offered them at all.

**Haiku is a third Claude target, reachable only through `delegate_claude`.**
It has no `delegation.targets` entry in Hermes's config, so `delegate_task` cannot
start it; `delegate_claude(tier="haiku")` can (see
[Claude delegation and the usage guard](#claude-delegation-and-the-usage-guard)).
While that tool is live, `haiku` joins the conductor's target list with the
advice "quick lookups", and it can be named in a preference chain like any other
target — typically for `explore`. Switched off in `callable`, it is never offered.

When both are offered the contract used to add "use `sonnet5` by default and
reserve `opus5` for consequential or hard work". That sentence dates from the
commit that made these targets reachable at all, when no preference mechanism
existed and a bare list of two names told the conductor nothing. One arrived 23
hours later and the older answer was never withdrawn, so two contradictory
instructions sat in the same paragraph — and the unconditional one beat the
hedged one every time. It is now emitted **only where the operator has configured
nothing**, and the check reads the configured chain rather than its currently
available winner, so a cooling `opus5` cannot revive the built-in default at the
one moment the operator's own order needs to be what speaks.

### Claude reasoning effort

`claude_delegation.reasoning_effort.sonnet` and `.opus` (`low`, `medium`, `high`
or `xhigh`; default `medium`) set the thinking effort `delegate_claude` gives its
Sonnet or Opus child. Hermes copies a delegating parent's own `reasoning_config`
into every child verbatim, which would give a Claude worker whatever effort the
parent happens to be running at rather than its own configured tier. Instead
this plugin installs a narrow bridge over Hermes's private
`tools.delegate_tool._resolve_child_runtime` seam: for the exact duration of one
`delegate_claude` call it substitutes that call's configured effort, and only for
the matching child of the same parent, same provider (`anthropic`) and same
model — any other concurrent delegation, any non-Anthropic child, is untouched.
When usage guard steps a busy Opus request down to Sonnet mid-call, the
substituted effort is Sonnet's own setting, not Opus's, since it follows the
final tier that actually runs. Haiku has no entry here: Hermes's Anthropic
adapter sends no thinking configuration for Haiku models, so there is nothing
for the bridge to override.

If the host's private seam has moved in a way this bridge does not recognise,
`delegate_claude` is refused outright with "Claude reasoning effort is
unavailable: `<reason>`" rather than silently running at the wrong effort;
Haiku calls, which never touch the bridge, are unaffected. In the Claude
account card, Sonnet and Opus effort dropdowns sit beside their model switches. Their
first choice is **Default (`medium`)**, which removes a tier-specific dashboard
override so a future router default can apply. Haiku has a disabled “no reasoning allowed” dropdown. If the host seam is
incompatible, Sonnet and Opus selects are disabled and the escaped reason is
available in their tooltips, using the same side-effect-free probe as the bridge.

If the dashboard reports the Claude reasoning-effort control unavailable with a
`ModuleNotFoundError`, this host's Hermes venv install map is stale after a
Hermes update; re-running Hermes's own update/install step resolves it.

A read-only CLI bridge also exists (`[opus-review]` / `[sonnet-review]`,
`coding_agent.delegated_review`). It replaces a single call rather than running
an agent, so it cannot write and holds a child slot for the duration; the native
target above supersedes it for ordinary work. CLI review identity is recorded as
requested, resolved and observed: a CLI alias is not proof of the served model.
An `exact` review requires recorded `exact_model` capability evidence and is
refused before the CLI starts when that evidence is absent. The default
`profile_preferred` review may use the configured ordinary provider route after
an attempted CLI failure. For a successful CLI step-down or a response-shaped
ordinary replacement, the router appends one bounded, deterministic,
plugin-owned `ROUTER SUBSTITUTION PROVENANCE v1` JSON line after the useful
verdict. That content survives the supported Codex Responses normalization and
child-result projection to the parent; arbitrary response attributes do not. It
carries the requested/resolved/observed review identity and evidence sources,
effort requested/applied state, policy/reason, planned replacement and final
ordinary provider/model observation. Missing provider or model evidence remains
explicitly `unknown`. The marker always says the result does **not** satisfy a
cross-provider review requirement: a same-provider step-down or ordinary
replacement is not independent review. Scalar legacy payloads with no supported
response-content carrier remain byte-for-byte unchanged. The Claude audit keeps
the typed failure and planned replacement separately; it is not the parent-facing
provenance carrier. An exact route that
cannot be admitted, or that fails after its one CLI attempt, stops at the
execution-middleware boundary with a zero-token refusal response;
the response states that no work was performed by that call. The direct bridge still
carries its typed failure evidence internally. Neither response counts as a Claude
review.

### A parent on a fallback account still orchestrates

Rewriting a model is provider-bound; orchestrating is not. When Hermes's own
fallback chain moves the orchestrator onto another account — Codex out of quota,
so the parent continues on Sonnet — that parent is still the orchestrator and
still gets the preflight, the delegation contract and the `model:` parameter
contract. Its model is never rewritten, because this middleware cannot change a
request's provider; only the instructions are added.

Until 1.9.1 it did not: `route_llm_request` returned early for any model outside
its own provider, so a parent on a fallback account silently lost its contract
and worked alone. A second gate compounded it by recognising only Sol and
`default_model` as orchestrators.

**Every parent tier orchestrates (1.19.0).** Grok, Qwen, Luna and Sol parents get
the same forced preflight as Terra; before, only Sol and `default_model` did, so a
session switched to Grok never delegated at all. Sol's own Opus-bridge preflight
still takes precedence when it is configured. `orchestration.min_chars` applies to
every router-owned parent; an external (Claude) parent is preflighted regardless
of length, as before.

**The written objective decides, not the user's message (1.20.0).** `min_chars`
only skips a *fresh* short prompt. A short follow-up after earlier assistant work
("csinald meg" after a discussed plan) is preflighted anyway, and the parent is
told to write the plan out as a self-contained goal, since the conductor never sees
the conversation. The `pre_tool_call` gate then measures that goal: under
`orchestration.min_goal_chars` (default 500; 0 turns it off) the planner call is
refused with a message, nothing is spawned, a `preflight_declined` event is logged,
and the parent continues with its full toolset. Only the router's own forced
planner call is measured, recognised by its pinned contract. Above the threshold
the conductor still decides how many workers (zero included) the task deserves.

The forced conductor uses `orchestration.conductor`, then the parent's own tier
(while `orchestration.conductor_follows_parent`, default on, is set), then the callable
`default_model` and its fallback chain, skipping targets that are switched off
or that the `delegate_task` schema cannot reach. On hosts without a `model` parameter,
an off-provider goal prefix cannot create an off-provider conductor; if no
planner route remains, the parent keeps direct worker coordination. Worker preference chains such as `code`
choose implementation workers independently. The host must allow a conductor
to spawn children; with a one-level limit, the parent coordinates direct workers.

One more thing had to change for this to actually fire. An Anthropic OAuth request
is normalised for Claude Code compatibility, which renames every tool to
`mcp__<name>` — so a Claude parent was told it had no `delegate_task` tool and
skipped its preflight. The lookup accepts both names now, and the four places that
each carried their own copy of it share one function, which is why the mismatch
survived as long as it did.

`preflight_skipped` events record the tool names that were on offer, so "no
delegate_task tool" can be told apart from "no tools at all" and from "a name this
router does not recognise" without adding instrumentation after the fact.

### Default model vs. preference chains

These look redundant and are not. `default_model` decides five things; a
preference chain overrides exactly one of them:

| | overridden by a chain? |
|---|---|
| the general-purpose route when no gate matches | **yes**, by the `default` kind |
| which tier counts as an orchestrator at all | no |
| the tier the forced conductor runs on | no |
| whether the preflight dispatches | no |
| **the model Hermes itself starts on** (`model.default`, `model.provider`) | no |

The last row is the one that matters: a chain re-routes a turn, it does not change
which model the process launches with. Setting `default_model` in the dashboard
writes Hermes's own config as well as the router's. It does so **only when you
actually change it**: the Settings page posts `default_model` on every save, so
writing it through unconditionally meant a callable toggle moved a Claude parent
back onto this provider's tier. A tier with no entry in `models` — a delegation
target such as `opus5` — is refused rather than written, because Hermes cannot
launch on it and `_decision` raises for it.

### Preferred models per kind of work

Configured under `preferences:` in `router_config.yaml`, or from the Settings
tab, as an ordered chain per work kind — `design`, `code`,
`explore`, `review`, `sensitive`, `critical`, `long`, `chat`, `default`. The
router walks the chain and takes the first entry that is switched on.

A chain entry means one of two different things, and the dashboard colours them
differently because the difference is not cosmetic:

- **A tier of the router's own provider** (`luna`, `spark`, `terra`, `sol`) is a
  real route. The router rewrites the model and the chain also replaces the
  built-in fallback order for that kind.
- **Anything on another account** (`opus5`, `sonnet5`, `haiku`, `qwen`) cannot be routed
  to at all: `route_llm_request` runs after the provider is chosen, so it can only
  swap models inside one provider. Such an entry is passed to the conductor as a
  delegation instruction instead — it reaches work through `delegate_task`
  (or `delegate_claude` for a Claude tier), and the
  conductor is the one that has to honour it.

A kind with no chain keeps its built-in route, so configuring nothing changes
nothing. A kind with a chain overrides that route **completely**, including the
safety defaults that send design, security and deployment work to Sol. That is
deliberate: the operator owns the mapping.

**A switched-off model is skipped, not honoured.** A chain can name a model whose
switch is off -- `review: [sonnet5, opus5, terra]` with Claude off, say. The
router passes over it: that review goes to `terra`, and the conductor is not
advised a Claude target. When nothing in a chain is switched on (`[sonnet5, opus5]`
with Claude off), the kind falls back to its built-in route, as if it had no
chain. The chain itself is kept, so switching Claude back on restores it.

**The whole order reaches the conductor, not just its winner.** Availability
folds in the cooldown, so naming only the first *available* entry meant a cooling
`opus5` erased `code` from the contract entirely — indistinguishable from a kind
nobody configured, and the conductor could not advance to an entry it was never
told existed. Cooling entries are annotated instead, and the conductor's own tier
belongs in the chain too: `code: opus5 > terra` rendered as `code: opus5` and
lost the very entry that has to take over.

```
The operator's target order per kind of work, highest priority first --
  design: sol > opus5; code: opus5 > terra; review: sonnet5 > terra.
A leaf of one of these kinds must take the first target in that kind's order,
and when an entry is marked unavailable must move to the next entry in the same
order rather than choosing freely.
```

**It is phrased as an instruction because advice loses.** It shares a paragraph
with the `[spark]` and `[sol]` rules, which are imperatives. Until 1.9.5 the
chain arrived as "Honour these when a leaf matches the kind and the target is
free" while an unconditional built-in default sat two sentences earlier — so
`code -> model:opus5` never once decided a leaf.

### Switched-off targets and `delegate_task`

`delegate_task` offers every entry of Hermes's `delegation.targets`, whatever the
dashboard says, so an agent can still call `delegate_task(model="qwen")` with Qwen
switched off. The `llm_request` middleware notices, but Hermes logs a middleware
error and sends the request unchanged — measured 2026-09-25, a Terra worker spawned
a Qwen child that died on a 403. A `pre_tool_call` hook therefore checks the
call-level and every per-task `model` (a full model name is mapped back to its
tier) and blocks the call before anything spawns, naming a working route instead:

```
Delegation target "qwen" is switched off in the dashboard. Use model "terra" or
delegate_claude with tier "opus" instead. Nothing was spawned.
```

A switched-off target is always refused. A target that is only cooling down is
refused while `delegation.fallback_providers` is empty; with a worker chain Hermes
can still move the child to another account. Names the router does not know are
left to Hermes, other tools are never touched, and a failing check lets the call
through.

### Hermes fallback chains

The Settings tab also edits the two chains that live in **Hermes's** config rather
than the router's, because that is where a cross-account rescue is decided:

| Chain | Key in `~/.hermes/config.yaml` | Applies to |
|---|---|---|
| Orchestrator | `fallback_providers` | the main agent, when its own provider cannot serve |
| Delegated workers | `delegation.fallback_providers` | every child spawned by `delegate_task` |

The second is not optional in practice. A child pinned to a target — anything
spawned with `model: "opus5"` and friends — **never inherits the orchestrator's
chain**, so without its own it runs with no fallback at all: a quota-exhausted
leaf simply dies mid-task. An empty list there means "no fallback", which is a
different instruction from the key being absent.

On the page the orchestrator chain is not a separate setting: **Main agent** shows
one chain whose first entry is the model Hermes starts on and whose remaining
entries are `fallback_providers`. The first entry offers the router's own tiers
(picking one also sets `default_model`) and every Claude model whose switch is
on. Picking a Claude model writes `model.default`/`model.provider: anthropic`
(no `api_mode` or base URL, as `hermes model` writes it) and leaves
`default_model`, the default route and the conductor, where it was. Setting the default and its
fallbacks in two places let the primary sit in its own fallback list — a step
that can never help, since when its provider is out so is that entry — so a save
drops the parent from `fallback_providers`. A route whose tier is switched off is
dropped from both chains on save, since Hermes does not consult the router's switches.

The **Workers** section holds the worker defaults side by side: the
`delegate_task` model (`delegation.provider`/`delegation.model`, Codex tiers only,
since that block carries no key or base URL), the `delegate_claude` default tier,
and the delegated-worker chain. The `delegate_task` model used to be invisible, so
changing the default model left every Codex worker on the old one.

Do not confuse either with the router's own `fallbacks:`, which substitutes tiers
*within* one provider and cannot cross accounts.

The picker offers only routes that exist as `delegation.targets`, so a chain
cannot name something the installation cannot run. Because this file is not the
router's — it also carries providers, approvals and the command allowlist — every
save first copies it to `config.yaml.bak-router-<timestamp>`, and a config that
cannot be read is refused rather than overwritten.

The delegated-worker fallback chain applies to native `delegate_task` workers.
`delegate_claude` supplies no fallback providers: a failure returns to the parent
for an explicit re-dispatch through the current worker order. The dashboard
states this distinction beside the worker fallback control.

Settings saves validate both configuration documents before writing. The page
queues edits and sends a revision token; conflicting saves require a reload.
After a failed save, queued edits stop and the error stays visible until a
successful reload supplies a fresh revision. API clients that omit `revision`
retain the legacy last-write-wins behavior; integrations that need conflict
protection must GET `/api/config` and send its `revision` on every POST.
Each file is replaced atomically, and a failed router write restores the previous
Hermes document unless another writer has changed it in the meantime. This
protects ordinary save failures; it is not a cross-file crash transaction.

### Reasoning effort per tier

A route carries an effort level, not just a model name. `effort:` sets it per tier,
and separately for the cases where the same tier means different work:

```yaml
effort:
  luna: low
  terra: medium
  sol: medium
  sol_long: medium            # reached by length, not by choice
  explicit_sol: medium        # you asked for Sol
  explicit_sol_xhigh: high    # you asked for Sol and said how hard
  explicit_luna_xhigh: high   # ...and the same for every other tier
  explicit_terra_xhigh: high
```

`[<tier>:xhigh]` is the label form. It parsed for every tier from the start but
was honoured only for Sol, so `[luna:xhigh]` ran silently at Luna's floor with no
way to say otherwise; `explicit_<tier>` was unreadable config everywhere else for
the same reason. Both keys are general since 1.9.5.

Resolution walks from the most specific key to the plain tier, and **a key you
have not written changes nothing** — that is what makes adding a tier here
optional rather than a behaviour change. An escalation degrades to its tier's
`explicit_` key rather than to the tier's floor, so a missing
`explicit_<tier>_xhigh` means "no escalation configured" instead of silently
capping the request.

The Settings tab edits the five plain routed-tier keys: `luna`, `spark`,
`terra`, `sol`, and `grok`, in dropdowns beside their switches in the Codex and
Grok account cards' **Models** blocks. `effort.grok` reaches Grok the same way
as the Codex tiers' effort. `opus5` is a route/log marker; Qwen has no dashboard effort
control because the router strips reasoning; and Claude-only `haiku` / `sonnet5`
are not routed here — this `effort:` map cannot reach a delegated Claude child at
all, since the router never runs on that call. Sonnet and Opus have their own
controls beside their switches in the Claude account card: see [Claude reasoning
effort](#claude-reasoning-effort). The situational `sol_long` and
`explicit_<tier>` / `explicit_<tier>_xhigh` keys remain file-only in
`router_config.yaml` or its local overlay.

### Images force a vision-capable route

Spark is text-only. An attached image routes to Terra as a policy decision, ahead
of every Spark path: label, benchmark override and tool loop alike. A `[spark]`
tag on a message with an image is not honoured, because the alternative is a leaf
that cannot see what it was asked about.

Historical image attachments are stripped from replayed context rather than resent.

### Quota fallback

Distinct from `fallbacks`, which handles a *disabled* tier. `quota_fallbacks`
handles a tier that answered `429` mid-turn:

```yaml
quota_fallbacks:
  spark:
    model: luna
    effort: medium
```

It fires once per turn and only if the replacement is itself callable, so an
exhausted Spark continues on Luna instead of failing the turn.

Note what that last condition rules out. A usage quota belongs to the account, and
the failure is recorded **before** a replacement is looked for — which benches
every sibling on that account first. So on a genuine quota exhaustion this
substitution cannot fire at all: the configured replacement is already cooling.
What survives here is the case it is actually good for — a transient blip, or a
model this account cannot use — where a same-account sibling is the right answer.
Moving quota-stopped work needs a different account, and that is a dispatch
decision, not a model swap.

### A model this account cannot use

Three different unavailabilities, three different mechanisms — the distinction is
what decides whether a leaf survives:

| Condition | Recognised by | What happens |
|---|---|---|
| Switched off in the dashboard | `callable: false` | the `fallbacks` chain, at routing time |
| Quota exhausted (429) | account/weekly quota wording | `quota_fallbacks`, then a 15-minute cooldown |
| Provider blip (5xx) | 500/502/503/504, connection resets | a hardcoded substitute, then a short cooldown |
| **Refused outright (400/404)** | *"is not supported when using…"*, *"does not exist or you do not have access"* | the configured `fallbacks` chain, then a 6-hour cooldown |

The last row was a gap until 1.8.1. A tier switched **on** but refused by the
provider — `The 'gpt-5.3-codex-spark' model is not supported when using Codex with
a ChatGPT account.` — matched no runtime failover, so the leaf simply aborted.

It deliberately walks the **configured** `fallbacks` chain rather than the
hardcoded transient map: if you wrote `spark: luna`, a Spark that does not exist
on your account belongs on Luna, not wherever the blip handler would have put it.
The cooldown is long because nothing about this recovers by waiting; tune it with
`cooldown.unavailable_seconds`. Design and image guards still apply — an
unavailable model is no reason to break a routing policy.

An ordinary malformed-request 400 is untouched and still raises.

### A leaf is classified from its goal, not from its contract

A delegated leaf carries the routing contract in its own first message, and that
contract necessarily talks about applying, implementing and committing. Classifying
the leaf from that combined text made every `[spark]` leaf look like mutating work:

```
goal alone        -> read-only work
goal + contract   -> not read-only, on the word "apply" from the contract
```

A failed read-only test is the only thing that consults the consequential check, so
a bounded discovery task whose goal merely says "API route or server-side code" was
not just rejected as a Spark leaf but promoted to Sol — the most expensive tier —
and stayed there for every call of that child.

Text the router injects is therefore removed before classification and only before
classification: the prompt preview and the route log still show what was actually
sent. The contract warns a conductor never to restate routing policy in a leaf goal
for exactly this reason; this is the same rule applied to the router itself.

### Preempted tiers in the log

A route entry records not only the tier that won but the ones that independently
qualified and lost (`vetoed_by`). Without it a tier that never fires looks exactly
like a tier whose preconditions are never met — the log cannot tell you whether a
gate is dead or merely outranked.

### Shadow benchmarking

Off by default. When enabled, a turn can be run a second time on a forced tier and
both outcomes recorded, so a routing rule can be judged on results rather than on
the argument that produced it:

```yaml
shadow:
  enabled: false
  limit: 0
  path: ~/.hermes/logs/spark-shadow-benchmark.jsonl
```

`MODEL_ROUTER_BENCHMARK_FORCE_MODEL=<tier>` forces a single run from the
environment. The Spark read-only and image restrictions still apply — a benchmark
may not route work somewhere policy forbids.

### Retrying a failed downstream request

The plugin can re-issue one failed provider call through Hermes's `llm_execution`
middleware (`retry_call`), which is what lets a route that fails on arrival be
answered by a different model without failing the turn. It needs a Hermes new
enough to pass `retry_call` to execution middleware.

### Live Dashboard

```bash
~/.hermes/hermes-agent/venv/bin/python ~/.hermes/plugins/model-router/web_viewer.py
# http://localhost:8765
```

Run it with the Hermes venv's own python, not a bare `python3`: Refresh on an
account card calls into Hermes's `agent.account_usage`, which only that venv
has installed.

On Windows Hermes lives in `%LOCALAPPDATA%\hermes`, not `~/.hermes`:

```bash
"$LOCALAPPDATA/hermes/hermes-agent/venv/Scripts/python.exe" "$LOCALAPPDATA/hermes/plugins/model-router/web_viewer.py"
```

The dashboard finds Hermes's state database, agent log and config the way Hermes
does — see [Where Hermes lives](#where-hermes-lives).

Three tabs: **Model Router**, **Settings**, and an embedded **Hermes Command
Center**. The interface is available in English and Hungarian.

The router tab opens with a counter card per tier, grouped by account with a
usage bar for each account — including the three Claude tiers, whose children the
router records without routing them — then routing decisions grouped by prompt,
each expandable into its individual API calls, with a grouped/raw toggle, tier
filters and search. Every prompt row carries delegation chips saying which account
(Codex or Claude) took each delegation of that turn.

Below that, the live agent tree. It reads Hermes's durable delegation registry
and the `turn_lifecycle` table, and nests running and recent children under their
parent session with a privacy-safe task preview, state and age. Each row carries
`own N · total N` — the calls that agent made itself, and the calls its whole
subtree made — so a conductor that is quietly doing the work instead of
delegating it (`own 21 · total 21`) is visible at a glance, as is a leaf that
correctly does not delegate further (`own 9 · total 9`). The pills beside it name
the tiers those calls went to; an `external` pill means the router observed the
call rather than routing it, because the model belongs to another provider.

Everything refreshes every 3 seconds.

The Settings tab opens with **one
card per account**, with Models, Limits and Delegation side by side, then the
weekly and 5-hour usage bars, whose last line carries the read age, the Refresh
button and the account's recent call count for the same window the conductor
is given. Inside Models, each switch and title has its effort dropdown immediately
beside it: Codex shows its four plain-tier controls; Grok shows its `effort.grok`
control; Claude shows Sonnet and Opus, plus a disabled “no reasoning allowed”
select for Haiku. When the host seam is
incompatible, Claude's disabled selects carry the escaped reason as a tooltip
(see [Claude reasoning effort](#claude-reasoning-effort)). Qwen has no effort
select. A tier switched off keeps its switch there, so it can be
turned back on. A tier that is enabled but cooling carries a pill with the
remaining time and the reason, since the switch alone would not explain why
traffic went elsewhere. The Claude card's Delegation row reports `delegate_claude`
on while at least one Claude model is switched on (see
[Switching Claude on](#switching-claude-on)). Below the cards, the **Load
balancing** section holds its switch and two thresholds (see
[Load balancing between accounts](#load-balancing-between-accounts)); it appears
while at least two accounts have a model switched on. Then come the main agent chain and the worker defaults
(see [Hermes fallback chains](#hermes-fallback-chains)), the per-work-kind
preference chains described above (each entry shows its account and that
account's soft/closed state), and the interface language. Everything is read
through the router's own helpers rather than recomputed, so the panel and the
routing decision cannot disagree.

The server binds to `127.0.0.1` only, so it is not reachable from the local
network. Every setting on that tab writes to `router_config.local.yaml` (see
[Configuration](#configuration)) and takes effect
on the **next routed call, with no restart** — the config is re-read on every
decision rather than cached. Changing the plugin's *code* does need a restart of
the Hermes process that loaded it.

The shipped `router_config.yaml` is never written by the dashboard. Saves into
the local file keep its comments and layout when `ruamel.yaml` is installed in the
venv the dashboard runs with; without it they fall back to a plain YAML dump,
which drops comments.

## Configuration

The plugin loads `router_config.yaml` from the plugin directory automatically.
That file holds the shipped defaults -- Codex and Qwen on, Claude and Grok
switched off until you have their subscriptions -- and is the
one under version control. Your own settings go in `router_config.local.yaml`
beside it: git-ignored, and merged over the shipped file mapping by mapping, so
it only needs the keys you change:

```yaml
# router_config.local.yaml
default_model: terra
callable:
  opus5: true
  sonnet5: true
  haiku: true
preferences:
  review: [sonnet5, opus5, terra]
usage_guard:
  accounts:
    anthropic:
      soft_percent: 80
```

The dashboard reads the merged result and saves into the local file, keeping
only what differs from the shipped one (it creates the file on the first such
save). A local file that fails to parse is ignored with a warning, and the
shipped settings apply. An override can change a shipped key but not remove it.

### Where Hermes lives

Paths in the config are written as `~/.hermes/...`. That prefix means *Hermes's
home*, and `hermes_paths.py` resolves it the way Hermes does: Hermes's own
`get_hermes_home()` when the plugin runs inside Hermes (so a profile's home is
honoured), and otherwise `HERMES_HOME`, then `%LOCALAPPDATA%\hermes` on Windows,
then `~/.hermes`. Any other path is only user-expanded.

Before this, `~/.hermes` was literal. On Windows, where that directory does not
exist, the router read no `delegation.targets` from Hermes's config, so a Claude
parent was never recognised as a delegation target and ran without a conductor.
Its logs, cooldowns and usage state went to a `~/.hermes` that Hermes never
reads, and the dashboard read an empty state database beside them.

### Key Settings

```yaml
# Model availability
callable:
  luna: true
  spark: false   # disabled here; [spark] leaves follow the `fallbacks` chain
  terra: true
  sol: true
  opus5: true
  sonnet5: true
  haiku: true    # the shipped file has all three Claude models off
  qwen: true

# Which account each tier spends. Claude tiers are delegation targets rather
# than routable tiers, but they are still counted and switched here.
tier_providers:
  luna: openai-codex
  spark: openai-codex
  terra: openai-codex
  sol: openai-codex
  opus5: anthropic
  sonnet5: anthropic
  haiku: anthropic
  qwen: qwen-token
  grok: xai-oauth

# Default parent model (the shipped file has qwen)
default_model: terra

# Preferred models per kind of work, best first (see the section above).
# Unset kinds keep their built-in route.
preferences:
  review: [sonnet5, terra]
  design: [sol, opus5]
  explore: [spark, luna, haiku]   # skipped while haiku is switched off

# Delegation limits (in ~/.hermes/config.yaml)
delegation:
  max_concurrent_children: 2
  max_spawn_depth: 2
  max_iterations: 16

# Audit logging
logging:
  prompt_preview_chars: 240
  redact_prompt_preview: true

# Orchestration: the parent hands the objective to a conductor, which
# plans the work and delegates the leaves.
orchestration:
  enabled: true
  min_chars: 1000      # too short to decompose; skip the planner round trip
  max_tasks: 2         # must not exceed delegation.max_concurrent_children
  rescue_min_calls: 6  # a turn this deep with no worker gets one late checkpoint

# Benching a tier that just refused. quota_seconds is only the fallback for a
# provider that does not say when the allowance returns.
cooldown:
  enabled: true
  quota_seconds: 900          # no reset hint in the error
  quota_max_seconds: 21600    # ceiling for a provider-stated reset
  unavailable_seconds: 21600  # a 400/404 "model not supported" refusal
  allowed_fails: 3            # repeated non-quota failures within the window
  failure_window_seconds: 60
  failure_seconds: 60
```

`max_tasks` above `max_concurrent_children` is a hard error, not a partial
run: `delegate_task` rejects the whole batch. Keep them equal.

Declining to dispatch is logged too. `terra-spark-orchestration.jsonl` records
`preflight_forced` when a conductor is created and `preflight_skipped` — with
the gate that rejected it — when one is not, so a turn that ran twenty calls
with no worker says why.

### Usage reporting

Spreading work across accounts used to be an instruction with nothing behind it:
the conductor was told to use separate accounts, but could not see that one had
taken every call for the last hour and another had taken none. The routing
contract now carries a live figure read from the router's own log:

```
Recent load over the last 60 minutes, in calls per account: openai-codex 82.
These are call counts from this router's own log, not quota readings — read
them as relative load. qwen-token has taken none in this window.
```

Call counts, deliberately: the runtime does not report tokens or cost to the
route log, so a percentage would be invented. The sentence states the count and
stops there — "no calls" reads as spare capacity, but it is equally what an
exhausted account looks like, and an earlier version of this line recommended an
account whose weekly quota had already run out. Only the tail of the log is
parsed, since it reaches tens of megabytes and this runs on the preflight path.

A cooling target is annotated in the same sentence rather than dropped from the
list. LiteLLM excludes a deployment that is over its limit, but its deployments
are interchangeable and these are not — hiding a cooling Sol would invite the
planner to send design work somewhere the classifier then refuses outright.

```yaml
usage_report:
  enabled: true
  window_seconds: 3600
```

### Switching Claude on

Claude needs a Claude subscription, so, like Grok, its three models ship
switched off:

```yaml
callable:
  opus5: false
  sonnet5: false
  haiku: false
```

To use it, log in with Claude Code (Hermes borrows that login for its
`anthropic` provider; see [Claude targets](#claude-targets)), then switch the
models on in the dashboard's Claude card, or in `router_config.local.yaml`:

```yaml
callable:
  opus5: true
  sonnet5: true
  haiku: true
```

Claude is then available exactly while at least one of the three is on:
`delegate_claude` is offered, the conductor is advised the switched-on Claude
targets, and preference chains that name them take effect. With all three off,
none of that happens, chains skip the Claude entries (see
[Preferred models per kind of work](#preferred-models-per-kind-of-work)), and
a Claude child already running is stopped before its next provider call. A
Claude parent keeps its own provider: the switches do not rewrite Hermes's
provider configuration.

The switches are read live. The router rereads them on every request, so
routing follows from the next one. `delegate_claude` is registered whenever the
host can carry it, and its `check_fn` decides whether Hermes offers it, so a new
session gains or loses the tool without a restart. Hermes memoizes tool lists
without rerunning check_fns, so the router clears that memo (a private
`model_tools` helper) when Claude's availability flips, on a routed request or
before a gateway message is dispatched.

**Older configs.** Before 1.21.0 a `workflow: codex | claude_delegation` key (and
`claude_delegation.enabled`) switched Claude on and off. Both are retired. A
`router_config.local.yaml` that still has them keeps working: at load time the
router reads `workflow: codex` (or `enabled: false`) as all three Claude models
off, and `workflow: claude_delegation` (or `enabled: true`) as on for each Claude
model the local file does not switch itself. The dashboard shows the same. Its
first save writes those three switches into the local file and drops `workflow`
and `claude_delegation.enabled`; a later save removes any switch that equals the
shipped default, as every save keeps only the delta, with no change in effect.
A `workflow` posted by a dashboard tab opened before the upgrade is
ignored.

### Claude delegation and the usage guard

Two config blocks, one shared mechanism (`usage_guard.py`), used differently by
each account.

`claude_delegation:` registers the `delegate_claude` tool, which reaches a
Claude tier (haiku/sonnet/opus) as a real worker without going through the
router's own tier map:

```yaml
claude_delegation:
  default_tier: sonnet  # used when a call to delegate_claude names no tier
  log_path: ""          # JSONL audit log: registration + one line per delegate_claude call
  tiers:                 # model each short tier name actually starts
    haiku: claude-haiku-4-5-20251001
    sonnet: claude-sonnet-5-5
    opus: claude-opus-5-5
```

One tier per call: `haiku` for quick lookups and exploration, `sonnet` as the
everyday worker, `opus` for hard or consequential work. They count on the router's
side as `haiku`, `sonnet5` and `opus5`, all on the `anthropic` account, and Haiku
is reachable only this way (it has no `delegate_task` target). Clearing a tier's
model in `tiers` removes that tier from what the conductor is offered.

The tool is registered whenever the host has the delegation API it needs, and
offered while at least one Claude model is switched on in `callable` (see
[Switching Claude on](#switching-claude-on)). A host that defers tool registration
until first use (Tool Search) may not offer `delegate_claude` immediately after
startup; the dashboard's Claude account card shows whether it is actually live
(`delegate_claude live`), and flags `restart Hermes to apply` only when
delegation is on but the tool never registered.

`usage_guard:` is the same soft/hard usage guard for every account
`tier_providers` names, keyed by account (`anthropic`, `openai-codex`, ...),
not by tier:

```yaml
usage_guard:
  cache_seconds: 300   # how long a fetched reading is trusted before refetching
  state_path: ""       # shared JSON cache across the interactive TUI and the gateway process
  accounts:
    anthropic:
      soft_percent: 70   # at/above this weekly %, the heaviest routed tier steps down
      hard_percent: 90   # at/above this (weekly OR 5-hour session), delegation to the account closes
      step_down:
        opus5: sonnet5
    openai-codex:
      soft_percent: 70
      hard_percent: 90
      step_down:
        sol: terra
```

An account absent from `usage_guard.accounts` is never touched: no reading is
fetched for it and its state reports `unknown`. Codex is stepped down by the
router itself (`_usage_step_down`); Claude is stepped down inside
`delegate_claude`'s own dispatch, since the router never routes a Claude call
in the first place. Either way the dashboard's account card shows the current
reading, the configured limits, and — while cooling — the pill's reason.

#### Load balancing between accounts

The soft/hard limits react to one account reaching a line. `balance` compares
the two, so a busy account is spared before it hits one. It runs after the
soft/hard guard has ordered the chain, and only moves targets that are switched on:

```yaml
usage_guard:
  balance:
    enabled: true        # off when the block is absent
    window: 5-hour       # compare the 5-hour windows (default), or `tighter`: the higher of weekly and 5-hour
    busy_percent: 20     # the first account's window is at least this
    margin_percent: 10   # and the next account in the same chain is at least this many points freer
```

The aim is roughly equal 5-hour usage on both accounts. The Opus parent's own
calls count on Claude's window, so Claude usually leads and delegation leans to
Codex; with the numbers above, Claude at 23% and Codex at 12% sends the next
review to Terra. The weekly window is left to the soft/hard limits: a Claude
parent keeps Claude's week ahead, and balancing on it would pull every
Claude-preferred kind to Codex for the rest of the week.

When both hold, the other account's first entry in the kind's chain moves to
the front: a busier Claude sends `review: [sonnet5, opus5, terra]` to Terra, and a
busier Codex sends `code: [terra, sonnet5]` to Sonnet. Only targets the chain
already lists move, so a kind whose chain stays on one account (or has no
chain) is untouched, and the Opus parent never moves. A reading that is missing,
or older than twice `cache_seconds`, turns balancing off for that turn.

The switch and both thresholds are also in the dashboard's own Load balancing
section, shown while at least two accounts have a model switched on. The routing advice and the forced planning
call follow the balanced order, and
say why in one line (`Balanced: review → terra first (Claude 5-hour 86% vs
Codex 6%)`); the orchestration log's `preflight_forced` event carries the same
text as `balanced`.

### Cooldowns

A tier that just rejected a call for quota is not a candidate for the next one.
A 429 puts it in cooldown; repeated failures inside a window do the same.

```yaml
cooldown:
  enabled: true
  path: ~/.hermes/state/model-router-cooldowns.json
  quota_seconds: 900
  allowed_fails: 3
  failure_window_seconds: 60
  failure_seconds: 60
```

Failures are recorded whatever provider served them. The execution middleware
returns early off-provider because it rewrites `request["model"]` within one
provider, but noticing that an account refused a call needs none of that — and
skipping it meant a weekly-quota 429 on the second account left no cooldown at
all, while the load report kept describing it as the one with no traffic.

The state is a file rather than process memory because the interactive TUI and
the gateway are separate processes — a note kept in memory would not be seen by
the one that needs it. A cooling tier is simply not callable, so the existing
fallback chain and the policy rule below both apply with no extra wiring: a
preference route moves on, a policy route says which tier is cooling and for
how long.

A quota cooldown lasts as long as the provider says, not as long as the config
guesses. Codex answers a usage-limit 429 with `resets_in_seconds` / `resets_at`;
that value wins, capped by `cooldown.quota_max_seconds` (6h) so a malformed hint
cannot bench a tier for a day. `cooldown.quota_seconds` remains the fallback for
providers that say nothing.

This matters more than it sounds. A three-hour account reset benched for the
configured 15 minutes produces a loop: the cooldown lapses, the tier is offered
again, and the next leaf spends its retries rediscovering the same wall.

A usage quota also belongs to the **account**, not the model. `tier_providers`
says which account each tier spends, so one tier's quota 429 benches its siblings
for the same duration — all four Codex tiers together, or Opus, Sonnet and
Haiku together. Targets on other accounts are untouched, which is the point: the
planner should be reaching for them.

### A worker stopped by a quota comes back as a re-dispatch

A leaf that dies on an account limit has not failed at its task, but nothing in
the delegation envelope says so. It reports the goal, the status and the
provider's error, and the conductor is left to guess whether to retry, re-plan or
drop — while re-sending the same goal to the same target fails identically until
the cooldown lapses.

Everything needed to answer that is already here: the cooldown state says which
accounts are refusing calls and for how long, the preference chain says what comes
next, and the classifier that decides every route can tell what kind of work the
goal is. So a delegation outcome carrying a quota-stopped task comes back with the
target named:

```
[ROUTER — A WORKER STOPPED ON AN ACCOUNT LIMIT]
- Implement and commit the backend portion of the reliability ledger.
  code work -> re-dispatch with model:opus5
Re-dispatch each one with the model: parameter named above and tell the retry to
continue from what the stopped worker already committed in its worktree instead
of starting over. Do not re-plan or narrow the goal: only the account changed.
```

With Claude switched on, the target is named by the call that reaches it:
`re-dispatch with delegate_claude(tier="opus")` for a Claude target, and
`delegate_task (goal prefix [terra])` for the others.

When the whole chain is cooling it says what to wait for and for how long, so
waiting stays a legible option instead of a guess.

### A delegation that failed before any worker existed

The notice above reads a delegation *outcome*. Sometimes there is none: the tool
returns an error inline, no child ever runs, and nothing will be delivered later
to explain it.

The error is also not a fact about the account it names, and this one is worth
stating precisely because it is easy to get backwards. `delegate_task` resolves
the configured **default** delegation provider once for the whole call — at
`delegate_tool.py:496`, before `_normalize_task_list` has even parsed the tasks
— and returns `tool_error` if that fails. The per-task target is resolved much
later, inside the dispatch loop.

**So an unavailable default provider blocks every delegation, including a task
that names a target on a healthy account.** Its `model:` value is never read.
With Codex exhausted, a `model: "opus5"` task fails on the Codex quota although
Opus 5 runs on Anthropic and was never contacted:

```
Cannot resolve delegation provider 'openai-codex':
Codex provider quota exhausted (429); retry after 3731s.
```

The parent's own conclusion — "the Codex quota is out, so the opus5 delegation
failed" — is therefore literally correct, however wrong it sounds.

The notice says that, and deliberately does **not** advise a retry with a
different `model:`, which would loop. It separates the targets that are
themselves healthy from the fact that none of them is reachable, and names the
three ways out: repoint `delegation.provider`/`delegation.model` at a working
route, do the work in the turn, or wait.

The real fix is in the host: resolve the default lazily, or tolerate its failure
when every task names its own target.

**It names the target and stops there.** Re-dispatching by itself would be the
hardcoded selection this design exists to avoid: what to do with a stopped leaf —
retry, narrow, wait, drop — is the conductor's call.

Both outcome shapes are answered: the consolidated batch envelope, and the early
single-child notice that arrives while siblings are still running, which exists
precisely so the conductor can act then rather than at batch end. The off-provider
path answers too — a `code` chain starting with `opus5` puts the conductor itself
on Claude, and that branch returns early, so the setup that needs this most would
otherwise have been the one to miss it.

Two false positives are excluded by construction. An ordinary failure is left
alone, because only a limit is safe to re-send unchanged. And the reason is read
from the envelope's own status/error lines rather than from the whole block — a
worker whose subject *is* quota handling otherwise reports itself as
quota-stopped, which the leaves of this very plugin do.

### Policy routes do not fall back

`fallbacks` exists for preference: a long request prefers Sol for capacity, and
demoting it to Terra is a quality trade. But some routes are policy — design work
reaches Sol because *only* Sol may do it, and consequential work escalates there
for the same reason. Satisfying those from the fallback chain would perform the
work on the tier the rule exists to keep it away from, precisely when Sol is out
of quota and the rule matters most.

Such a decision is marked at the point it is made and declines the chain, so a
disabled Sol fails loudly instead of quietly landing design work on Terra.

This is the *default*. A preference chain configured for that work kind replaces
it — see above — because the operator asked to own the mapping. A single-entry
chain (`sensitive: [sol]`) keeps the loud failure; add a second entry only if you
would rather the work continue elsewhere than stop.

### Runtime capability matrix and worker topology

`runtime_capabilities.py` builds a read-only, frozen snapshot of what the host
could support: spawn depth, concurrent children, the orchestrator switch, the
checked delegation seams, and per-transport capabilities (`hermes_codex`,
`hermes_claude`, `claude_cli`) for submission, model parameter, effort
application, cancellation, async delivery, fallback ownership and mixed-target
batches. Each entry is `supported`, `unsupported` or `unknown` with a reason; a
missing, false or unverifiable seam is never `supported`. Discovery reads the
request's tool schema and in-process host settings only: no model probe,
subprocess, network call or config write.

The default topology is `parent_direct` (the parent plans, workers run directly,
host depth untouched). `nested_conductor` is granted only when depth is at least
2, the orchestrator role is enabled, the host seams are callable and the chosen
transport's submission is `supported`. An explicit nested request that fails
returns `unsupported` with a reason and `parent_direct` as the selected topology;
the router never raises the host depth, and a configured conductor alias alone
selects nothing.

`model_router.runtime_diagnostic(request, cfg)` returns a JSON-able view that keeps
three things apart: *configured* features (router config), *runtime* capability
(snapshot) and the *topology* (requested/selected; `active` is always `unknown`,
since a snapshot cannot observe the running topology). Snapshots are cached (30 s,
16 entries, locked) under a fingerprint of host limits, schema, seams and routing
config, so any change invalidates them. A snapshot is evidence, not admission:
fresh admission is still required immediately before execution. The diagnostic
is not called on the routing path, so existing routing decisions are unchanged.

### Target identity and configuration drift

`execution_contracts.py` holds the frozen target-identity record: provider/account,
transport, operator alias, requested/resolved/observed model (each with its evidence
source), selection mode (`exact` or `profile_preferred`) and effort (requested/applied,
with explicit `unknown`/`not_applicable`). `exact` is never substituted silently: a
disagreement is a typed `ExactRouteMismatch` (failure class `exact-route-mismatch`);
`profile_preferred` records the substitution. `target_identity.resolve_target()` builds
the record from router config, `claude_delegation.tiers`, the host's named targets and
the CLI alias map, cached by a fingerprint (30 s, 16 entries, locked) so a changed
alias, tier or capability invalidates it. The Claude CLI is invoked with an alias, so
its resolved identity is that alias; exact canonical CLI selection is `unsupported`
unless `runtime_capabilities` reports `exact_model` as `supported`.

`model_router.identity_diagnostic()` is a read-only, no-network view that lists each
config owner's value per Claude tier and flags drift and observed mismatches.
`target_identity.propose_host_migration()` returns an optional unified diff for
`~/.hermes/config.yaml`; nothing is ever applied. No routing behaviour changes.

### Execution records and offline adapter doubles

`execution_contracts.py` also freezes the scheduler-facing records without adding a
scheduler or dispatch path: a bounded `ExecutionRequest`, an `AttemptLifecycle`, a
non-terminal `Submission`, and a terminal-only `WorkerResult`. The main records
and adapter capability record carry explicit schema versions; boundary records
have strict `as_dict()`/`from_dict()` validation: unknown
routing fields, malformed IDs, invalid types, illegal lifecycle transitions and
oversized inline payloads are rejected. A submission acceptance is only a handle;
it cannot be consumed as a completed worker result. Execution success remains
separate from validation evidence and later parent acceptance.

Unknown effort and provider usage remain explicit (`"unknown"`) and provider usage
may carry bounded future meter fields. Inline output is bounded; callers retain a
bounded artifact reference plus failure and validation evidence rather than silently
dropping those records. `OutputReference.bounded()` and `bounded_evidence()`
require a caller-supplied full-content artifact reference when inline detail is
oversized; they do not persist or verify the referenced artifact. Failure message
and details have their own output references. Sequences are normalized to tuples,
and nested usage/metadata are defensively copied into immutable collections.
Usage/metadata have depth, finite-number, string and aggregate payload bounds;
credential-named metadata fields (including nested API keys, cookies and bearer
fields) are redacted before storage. This is key-based redaction, not a guarantee
that arbitrary free text is secret-free; adapters must avoid supplying secrets.

`legacy_direct_tool_request()` gives direct tools stable `legacy:<tool-id>`
workflow/task/attempt identities without a TaskGraph (long IDs use a SHA-256
suffix). Callers must explicitly supply permissions, mutation/write scope,
acceptance criteria, tools, budgets/deadline and policies; the wrapper invents no
authority or already-expired deadline.

Lifecycles/results retain plan version, and results include changed-file paths.
Post-submission states require the original handle; transitions cannot replace it.
`unknown` cannot transition ordinarily: `reconcile()` explicitly records the
caller's reconciled running/terminal observation without releasing workspace
ownership or implementing reconciliation policy. Fake cancellation preserves
terminal results and unknown liveness. Non-success results require typed failures,
and success cannot carry failure or contradict a retained exact model/provider/
effort constraint. Unknown observations are still unknown: execution success does
not prove exact-route observability, verification or parent acceptance; production
adapters/admission must enforce those requirements before accepting exact work.
Retryability is a policy-owner input, not authorization inferred from model prose.

`execution_test_doubles.py` supplies no-I/O scripted adapters for delayed completion,
partial batches, capability refusal, observed-identity mismatch, unknown liveness
and cancellation. They freeze the future adapter seam (`capabilities`,
`can_execute`, `submit`, `result`, `cancel`) for S05 without wrapping production
transports. TaskGraph construction (S08), event persistence/reconciliation (S06),
scheduling and verification remain deferred.

### Transport adapters

`execution_adapters.py` wraps the actual legacy transport boundaries. Registered
`delegate_task` execution retains the existing account guard as its sole admission
owner; `delegate_claude` retains its existing admission/step-down and scoped effort
handler, then invokes an explicitly raw host callable. Every Claude CLI entrypoint
(the standalone `claude_opus_bridge.py` script, direct Python calls to
`claude_opus_bridge.dispatch` and the middleware bridge) crosses one normalised CLI
boundary exactly once, owned by the public `dispatch`: it validates input, then
records one receipt around a private raw subprocess operation that never calls a
public entrypoint back. Invalid input and an exact refusal raise before the boundary
(nothing launched, no receipt); an attempted run leaves one receipt, one lifecycle
`started`/`terminal` pair and, only on a validated success, one routed-call record.
A caller that passes no identity gets legacy facts: requested model from the
explicit tier, resolved value from the actual `--model` alias argument, observed
model only from the validated CLI result; account stays `unknown` and nothing else
is invented. A caller's own identity is passed unchanged, including the exact
canonical model argument; an exact identity is re-checked against the CLI
`exact_model` capability before launch, so a direct call cannot bypass it. No
boundary calls its own public entrypoint recursively or performs a second account
admission.
Legacy goals, context, supported tool payloads and host-owned budgets/permissions
remain authoritative. They are **not** converted into fabricated `ExecutionRequest`
permissions, criteria or deadlines.

Structured requests are more restrictive than those legacy tools. On the current
transports, the adapter cannot enforce the complete mandatory request contract:
trusted context/revision resolution, permission and required-tool fit, workspace and
write-scope enforcement, deadline/timeout ownership, credential-account binding and
substitution policy must all be proven. Consequently **all current structured submit
cells explicitly return typed `unsupported`/capability refusal before dispatch**.
A supported host model or effort seam alone does not make a full request supported.
Requested Claude effort is not silently replaced with configured per-tier effort;
Haiku accurately remains `not_applicable`. Capability lists advertise implemented
adapter methods only, not host cancellation or structured submission that the
adapter cannot deliver. There is no scheduler or activation switch in this slice.

Legacy responses are returned unchanged and separately normalized into bounded,
process-local execution receipts. Real synchronous results (including background to
synchronous fallback) are terminal execution evidence even without a top-level
handle; they do not prove acceptance criteria or served native model identity.
Claude tier-adjustment evidence is retained as resolved selection, not served model
proof. CLI observed model evidence comes only from its validated bridge result.
Malformed/partial responses, background acceptance and ambiguous exceptions remain
`unknown` in the receipt; a top-level `{"error": ...}` payload and a typed
`ClaudeBridgeFailure` are recorded as `failed` (`timed_out` for a CLI timeout).
None of them keeps the record in flight after the call returns: the host exposes no
completion signal correlated to an accepted background handle, so the legacy
boundary never refuses a dispatch the unchanged host would accept.

The legacy boundary adds no capacity check of its own: for `delegate_task`,
`delegate_claude` and the CLI bridge the unchanged host (its batch-width check and
async-pool fallback) is the sole admission authority, so distinct concurrent calls on
one parent are never refused by the plugin. One locked journal is shared by the
native transports per active parent. Missing host delegation support refuses before
launch. Full attempt keys prevent reuse
across workflows; where the host supplies session/turn/tool-call IDs, the public
boundary deduplicates that invocation and rejects changed payloads under the same
IDs. A legacy call with no stable invocation IDs receives observation-only IDs: a
later new call is not invented to be the same attempt. In-flight duplicates do not
dispatch again, and retained terminal duplicates replay instead of re-running. Each
completion and public seal is bound to the exact claim that started it, so a late
invocation can never overwrite, replay into or unprotect a newer claim under the same
IDs; a public invocation's record is pinned against eviction until its final response
is sealed or the invocation unwinds. CLI runs keep their pre-S05 concurrency (no
plugin cap).

S06 owns persistence, background completion correlation and unknown-work
reconciliation. The receipt journal keeps in-flight and seal-pending public records
unconditionally and
finished records (terminal, unknown, observation-only) for one hour, bounded to 4096
by least-recently-used eviction; a full journal evicts, it never refuses. While a
sealed session/turn/tool-call record is retained, an identical repeat replays its
cached response (up to 8192 characters) and an uncached or changed repeat refuses;
after eviction a repeat is treated as a new call. Receipts do not survive restart.
Do not automatically replay unknown work after a
restart. No production configuration/log write, provider probe or new routing-hot-path
subprocess is introduced. Running Hermes/dashboard processes need their normal
restart to load the changed boundaries.

## Usage

### Explicit Model Override

Prefix your message with a tag:

```
[luna] Simple question
[sol] Complex security analysis
[opus] Diagnostic review through the standalone bridge
```

A root turn is still assessed for you: design work reaches Sol whatever label
you type, and `[spark]` on a root turn defers to the orchestrator rather than
sending user-facing work to a read-only worker.

### Labels inside a plan

The same labels mean something stronger on a delegated worker, because there
the label was written by a conductor that saw the objective, the repository and
any screenshot — a better-informed decision than a keyword test on the goal
text. So a plan label is authoritative, and the design gate does not re-judge it.

The label still has to be true. A `[spark]` **or `[luna]`** leaf must actually be
read-only: one that writes is rejected, and one touching production, security,
credentials or payments escalates to Sol. Both are judged from the verbs,
independently of the subject matter — "identify the layout branches" is source
discovery, not design work.

Luna faced no check at all until 1.9.5, which is how a conductor handed it
"Stabilize, correct, test, and commit the dirty foundation now" and the router
obeyed. The write-verb list had a matching hole: `commit` was not in it, and
neither were any Hungarian imperatives, so the guard would have passed that goal
even once it existed.

Closing that hole opened another one, and the two changes were in the same
series. `commit` is also a noun, and the goal contract requires a goal to name
the commit it builds on — so "…at commit 7abc123" made a read-only source map
read as mutating, and a `[spark]` leaf was escalated off Spark for saying exactly
what it had been told to say. A commit *reference* is stripped before the
write-verb test; an instruction to commit still counts.

**`[opus5]` and `[sonnet5]` are not labels.** They read like `[sol]` and do the
opposite of what the writer meant: the override vocabulary knows only this
provider's four tiers, so the prefix is inert, the goal is classified on its
remaining text, and the leaf works to completion on the account the dispatcher
was trying to spare. A delegated leaf whose goal opens with one — while running
on one of this provider's models — is stopped at its first call, its tool use
switched off, and its single answer is the correction, which reaches the parent
as the leaf's own summary:

```
MISDISPATCHED: this goal names opus5 in its text, which is not a route.
Re-dispatch it unchanged with delegate_task(model="opus5").
```

Both facts are required. A leaf already on that account has a redundant prefix
rather than a wrong one, and a **root** turn carrying `[opus5]` is you asking for
Opus, not a dispatch bug. `[opus-review]` and `[sonnet-review]` are unaffected:
those are real labels, routed to the read-only CLI bridge.

The check looks for contradiction, not corroboration. A leaf that names no write
verb passes, because the conductor already declared it read-only by labelling it;
requiring a second positive signal would let a hand-written verb list overrule
that declaration on phrasing alone. A root `[spark]` is a label someone typed
with nothing behind it, and there the stricter form still applies: it has to show
its read-only intent.

### Delegation

The parent delegates independent bounded subtasks, and picks the route for
each one with `model`:

```python
delegate_task(tasks=[
  {"goal": "[spark] Read-only source discovery for the calendar renderer.",
   "model": "luna"},
  {"goal": "[sol] Diagnose and fix the card layout.",
   "model": "sol"},
])
```

`model` is what actually selects the route; the enum is built from
`delegation.targets` in `~/.hermes/config.yaml`, filtered to tiers that are
currently callable. `callable` and `delegation.targets` are separate switches, so
without that filter a planner can pick a tier the cross-provider guard then
refuses mid-session, producing a leaf that never runs. A goal-text prefix only renames
the model *inside the default provider*, so it cannot reach a target on another
account — a leaf meant for Qwen must carry `model: "qwen"`.

That rule was here from the start and lost anyway, seven goals running, because it
shares a paragraph with `[spark]` and `[sol]` — which *are* prefixes. `[opus5]` is
the obvious blend of the two mechanisms. Since 1.10.1 the contract names the
mistake rather than restating the rule, and a leaf that makes it is stopped
instead of quietly becoming a Sol leaf.

Claude is one of those targets, and a Claude leaf must carry every fact it needs
in its goal — it does not share the conversation:

```python
{"goal": "Diagnose and fix the fullscreen calendar card in /path/to/repo. …",
 "model": "sonnet5"}
```

With Claude switched on, the same leaf can also go through `delegate_claude`,
which takes the same `tasks` shape plus one tier for the whole call:

```python
delegate_claude(tier="sonnet", tasks=[
  {"goal": "Diagnose and fix the fullscreen calendar card in /path/to/repo. …",
   "context": "…"},
])
```

### A goal carries what the worker cannot see

That rule is not specific to Claude. **Every** delegated worker starts at
`history=0` on every target, so a fact the conductor knows and does not write
down is a fact the worker spends iterations rediscovering — against a budget it
cannot raise: `delegate_task` accepts a `max_iterations` argument and the host
ignores it, because `delegation.max_iterations` is authoritative "so budgets stay
predictable".

Measured on one Opus leaf: sixteen iterations, twenty tool calls (13 `terminal`,
6 `read_file`, 1 `search_files`), context grown from 20k to 56k, and **not one
edit**. The whole budget went on reconstructing a repository the goal never
described, because the goal was a product requirement:

> Implement a tenant-scoped, safe customer-profile merge capability for Booking
> SaaS: an authorized admin can review two duplicate customer profiles and merge
> a phone-only and email-only record into one canonical profile…

No path, no branch, no base commit, no files. The leaf that finished did so in
nine iterations with one write, and the only difference was its goal: it carried
its own state ("the user already ran the `ALTER USER` command, it succeeded") and
asked for a single artefact.

So the contract now requires every goal to state the absolute worktree path, the
branch and the commit it builds on, what already exists there, which files or
modules are in scope, and how the result is verified — and to give one worker one
finishable artefact rather than a feature to implement. A goal phrased as a
product requirement has no boundary, and it is spent before the first edit.

Raising `delegation.max_iterations` is the blunt instrument here, not the first
move: it is global, so it also widens every Codex leaf on the shared quota.

**These requirements ride on the `delegate_task` schema, not on the message.**
They used to travel only inside the forced preflight, so a turn that skipped it
delegated with nobody having been told what a goal must carry — and a root prompt
shorter than `orchestration.min_chars` (1000 in the shipped config) skips it, creating no conductor at
all. A twelve-character `inplementald` produced exactly that: a whole-feature
goal with no worktree, branch or base commit in it.

The schema is the right carrier because a middleware edit does not persist into
the conversation — that is why the preflight needs a rescue pass at all. The
parent may delegate on any call of the turn, so an appended sentence would have
to be repeated on every one of them; a tool description is read once, exactly
where the goal is written. When a preflight *does* fire, the conductor's contract
already carries the same rules and the schema is left alone.

**And `context` is required, because a description is advice.** Measured after
the description shipped and was live: the parent read it and dispatched
"read-only release readiness review of *the customer-profile merge worktree*"
anyway — no path — and the reviewer spent 31 shell commands over 16 iterations
without reaching a verdict. That was the third time a description lost, after
the built-in `sonnet5` tie-breaker and the `[opus5]` prefix rule.

So the per-task `context` — which the schema already describes as the place for
"file paths, error messages, constraints", and which genuinely reaches the child
— is moved into `required`, and its description says what belongs in it. A
context-free call is now invalid rather than merely discouraged, and on a tool
carrying `strict: true` the provider is the one enforcing it. This is the same
move the preflight already makes for `role` and `context` on a conductor, for
the reason its own comment gives: *natural-language instructions alone are not a
reliable control plane*.

## Diagnosing a parent that will not delegate

`~/.hermes/logs/terra-spark-orchestration.jsonl` records why a preflight did not
run. Read its tail first — the answer is usually one field:

```bash
tail -5 ~/.hermes/logs/terra-spark-orchestration.jsonl \
  | jq '{event, parent_model, skip_reason, tools_seen}'
```

`preflight_forced` means the contract went out and the parent was asked to
delegate; what it does next is the model's decision. `preflight_skipped` names
the gate instead:

| `skip_reason` | Meaning |
|---|---|
| `orchestration_disabled` | `orchestration.enabled: false` |
| `tier_not_orchestrator:<tier>` | neither Sol, `default_model`, nor a delegation target |
| `subagent_turn` | already a delegated child; children do not orchestrate |
| `no_delegate_task_tool` | no delegation tool in the request — `tools_seen` lists what was there |
| `sol_preflight_disabled` | `sol_opus5_preflight.enabled: false` |
| `delegation_completion_delivery` | the turn is delivering a finished child's result |
| `explicit_delegation_tool` | your message names `delegate_claude` or `delegate_task`, so the forced planning call steps aside and your choice stands |

`tools_seen` exists because `no_delegate_task_tool` reads identically whether the
request had no tools at all, the wrong wire shape, or a name the router did not
recognise — and those need different fixes. Establishing that distinction took a
live probe before the field was added.

If the parent *is* orchestrating and the work still lands on one account, read the
routing log instead: an `external delegation target` line means the router observed
the call rather than routing it, which is normal for Claude and Qwen.

## Policy

1. **Stable Parent** — The user-facing conversation does not silently switch models.
2. **Delegation is an exception, not the default** — Only for genuinely independent subtasks.
3. **Worker limits enforced** — Max 2 concurrent children, 2 spawn depth, 16 iterations.
4. **Privacy-safe audit** — 240-char bounded preview, redacted sensitive data.
5. **The plan decides the route** — On a delegated worker the conductor's label wins; the router enforces only what the label claims (read-only, non-consequential).
6. **Documented changes** — Every policy change updates README and tests.

## Tests

Balancing preserves usage-guard precedence: a soft-limited or closed account,
or a cooling target, cannot be promoted because its five-hour window is freer.
Unavailable targets remain visible in advice, but cannot win a forced Claude call.
The root and conductor use the same effective preference order. Local Codex tiers
remain candidates without named host targets; a conductor receives current order
advice on subsequent calls so an old capacity snapshot cannot override it.
Hard usage limits refuse new workers on both accounts. Running workers return a
router stop without a provider call once a fresh reading closes their account;
parent continuation remains available. Same-account model step-down is only a
soft-limit policy, never a remedy for an account's hard limit.
Usage step-down applies to workers only; it cannot change the parent model.

The tests import the plugin as the `model_router` package (and a few modules by
bare name), and the Hermes venv ships neither pytest nor pip, so they run under
`unittest` in a fresh scratch directory. Use the tracked harness rather than
copying a live Hermes configuration: it writes only a sanitized `delegation`
section, links the checkout locally, exports `PYTHONDONTWRITEBYTECODE=1`, and
passes the Python process exit status through unchanged. It resolves the default
Hermes venv relative to `ROUTER_TEST_PYTHON`; set that variable when the local
validated interpreter lives elsewhere.

```bash
# Full suite
ROUTER_TEST_PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python" \
  scripts/run_suite.sh --full

# The execution-routing compatibility modules
ROUTER_TEST_PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python" \
  scripts/run_suite.sh --focused

# Prove user delegation depth does not change the result
ROUTER_TEST_MAX_SPAWN_DEPTH=2 \
ROUTER_TEST_PYTHON="$HOME/.hermes/hermes-agent/venv/bin/python" \
  scripts/run_suite.sh --full
```

The full-page dashboard test requires Node.js and `jsdom`; the harness exposes a
nearby Hermes checkout's `node_modules` through `NODE_PATH`. Missing either
dependency fails the test instead of silently skipping the visible-filter check.
Test fixtures declare whether they require a depth-one direct-worker topology or
a depth-two nested conductor, so no test needs the user's `~/.hermes/config.yaml`.

Use a fresh `RUN` for every run. Some tests still write the default orchestration
log inside that home, and `test_root_parent_is_pinned_when_classifier_wants_sol_worker`
fails on a second run over the same one.

Every test is a `unittest.TestCase`, so the command above collects the full suite. They
were not always: `test_artifact_name.py`, `test_leaf_label_contract.py` and
`test_callable_and_qwen_guards.py` held 25 tests written as module-level
`def test_*` functions. unittest never collects those, so the first two reported
`Ran 0 tests ... OK` — passing by running nothing — and the third needed pytest
to import at all. Keep new tests in a `TestCase`; a bare `def test_*` is silently
skipped here.

## Version

**1.24.0** — Small-task budget and exact lifecycle accounting, after a pricing-copy
fix fanned out into a conductor plus two workers (about 100 routing decisions).
`task_budget` is now keyed on the root user turn, no longer exempts a goal for
mentioning "production", accepts one further worker only with `escalation_evidence`,
and covers `delegate_claude` as well as `delegate_task`. `low_risk_max_chars` is 1200.
The orchestration log keeps distinct phases of a turn, and `view_log` and the
dashboard collapse replayed async-delegation completions to one routing decision.
The gateway must be restarted to load the new `__init__.py`.

**1.23.0** — The Sol tier runs GPT-6.1 Sol (`gpt-6.1-sol`) instead of GPT-6 Sol.
The live Codex model list for the account offers it (first by priority). Only the
`models.sol` default moved; the tier key stays `sol`, so preference chains,
fallbacks and existing logs need no rewrite, and the dashboard still counts older
`gpt-6-sol` log lines under the Sol card. GPT-6.1 Sol has no `none` reasoning
effort; the Sol efforts the router sends (medium, high, xhigh) are all supported.

**1.22.1** — A write verb inside a noun phrase no longer overrules a `[luna]` or
`[spark]` label. A bug diagnosis must name the path that still works ("an
existing service color update works") or the flow it inspects ("the create/edit
data flow"), and that word read as an instruction, so a read-only Luna leaf ran
on Terra. `_VERB_AS_NOUN` gained two arms, both behind a determiner so an
imperative never matches: a verb followed by a behavioural predicate, and a verb
qualifying a noun such as flow, form or endpoint.

**1.22.0** — The Sonnet tier runs Claude Sonnet 5.5 (`claude-sonnet-5-5`)
instead of Sonnet 5. The `claude_delegation.tiers.sonnet` default and the
bridge's `sonnet` review model moved, so the bridge accepts a Sonnet review only
when `modelUsage` reports `claude-sonnet-5-5`. The target key stays `sonnet5`,
so preference chains, fallbacks and existing logs need no rewrite; the dashboard
still counts old `claude-sonnet-5` log lines under the same card. The Hermes
`delegation.targets.sonnet5.model` has to move with it.

**1.21.2** — Overview account groups: the weekly usage bar sits at the bottom
of its card even when the group holds a single model (Grok), so the bars line up
across groups, and the percentage and its reset time no longer run together.

**1.21.1** — A review goal naming a path the OS refuses (a component over the
name-length limit) no longer raises out of the `delegate_task` admission guard,
which Hermes would then skip; the lookup reads it as "no repository", and a
failed review lookup falls through to the ordinary target check.

**1.21.0** — Reasoning effort for delegated Claude and Grok, Claude as a
switchable account, and a Claude main agent from the dashboard.

- Delegated Claude reviews use the repository they were asked to review, have their
  own turn limit, and show a clear reason when the Claude review bridge fails
  before the work continues as an ordinary worker.
- `delegate_claude` children get their own per-tier reasoning effort:
  `claude_delegation.reasoning_effort.sonnet` / `.opus` (default `medium`) reach
  the delegated Sonnet or Opus child through a guarded bridge over Hermes's private
  child-runtime resolver. Haiku is unaffected, since Hermes's adapter sends it no
  thinking config, and a host whose seam has moved refuses `delegate_claude`
  rather than silently dropping the setting.
- Reasoning-effort dropdowns sit beside the model switches in the Settings
  account cards. The Codex card edits Luna, Spark, Terra and Sol; the Claude card
  edits Sonnet and Opus and shows a disabled "no reasoning allowed" select for
  Haiku; the Grok card edits `effort.grok` (`low`, `medium`, `high`, `xhigh`),
  which the router forwards to Grok 4.7. A switched-off model's dropdown is
  disabled, an unavailable one gives its reason in the tooltip, and an unrelated
  save no longer pins an effective default into `router_config.local.yaml`.
- The *Codex only* / *Codex + Claude* workflow switch is retired. Claude is an
  account like Qwen or Grok: available exactly while at least one of `opus5`,
  `sonnet5` and `haiku` is switched on, and all three ship switched off, so an
  install without a Claude subscription works out of the box. A legacy
  `workflow:` or `claude_delegation.enabled` in `router_config.local.yaml` is
  translated into those switches at load time, and the dashboard's next save
  writes them into the file and drops both keys (see
  [Switching Claude on](#switching-claude-on)). Preference chains skip
  switched-off models and fall back to the built-in route when nothing in them is
  on. The dashboard's Workflow section is gone; load balancing has its own
  section, shown while two accounts have a model switched on.
- The main agent's first entry also offers the switched-on Claude models, so
  Hermes can be started on Opus (or Sonnet) from the dashboard. Picking one
  writes `model: {default: <claude model>, provider: anthropic}` into
  `~/.hermes/config.yaml` and leaves `default_model` alone. Picking a router tier
  moves a Claude parent back. The pick is sent only with the save it triggers, so
  no other save can move the parent.
- Hermes modules are found through the Hermes checkout when the venv's install
  map predates a newly added module (`hermes_yaml`). Before, usage Refresh failed
  and the Claude effort dropdowns stayed disabled until a Refresh happened to
  fix the import path. A refused save names the YAML error in
  `~/.hermes/config.yaml`, and the effort-save tests no longer write their stub
  over the Hermes config in `HERMES_HOME`.
- The Claude Code bridge imports inside the plugin package. Under Hermes it is
  loaded as `hermes_plugins.model_router`, where its top-level `model_router`
  import failed, so the `delegate_task` admission guard raised on every call and
  dispatches went out without the usage-limit or Claude-switch check.
- Hermes's appended `<memory-context>` and Superpowers bootstrap blocks are
  removed before the delegated-leaf classifier runs, so their instruction words
  cannot promote a read-only `[luna]` or `[spark]` leaf to Terra. The raw request
  remains available to logs and the Claude CLI bridge.

**1.20.0** — The objective the parent writes decides whether a conductor is
spawned, not the length of the user's message: a short follow-up is preflighted,
and a composed goal under `orchestration.min_goal_chars` returns to the parent
without spawning anything.

**1.19.0** — Every parent tier orchestrates, not only Terra and Sol: a Grok,
Qwen, Luna or bridge-less Sol parent gets the forced conductor preflight, and the
conductor follows the parent's own tier by default
(`orchestration.conductor_follows_parent`), so planning stays on the account the
session runs on instead of always returning to Codex.

**1.18.1** — A tier switched off in Settings leaves the Model Router overview (and
its account box once no tier is left), and it is dropped from Hermes's fallback
chains on save and no longer offered there: Hermes fails over without consulting
the router's switches, so a switched-off Qwen stayed a live fallback.

**1.18.0** — A `grok` tier (Grok 4.7) on a SuperGrok subscription through Hermes's
`xai-oauth` provider: a delegation target and `heavy` peer, with its own dashboard
card, colour and account. It ships switched off. A Grok parent is written with the
Responses API instead of `chat_completions`.

**1.17.1** — A `pre_tool_call` gate refuses a `delegate_task` that names a target
switched off in the dashboard (or cooling down with no worker fallback chain) and
points the agent at a working route; before, the call reached the disabled account.

**1.17.0** — The Settings tab now edits the plain `effort` values for Luna,
Spark, Terra and Sol. The picker offers the four shared provider levels — `low`,
`medium`, `high` and `xhigh` — and writes only its changed tier into
`router_config.local.yaml`; `router_config.yaml` remains the shipped default.
Route markers and situational effort keys, including `opus5`, `sol_long` and the
`explicit_*` keys, stay file-only because they do not mean one ordinary tier
setting.

**1.16.0** — The Luna and Sol tiers run GPT-6 Luna (`gpt-6-luna`) and GPT-6 Sol (`gpt-6-sol`) instead of GPT-5.6. The `models.luna` and `models.sol` defaults moved together, while the tier keys stay `luna` and `sol`; preference chains, fallbacks and existing logs need no rewrite. The dashboard still counts older GPT-5.6 Luna and Sol log lines under the same cards.

The dashboard Settings tab shows the main agent as one chain (default model first, then
Hermes's fallbacks; the primary is dropped from its own fallback list), a Workers
section gathers the `delegate_task` model, the `delegate_claude` default tier and
the worker fallback chain, and each account card takes two lines instead of five.

**1.15.0** — The Opus tier runs Claude Opus 5.5 (`claude-opus-5-5`) instead
of Opus 5. The `claude_delegation.tiers.opus` default, the bridge's canonical
model and the Sol preflight's `bridge_model` all moved, and the bridge accepts
a run only when `modelUsage` reports `claude-opus-5-5`. The target key stays
`opus5`, so preference chains, fallbacks and existing logs need no rewrite; the
dashboard still counts old `claude-opus-5` log lines under the same card.

**1.14.5** — The dashboard's two workflows are named for what differs between
them. *Codex (original)* said only which one came first, and *Claude
delegation* only that Claude is involved; neither named the thing the choice
turns on, which is how many accounts do the work. They are now *Codex only* and
*Codex + Claude* (*Csak Codex* / *Codex + Claude*), and each description opens
with that -- "One account works: ..." / "Both accounts work: ..." -- before the
mechanics. The `workflow:` values are untouched: `codex` and `claude_delegation`
still name themselves in the file, so nothing saved needs rewriting.

The Codex description also stopped saying Opus stays available as the Hermes
parent. The parent tier is chosen at spawn time and is not always Opus, so the
line read as a guarantee the workflow does not make; it now says the Claude
account stays available as the parent.

**1.14.4** — Two more shapes a read-only goal cannot avoid. A write verb
used as a noun ("a test case that would fail before the requested edit") and
one the goal explicitly refuses ("make no edits") no longer read as
instructions to write. Both are stripped as phrases, so a real instruction
standing beside them still counts: "make no edits but rewrite the config" is
still a write. The adjective between the determiner and the verb comes from a
closed list, because an open one walked over the noun in front of a genuine
instruction and swallowed it.

Observed 2026-09-21: a `[luna]` goal mapping issuer-mode behavior ran seven
calls on Sol as "consequential Luna task requires Sol" — escalated for the
word "edit" in a clause that forbids editing, with "authorization" supplying
the consequential half. It now keeps its Luna label, while the same goal
ending in a real instruction still escalates.

**1.14.3** — The recorded goal in `test_evidence_goal_verbs.py` now says "dump"
where it used to name the shell command that prints a file. Hermes's
install-time scanner reads that verb beside a secrets filename as
`read_secrets_file` (critical) and never sees the "Do NOT" in front of it, and
one critical finding makes the verdict `dangerous`, which `--force` does not
override — so `hermes plugins install` refused this plugin outright. The verdict
is now `caution`. The prohibition is unchanged, and the routing decision for
that goal is identical in every field.

Note for future entries: describing that rule is enough to trip it. Name the
verb and the filename in the same line and the scanner matches the sentence
itself, which is how this very entry blocked the install once already.

**1.14.2** — The Anthropic usage reader no longer gives up on a 401. It now
falls back to the refreshing Claude Code resolver, because
`resolve_anthropic_token` reads the credential pool with `refresh=False` (so
diagnostic callers never mutate auth.json), and a pool row that is not the
Claude Code one can shadow valid credentials with an expired token. Measured
on the operator's host 2026-09-21: the resolver returned a 401 token while the
Claude Code credentials were valid for another six hours, which left the Claude
usage guard and load balancing silently inert.

**1.14.1** — `test_root_parent_is_pinned_when_classifier_wants_sol_worker` no
longer asserts that a pinned root parent skips the forced delegation preflight.
That expectation encoded the `pin_root_parent` gate removed from
`_orchestration_eligible` in c7d59eb (2026-09-07) because it stopped a stable
parent from delegating; the test had been failing ever since, unseen while
unittest silently skipped this file's pytest-style functions. The whole suite
(680 tests) is green again.

**1.14.0** — `router_config.yaml` ships the original Codex workflow again
(`workflow: codex`, `default_model: qwen`, no preference chains, Claude
delegation off), and an operator's settings live in a git-ignored
`router_config.local.yaml` merged over it. The dashboard reads the merged
config and saves only the differences into the local file.

**1.13.0** — Under Claude delegation, `usage_guard.balance` evens out the two
accounts' 5-hour windows: when a kind's first account is at 20% or more and the
next account in the chain is 10 points freer, that account goes first. It is
switched and tuned from the dashboard's Workflow section. The soft/hard limits
are unchanged and apply first.

**1.12.0** — `workflow: codex | claude_delegation` switches between the original
Codex workflow and Claude delegation from one setting, live and without a
restart, with the switch at the top of the dashboard's Settings tab. Under Claude
delegation, a turn whose first choice is a Claude tier gets a forced planning call
that offers `delegate_claude` next to `delegate_task` (a review turn had been
forced onto Terra); the Codex workflow keeps the `delegate_task`-only call.
Dashboard saves keep `router_config.yaml`'s comments; a default-tier save follows
Hermes's parent only when its model *and* provider are a router tier's, a
Hermes config that changed mid-save is no longer overwritten, and a Hermes write
that fails now fails the save instead of reporting success.

**1.11.0** — The dashboard imports the router when launched as documented
(the plugin directory is hyphenated, `model-router`, not `model_router`), a
switched-off tier keeps its Settings switch, Claude/Codex delegation chips
share one hover format and are assigned by turn id first, preference-chain
chips show their account and its state, and `claude_delegation:` /
`usage_guard:` are documented

**1.10.13** — Every leaf goal must open with its own tier label, not only a Spark or Sol one: two read-only source-discovery leaves went out unlabelled and both opened on Sol, vetoed by the design gate on the bare word "ui" in "UI components". The contract also stops naming Spark when `callable.spark` is off, which had left read-only discovery with no labelled tier to use at all

**1.10.12** — Three write verbs a read-only evidence goal cannot avoid using — a base commit linked to its hash by "is", a promise to write `[REDACTED]` instead of a secret, and a question about what a module implements — no longer contradict a `[luna]`/`[spark]` label: a receipt-integration evidence report was escalated to Sol for being precise about all three

**1.10.11** — The conductor's tier is pinned as a `model` on the planner call, not only as a goal prefix: a `[qwen]` conductor was being created on the delegation default and ran 24 calls on the account it was meant to spare

**1.10.10** — A write verb that *names* what to inspect ("admin save/update API") no longer contradicts a `[spark]` label: the endpoint was read as an instruction to update it

**1.10.9** — A leaf is classified from its goal; the routing contract the router attaches no longer decides its route

**1.10.8** — A cooling `sonnet5` shows its cooldown in the dashboard: the status panel walked a hardcoded tier list that omitted it, so that account read as merely idle

**1.10.7** — A `delegate_task` that cannot resolve its provider comes back explaining that the host resolves the default route before it reads the tasks, so an exhausted default blocks even a task naming a healthy account — and that retrying with another `model:` would loop

**1.10.6** — A goal that names the base commit it builds on, as the contract requires, no longer reads as an instruction to commit: a read-only `[spark]` leaf was being escalated off Spark for complying

**1.10.5** — Per-task `context` is required on the delegate_task schema, with a description that says what belongs in it: the goal description alone was live, read, and ignored

**1.10.4** — A parent that dispatches twice inside the matching window no longer shows one worker twice under the wrong model while the other runs unlisted: a child session is matched by its goal and claimed once

**1.10.3** — The goal requirements ride on the `delegate_task` schema too, so a parent that never got a preflight — a root prompt under `orchestration.min_chars` creates no conductor — still writes goals that carry a worktree, a branch and a boundary

**1.10.2** — The contract requires a goal to carry what the worker cannot see — worktree, branch, base commit, scope, verification — after an Opus leaf spent all sixteen iterations rediscovering a repository its goal never described, and made no edit

**1.10.1** — A goal naming `opus5` or `sonnet5` in its text is stopped at its first call and returned for re-dispatch, instead of running to completion on Sol; the contract names that mistake rather than restating the rule

**1.10.0** — A worker stopped by an account limit comes back as a re-dispatch with the next target named, from both the batch envelope and the early single-child notice

**1.9.5** — Implementation work stops landing on Luna and on a hardcoded default: `sonnet5` leaves the light substitution group, `[luna]` faces the read-only check `[spark]` already had, `explicit_<tier>` and `[<tier>:xhigh]` work for every tier, and the operator's chain is stated in full as an instruction rather than as advice

**1.9.4** — The forced conductor follows the `code` preference chain, so planning does not have to sit on the primary quota

**1.9.3** — A long cooldown reason wraps inside its card instead of displacing the switch; the settings labels say what only they control

**1.9.2** — A Claude parent's `mcp__`-prefixed delegate_task is recognised, so the preflight it was granted in 1.9.1 actually fires

**1.9.1** — A parent moved onto a fallback account keeps its delegation contract, the forced conductor follows the callable chain, and the contract no longer claims Claude is unavailable

**1.9.0** — Hermes's orchestrator and delegated-worker fallback chains are editable from Settings, with a restore point before every write

**1.8.2** — Quota cooldowns last as long as the provider says and cover every tier on that account, instead of 15 minutes on the one tier that happened to ask

**1.8.1** — A model the account cannot use (400/404 refusal) now takes the configured fallback chain and a long cooldown instead of aborting the leaf

**1.8.0** — Preferred models per kind of work as an ordered chain, configurable from Settings; the reference now also documents effort levels, vision routing, quota fallback, preempted tiers, shadow benchmarking and the agent tree

**1.7.0** — Claude tiers on: the 400 was a Hermes version authenticating with its own OAuth app, not an account or plan limit

**1.6.2** — Claude delegation targets ship off pending an account-side 400; what was ruled out, and how to check it, is documented rather than guessed at (superseded by 1.7.0)

**1.6.0** — Substitution groups across accounts, cooling targets annotated with their replacement, orchestrator selector restricted to routable tiers

**1.5.0** — Claude reached natively as a delegation target on subscription OAuth, Claude tiers counted and switchable like any other

**1.4.0** — Cooldowns after quota and repeated failures, per-account load in the routing contract, policy routes that decline the fallback chain, cooling tiers and load shown in the dashboard

**1.3.0** — Per-worker route selection across accounts, delegated Claude review leaves, plan labels authoritative on delegated workers, orchestration preflight with a late rescue and logged skip reasons

**1.2.0** — Stable parent policy, bounded delegation, privacy-safe logging

## License

MIT

## Author

SENTINEL — Hermes Agent Model Router
