# thinkmoney AI Customer Service — Subscription Manager

A multi-agent customer service system built with [LangGraph](https://github.com/langchain-ai/langgraph).

The provided system was a triage agent with a knowledge base and 22 mock tools, and no
sub-agents. This submission adds a **subscription manager**: it detects recurring payments
across three payment rails, tells the customer what they actually cost, finds nine distinct
ways to spend less, and then cancels or blocks them — pausing for explicit confirmation
before any money-affecting call.

Against the mock customer (`USR-2847`) the opening line is arithmetic, not model output:

> You are committed to **£1,507.39/yr** across 8 discretionary subscriptions (£118.95/mo).
> **£691.63/yr** I can identify from your account alone, and a further **£528.00/yr**
> depends on a question only you can answer.

Those two figures are deliberately never added together. See
[What the bank can and cannot know](#what-the-bank-can-and-cannot-know).

Rent (£1,200.00/mo) is detected, reported separately as essential, and never offered for
cancellation.

## Quick Start

### Prerequisites

- [UV](https://docs.astral.sh/uv/) (Python package manager)
- One of the following LLM providers:
  - **Ollama** (local, free) — [install guide](https://ollama.ai)
  - **OpenAI** API key
  - **Anthropic** API key

### Setup

```bash
# Install dependencies
uv sync

# If using Ollama, pull the model first:
ollama pull gpt-oss:20b

# Run with your chosen provider:
uv run thinkmoney --provider ollama
uv run thinkmoney --provider openai --model gpt-4o-mini
uv run thinkmoney --provider anthropic
```

For OpenAI/Anthropic, set the API key as an environment variable:
```bash
export OPENAI_API_KEY="sk-..."
# or
export ANTHROPIC_API_KEY="sk-ant-..."
```

The whole demo below runs on **Ollama with no API key**.

### Provider Defaults

| Provider | Default Model | Notes |
|---|---|---|
| `ollama` | `gpt-oss:20b` | Must be running locally |
| `openai` | `gpt-4o-mini` | Requires `OPENAI_API_KEY` |
| `anthropic` | `claude-opus-5` | Requires `ANTHROPIC_API_KEY`; sends `effort`, no sampling params |

The Anthropic branch passes `effort="medium"` and deliberately sends **no**
`temperature` / `top_p` / `top_k` — Opus 5 rejects sampling parameters outright, and
`temperature=0` (the previous default) would 400 every request. This is asserted against
the serialised request payload in `tests/test_anthropic_provider.py:1`, not against the
client attribute, because LangChain serialises a field regardless of how it was set.

`--provider` also selects the cancellation-guidance search backend: Anthropic and OpenAI
each use their own hosted web search, and Ollama has none, so on it that one lookup answers
from the local directory and says so. Everything else works identically on all three.

## Try it

Copy-pasteable prompts. The figures are what the fixture actually produces — they are
arithmetic over the transaction corpus, not model output, and the headline ones are pinned
by `tests/test_subscriptions.py:200`.

**Getting to the subscriptions agent is model-judged, unlike everything after it.** The
deterministic handoff described [below](#why-delegation-is-deterministic-not-model-judged)
governs subscriptions → cancellation research. The *front door* does not: triage decides
whether to call `route_to_agent` (`src/agents/triage.py:31`), and there is no keyword
fallback — `_get_route_target` (`src/graph.py:61`) only reads the tool call the model chose
to make. A prompt that never says "subscription" relies on the triage model connecting it to
the agent description in `src/graph.py:43`. *"Where can I save money?"* is the weakest of
these: it also matches triage's own "general question → search the knowledge base" guideline
(`src/agents/triage.py:69`), so a smaller model may answer it from the knowledge base and
never route at all. If a prompt lands on triage instead, say "my subscriptions" explicitly
and it goes through. Nothing below is asserted by a test — the routing tests use stubbed
LLMs, so these are demo prompts, not a contract.

```
# Detection and the headline
What subscriptions am I paying for?      -> 9 recurring payments, £118.95/mo, £1,507.39/yr
What am I paying for each month?         -> the same report, asked without the routing keyword
Where can I save money?                  -> £691.63/yr identified + £528.00/yr conditional

# One prompt per saving strategy
Am I paying for anything twice?          -> Netflix DD-4471 vs SO-102 "Shared Netflix"  £300.00
Is my gym membership worth keeping?      -> asks: it cannot see whether you go           £528.00*
Can I keep Spotify but pay less?         -> downgrade Premium -> Free                   £155.88
Did I forget to cancel a free trial?     -> Adobe £0.00 -> £9.99                        £119.88
Any subscriptions about to renew?        -> Namecheap domain renewal, ~11 days out       £79.99
Do any of my subscriptions overlap?      -> Apple iCloud+ vs Google One, both cloud       £35.88
Could I pay less for Netflix?            -> Standard -> with adverts £132.00  (superseded)
Would paying annually be cheaper?        -> Adobe monthly -> annual   £24.00  (superseded)
Has anything gone up in price?           -> Spotify £11.99 -> £12.99  £12.00  (superseded)

# All three cancellation rails
Cancel the duplicate Netflix payment     -> standing order: confirmation gate, then cancel
Cancel the domain renewal                -> direct debit DD-4472: confirmation gate, then cancel
Cancel my gym membership                 -> card-on-file: handoff to cancellation research, guide + block + caveat

# The judgment case
Cancel everything                        -> refuses to sweep up the rent
```

The nine strategy prompts do not each trigger a different analysis. `find_recurring_payments`
returns the whole report on every call, so all nine findings are already in the payload; the
prompt only steers which part of it the agent narrates. The figure on each line is what the
tool put in front of the model, not a guarantee about the wording of the reply.

`*` The £528.00 is **conditional**, not identified — it counts only if you tell the agent
you have stopped using the gym. It is reported separately for that reason.

Three of those savings are **detected but not counted**. The Netflix downgrade (£132.00) is
superseded by the duplicate finding (£300.00), the Adobe annual switch (£24.00) by the larger
converted-trial finding (£119.88), and the Spotify price rise (£12.00) by the downgrade
(£155.88) — one service can only bank one saving, so the report carries a `superseded` list
alongside the counted `strategies`. Ask about them by name and the agent still answers; they
just do not double-count into £1,219.63.

Also worth trying, because they exercise the honest-degradation paths:

```
How do I cancel Disney+?                 -> answers about a service the customer doesn't hold
How do I cancel my FitLife Gym?          -> flagged as illustrative, not verified guidance
```

## Design

Diagrams of the graph topology, tool ownership, the deterministic handoff and the
confirmation gate are in **[ARCHITECTURE.md](ARCHITECTURE.md)**. The sections below argue
*why*; that file shows *what is wired to what*.

### Subscription Manager

Helping clients with their subscriptions is feature that would actually add value, and makes use of a lot of the existing structures along the way to providing the service.

Detection reads **transactions**; the rails it finds live in **payments** (direct
debits, standing orders) and **cards** (card-on-file); the remedy differs per rail; and the
one rail the bank *cannot* cancel forces a handoff to a second agent.
The idea is that the service is able to suggest stragergies on what can be done rather that just listing subscriptions.

Existing tools used:
`get_transaction_history`, `cancel_standing_order`, `list_cards`, `get_payees` and
`search_knowledge_base`.


### Why two agents, and where the boundary falls

The boundary is **what the bank knows** vs **what the merchant knows**
([drawn out here](ARCHITECTURE.md#2-tool-ownership)):

| Agent | Owns | Tools |
|---|---|---|
| `subscriptions` (`src/agents/subscriptions/agent.py:1`) | Internal banking data and every money-moving action | 8 |
| `cancellation_research` (`src/agents/cancellation_research/agent.py:1`) | External merchant cancellation knowledge | 1 |

The cancellation research agent holds **no** banking data and **no** money-moving tool, by construction.
That is the point of the split: the agent that talks about the outside world cannot move the
customer's money, and the agent that moves money does not improvise merchant policy. It also
binds exactly one tool (`find_cancellation_guide`) with no independent web search, so the
directory-first lookup order is enforced *inside the tool* and cannot be bypassed by model
choice.

### Why delegation is deterministic, not model-judged

The obvious design is to let the subscriptions agent decide when it needs cancellation research. It is
also the design that fails intermittently, and intermittent routing is the worst kind of bug
to demo.

Instead the handoff is triggered by **payment rail**, which is a fact in the data, not a
judgement:

- `direct_debit` → the bank can cancel the mandate → `cancel_direct_debit`
- `standing_order` → the bank can cancel the order → `cancel_standing_order`
- `card_on_file` → **the bank cannot cancel this** → the merchant must → hand off to cancellation research,
  but only for a merchant the customer actually raised

The wrapped `subscriptions_tools` node reads `card_on_file` labels straight out of the
`find_recurring_payments` result, then keeps only the merchants **this turn is actually
about** — the ones the customer named in their own words, or that a `block_merchant_on_card`
call names — and pushes those onto `unresolved_card_subs` (`src/models.py:1`). The router
sends the turn to cancellation research while that queue is non-empty and back to
subscriptions when it drains. Both halves are parsing, so the model never votes.

That second filter was missing at first, and its absence is worth recording. The rail label
alone was the whole trigger, which meant the handoff fired on the shape of the report rather
than on the request: holding *any* card-billed subscription sent every one of them to
cancellation research. Asked "what subscriptions am I paying for?", the agent answered with
the list and then, unprompted, gave the cancellation steps for a gym — five directory
lookups to produce advice nobody had asked for. The fix was not to make delegation
model-judged after all; it was to notice the trigger was reading the *data* when it should
have been reading the *request*. `card_subs_needing_research`
(`src/agents/subscriptions/agent.py:128`) does the narrowing, and
`tests/test_delegation.py::TestReadOnlyTurn` pins that a question naming no merchant never
reaches the research agent.

The scoping is a heuristic where the rail label was a fact, and it is worth being honest
about the seam: naming a merchant is not the same as wanting to cancel it, so *"can I keep
Spotify but pay less?"* still queues Spotify. That costs one directory lookup on a downgrade
question. Tightening it further means keyword-matching intent, which is the brittle thing
this design set out to avoid.

`unresolved_card_subs` is a **work queue, not an inventory** — the router re-reads it on
every exit from the subscriptions agent, so the cancellation research node must return a drained list or
the turn cycles to LangGraph's recursion limit. Both drain rules are pinned by
`tests/test_delegation.py:1`, including a test that monkeypatches the drain away and proves
the loop is real rather than theoretical.

### Why the write tools take a merchant, not a reference

Detection already knows which rail every subscription bills on. That fact used to be computed
here, printed into a JSON report, and then **re-derived by the model** when it picked a tool
and an identifier — and the second derivation is the one that failed. Asked to cancel Google
One, which is card-billed and has no mandate at all, the agent called `cancel_direct_debit`
on `DD-4472`: Namecheap's domain renewal. It then reported Google One as cancelled.

So the model is no longer asked. `src/tools/rails.py:1` owns merchant → (rail, reference) as
a single function over the same fixtures, and the write tools take the merchant and look the
reference up themselves. A reference can still be passed, but only as a claim to check —
supply one belonging to another merchant and the write refuses rather than proceeding. The
wrong rail refuses too, naming the tool that would work:

> Google One is billed to a card (CARD-8834), not a direct debit. There is no mandate to
> cancel — the charge can only be blocked. Use block_merchant_on_card instead.

The model says *what* to stop; the tool decides *how*. That is the same principle as the
confirmation gate below — take the decision out of the prompt rather than instruct the model
harder. The prompt already said "match the remedy to the rail", and that is precisely what
got ignored.

One decision is left with the model: **which merchant**. A wrong name still produces a
correct cancellation of the wrong thing, and no lookup can fix that — which subscription
someone means by "cancel it" is not in the data, it is in their head. So a write only runs
when it is **bound to a subscription the customer identified**:

```
"cancel Google One"        + Google One  -> proceeds to the gate
"cancel Google One"        + Spotify     -> refused, contradiction
"cancel it"                + Spotify     -> refused, ask which one
"cancel my gym membership" + FitLife Gym -> refused, ask which one
"cancel Netflix"           + either      -> proceeds, both are named
```

An unbound write comes back with a **numbered** list of subscriptions that tool could act on,
and the agent puts it to the customer. The binding then comes from a list the code generated
and a person who knows the answer, instead of from inference.

Numbered, because the answer has to be bindable. The first version asked for a name, and a
customer replying "2" named nothing — so the write was refused again and the same question
came back forever. The offered list is now held in state and the reply resolved against it,
so cancelling Netflix takes two turns instead of never. Only a bare number counts: "£2.99" is
not a selection.

The list is scoped to the conversation, not just to the rail. Asked about Netflix, the
customer was once shown every direct debit on the account — including their domain renewal.
It is now narrowed using the merchant the model proposed: a hint about what is being
discussed, safe to use because the customer still chooses. Both Netflix entries appear,
labelled by rail, because telling them apart is the whole question; an unmatched name falls
back to the full list, and a name not on the list still works.

Essentials never appear in it. It once offered the customer their rent alongside a shared
Netflix subscription. Housing, Utilities and Insurance are excluded — the same categories
detection keeps out of the savings figures. They stay cancellable by name; they are just
never proposed.

Replies are read the way people write them. "cancel 1" selects, because that is how the
customer answered and requiring a bare digit sent them round the same question again.
Anything with more in it than a choice — "cancel 2 of these", "1 and 2" — still selects
nothing.

The first version of this allowed an unbound write, so it failed *open* — "cancel it" let the
model pick freely. Asking costs an extra exchange when the customer was vague; guessing costs
them the wrong subscription. Name matching is still fuzzy, but it can now only *unlock* a
write, never license a guess: no match means ask.

### The confirmation gate

Write tools are gated **structurally**, not by prompt. Each one is wrapped in
`@requires_confirmation` (`src/confirmation.py:1`), which calls LangGraph's `interrupt()`
before the tool's body runs — before, so a resume cannot re-run a side effect. Anything the
customer answers that is not an explicit approval is treated as a refusal, so a garbled
answer, an empty answer or walking away all leave the money where it is.

The gate rides on the tool rather than the tool node because that is where LangGraph's own
docs put it, and because it makes a refusal self-answering: the tool returns an ordinary
`success: false` result, so no `tool_call_id` is ever left unanswered — that would be a
provider 400 — and the graph needs no knowledge of which tools move money. A mixed batch
(one read call, one write call) runs the read and declines the write, with no message
surgery anywhere.

Carrying the decorator is the registration. `CONFIRMED_TOOLS` collects every gated name and
`tests/test_confirmation_gate.py:1` asserts it covers exactly the money-moving tools, so a
write tool added without a gate fails a test instead of shipping a hole.

Without a checkpointer the graph still surfaces `__interrupt__` and stops rather than
raising, so an un-checkpointed caller fails safe: no write happens.

### What the bank can and cannot know

The systems cant know if the subscription is still being used; so it prompts the question.

> *"FitLife Gym has taken £44.00 a month since 2026-06-04 — £528.00 a year, your largest
> discretionary subscription. Are you still using it? I have no way of knowing that from
> your account. If you are not, thinkmoney cannot cancel this one — the mandate is held by
> FitLife Gym, not by us. I can block future charges on CARD-5521 and give you FitLife
> Gym's own cancellation steps. Blocking stops the payment, not the contract, so you may
> still owe them."*

This forces the savings total to split, because the two numbers are two different kinds of
claim:

| | Provable from the account | Example |
|---|---|---|
| `identified_saving` **£691.63** | yes | one service billed twice; a trial that converted; a renewal dated in the future |
| `potential_saving` **£528.00** | **no** — needs the customer's answer | a subscription whose usage the bank cannot observe |

The agent is instructed never to add them. A combined £1,219.63 headline would restate a
question as a finding, and the conditional part is 43% of it.


### The audit trail, and the bug that made it necessary

Every money-affecting action is recorded to an append-only JSONL trail
(`src/audit.py:1`), written at three points: the request, the customer's answer, and the
execution with the tool's own success flag.

```
write_requested   block_merchant_on_card  {card_id: CARD-5521, merchant: Apple iCloud+}
write_requested   block_merchant_on_card  (interrupt() replays the node — see below)
write_approved    block_merchant_on_card  answer: "yes"
write_executed    block_merchant_on_card  succeeded: true
```

This exists because of a real failure. The prompt used to say
*"before calling any of them... wait for the customer to say yes"*, which guaranteed the
model **never called the write tool**. It asked in prose and ended its turn — so the
`interrupt()` gate, never fired,
because its trigger was never reached. The turn returned to triage, the customer's "yes"
landed on an agent with no pending action and no write tools, and the reply was:

> *"I'm blocking Apple iCloud+ on your card right now. You'll no longer be charged £2.99
> per month."*

Nothing had been blocked. Nothing was recorded. The only evidence was a terminal text.

Two fixes. The prompt now tells the agent to **call the tool and let the gate do the
asking** (`src/agents/subscriptions/agent.py:1`), keeping confirmation inside one turn instead of
spanning two. And the trail makes the failure mode visible after the fact: a
`write_requested` with no matching `write_executed` is a write that was asked for and
silently did not happen.

`tests/test_audit_trail.py:1` drives the whole path — triage routes, the agent emits the
write, the gate halts, the customer answers — and asserts that approval produces an actual
execution. It fails loudly if prose confirmation ever returns. It also pins that the node
replaying on resume does not execute the write twice: `interrupt()` re-runs the node from
the top, so the request appears twice in the log while the money moves once. Dedupe on
`tool_call_id` when reading the trail.

The log path defaults to `audit.log` and is overridable with `THINKMONEY_AUDIT_LOG`. It is
gitignored, and a file that cannot be written is swallowed rather than raised — a full disk
must not be the reason a cancellation fails.

### Not a cancellation machine

Three of the eight strategies leave the customer **still subscribed**: **downgrade** to a
cheaper tier, switch to **annual** billing, and drop one of two **overlapping** services
while keeping the other. A bank that recommends cancelling something a customer values has
not helped them, and an agent whose only verb is "cancel" is a blunt instrument. The tool
also refuses to sweep up essentials — rent is detected, marked essential, excluded from
every headline total, and never proposed for cancellation.

Honesty is enforced in the tool output rather than left to the prompt:
`block_merchant_on_card` always carries the caveat that blocking stops the *payment*, not
the *contract* — the customer may still owe the merchant and can still be chased for it —
and the cancellation directory labels invented entries as `illustrative` with
`verified_on: None`, so a plausible-sounding guide is never presented as researched fact.
Real entries carry a `verified_on` date and a vendor source URL, which the cancellation research agent is
required to cite.

### The `ALL_TOOLS` count changed from 22 to 24

`tests/test_tools.py:1` pins the total tool count, and adding write tools moved it. That
assertion was updated deliberately, in the commit that adds them — not quietly, and not
because it was in the way.

Only two of the new tools are in that total: `cancel_direct_debit` and
`block_merchant_on_card`, which join existing capability groups because any agent working on
payments or cards could reasonably need them. `find_recurring_payments` and
`find_cancellation_guide` are **not** in `ALL_TOOLS` — each belongs to one agent and lives in
that agent's package rather than on the bank's shared tool surface, which
`test_agent_owned_tools_are_not_in_the_shared_groups` asserts directly.

A further test asserts `len(ALL_TOOLS)` equals the sum of every group's length, so the next
person to add a group gets a loud failure instead of a silently drifting constant.



### Consciously deferred

- **State is in-memory.** `MemorySaver` holds the thread, so conversation history dies with
  the process. A real deployment needs a durable checkpointer.
- **Write tools are stateless.** Cancelling a direct debit validates and reports; it does not
  mutate the fixture. Detection output is asserted byte-identical across calls, and a mutating
  write would break unrelated tests. Real persistence is a database, not a mutable module global.
- **Live web search has no Ollama backend.** `find_cancellation_guide` falls back to the
  configured provider's own hosted search (`src/agents/cancellation_research/web_search.py:1`) — Claude's server-side
  `web_search` tool on `--provider anthropic`, the Responses API's `web_search` tool on
  `--provider openai`. A local model has neither, so on Ollama the tool answers from the
  directory alone. That is not a crash: `_search_available()` is false, the degraded response
  says live search was unavailable rather than claiming it looked and found nothing, and the
  directory stays the demo's source of truth on every provider.
- **No live Anthropic call was made.** There is no API key in this environment, so the
  "Opus 5 will not 400" claim rests on payload assertions. Anyone with a key should run one
  real turn before relying on it.
- **The local model cannot drive the write path.** `gpt-oss:20b` will not emit a write tool
  call however directly it is asked — it answers conversationally and claims success. The
  approve/decline transcripts were produced by driving the real CLI loop with a scripted LLM;
  Ollama verifies the read path only. Use `--provider anthropic` or `--provider openai` to
  see the confirmation gate live.

## Project Structure

```
├── src/
│   ├── main.py               # CLI entry point (checkpointer, interrupt prompt, resume)
│   ├── config.py             # LLM provider factory
│   ├── models.py             # State definitions (unresolved_card_subs,
│   │                         #   offered_subscriptions), mock user data
│   ├── graph.py              # Agent graph, deterministic routers, work queue
│   ├── confirmation.py       # @requires_confirmation write gate + CONFIRMED_TOOLS
│   ├── audit.py              # Append-only JSONL trail for money-affecting actions
│   ├── agents/
│   │   ├── triage.py         # Triage agent (provided, reference impl)
│   │   ├── subscriptions/    # Subscriptions agent — internal data, money-moving actions
│   │   │   ├── agent.py      #   Prompt, tool binding, card-on-file rail parsing
│   │   │   ├── detection.py  #   find_recurring_payments: detection + savings arithmetic
│   │   │   └── data.py       #   Engagement, plan tiers, billing options
│   │   └── cancellation_research/  # Cancellation research agent — external merchant
│   │       ├── agent.py      #   Prompt, one read-only tool, queue drain
│   │       ├── guide.py      #   find_cancellation_guide: directory first, search second
│   │       ├── directory.py  #   Merchant cancellation guidance (7 researched, 2 illustrative)
│   │       └── web_search.py #   Provider-aware hosted search, fallback only
│   ├── knowledge_base/
│   │   ├── loader.py         # ChromaDB semantic search
│   │   └── data/             # Markdown knowledge base files
│   └── tools/
│       ├── __init__.py       # Tool groups (ACCOUNT_TOOLS, ..., SUBSCRIPTION_TOOLS)
│       ├── account.py        # Account management tools
│       ├── cards.py          # Card management tools (+ MOCK_CARDS, block_merchant_on_card)
│       ├── transactions.py   # Transaction query tools (+ the 60-entry corpus)
│       ├── rails.py          # merchant → rail + reference, the single source of truth
│       │                     #   every write resolves through; also the choice offered
│       │                     #   when a write is not bound to one
│       ├── payments.py       # Payment & transfer tools (+ mandate/order fixtures,
│       │                     #   cancel_direct_debit, cancel_standing_order)
│       └── kyc.py            # KYC/compliance tools
└── tests/
```

**[ARCHITECTURE.md](ARCHITECTURE.md)** — Mermaid diagrams of the above: graph topology, tool
ownership per agent, the rail-driven handoff sequence, and the confirmation gate.

Every fixture date is generated relative to today, so the demo does not rot.

## Running Tests

```bash
uv run pytest -v
```

## The Original Brief

See **[TASK.md](TASK.md)**.
