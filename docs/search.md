# Search and connections

[← Documentation index](../README.md#documentation)

Ranked, typo-tolerant search with a query language that mixes text with tags, folders, properties and links; related notes; links to add and notes to write; unlinked mentions. All classical, offline, with a reason for every result. Examples use a copy of the sample vault: `cp -Rp tests/fixtures/vault /tmp/sample-vault && vault index /tmp/sample-vault`.

## Search and queries

`vault search <vault> '<query>'` answers questions that mix text with structure:

```bash
vault search /tmp/sample-vault '"reward hacking" links-to:[[grpo-group-relative-policy-optimization]] -path:10_SOURCES'
```
```
Search '"reward hacking" links-to:[[grpo-group-relative-policy-optimization]] -path:10_SOURCES': /tmp/sample-vault
  index 2026-11-07T00:24:20+00:00 (up to date)

  note: [[grpo-group-relative-policy-optimization]] is not a file in the vault; using the links written to it (10 notes)
  "reward hacking"  46 notes (phrase, required)
  2 of 180 notes match · by score

   1. instance-specific-rubric-as-reward-function                                               2.10
      20_ZETTLEKASTEN/instance-specific-rubric-as-reward-function.md
      "reward hacking": body ×2
      … evaluation]]. The untested risk is [[«reward-hacking»-as-specification-gaming]] — 300 training …

   2. the-critic-deletion-tension-in-agentic-rl                                                 1.55
      20_ZETTLEKASTEN/the-critic-deletion-tension-in-agentic-rl.md
      "reward hacking": body
      … is a direct defense against [[«reward-hacking»-as-specification-gaming]] — the model producing …

  Tags (notes)
      2  #baby

  Folders (notes)
      2  20_ZETTLEKASTEN/
```

Text ranks (BM25F over each note's title, aliases, headings, tags and body; a match in the title counts 3×,
in a heading, alias or tag 2×, in the body 1×). Filters narrow the results and never change a score. Every
result list ends with the tags and folders of *all* matching notes and how many notes have each, so you can
add `tag:…` or `path:…` to drill down. The index behind it (postings, properties, links) lives in the same
SQLite file and is kept up to date incrementally by `vault index`.

**Syntax**

| Write | Finds notes that |
|---|---|
| `reward hacking` | contain any of the words (at least one must match; more and better matches rank higher). Each word forgives typos: 1 edit for 4–7 characters, 2 for longer words, none for shorter ones or words with digits; a swap of two letters is one edit |
| `"reward hacking"` | contain the exact phrase, in one field (never across two headings) |
| `"reward"` | contain exactly this word, no spelling variants |
| `verifier NEAR/3 reward` | have the two words at most 3 positions apart (2 words between), either order, same field and section; n is 1–99 |
| `tag:ai-safety` | have the tag or a tag nested below it (`tag:rl` also finds `#rl/ppo`); `#` optional, case-insensitive |
| `path:50_MOC/discourse` | have this text in their path (folder and file name), case-insensitive; quote paths with spaces: `path:"Daily notes"` |
| `links-to:[[Note]]` | link to the note (links and embeds, self-links excluded). A note that does not exist matches the links written to it, with a warning |
| `linked-from:[[Note]]` | are linked from the note |
| `within:2 of:[[Note]]` | are at most 2 links away from the note, following links in either direction between existing notes (the note itself is not included); N is 1–10 |
| `[status]` | have the property, with any value (also `has:status`) |
| `[status:baby]` | have a value containing the text (case-insensitive) |
| `[status=baby]` | have a value equal to it; `#tags` and `[[links]]` inside a value count as values: `Status: #baby #processed` equals `baby` and `processed` |
| `[created>=2026-09]`, `[rating<3]` | compare dates with dates and numbers with numbers: `>` `>=` `<` `<=` |
| `[created:2026-01..2026-03]` | have a date in the range (inclusive) |
| `modified:>=2026-09-01` | were last modified (file time) in, before or after a period; `modified:2026-09`, `modified:2026-01..2026-03` |
| `has:X`, `missing:X` | X is `tags`, `aliases`, `headings`, `links` (to another note), `backlinks`, `unresolved` (links), `frontmatter` or `properties`; anything else is a property name: `missing:status` |
| `-X`, `NOT X` | do not match X. A negated word matches exactly: `-root` does not drop notes about robots |
| `X Y`, `X AND Y` | match both. Bare words written side by side are the exception: at least one of them must match. `AND` between words makes both required |
| `X OR Y` | match either |
| `( … )` | group: `(tag:rl OR tag:ml) -path:Archive` |

`NOT` binds tightest, then `AND` (written or implicit), then `OR`: `a OR b c` means `a OR (b c)`.
`OR`, `AND`, `NOT` and `NEAR/n` are operators only in capitals. A note reference can be `[[Note]]`,
`"Note name"` or `Note`, by file name, path or alias, like a link; `#heading` and `|alias` are ignored.
Dates in a query are a year, month or day (`2026`, `2026-09`, `2026-09-24`) and stand for the whole period,
so `>2026-09` means after September. Results are ordered by how many of the words, phrases and NEAR pairs
match exactly, then by score; notes that match the filters but none of the words come last; a query
without words lists notes by path.

**Properties** come from three places, all matched the same way (keys are case-insensitive):

- frontmatter (Obsidian properties);
- inline fields, `key:: value` on their own line or as a list item (Dataview style);
- header fields: the `Key: value` lines at the very top of a note, as this project's note format writes
  `Created:`, `Tags:` and `Status:`. Only the leading run of such lines counts; the first other line ends it.

Dates in values are ISO (`2026-09-24`, `2026-09-24T19:20`) or this project's day-first `24-09-2026 19:20`.
`vault stats`, `vault health` and the note view still report frontmatter properties only.

**Three questions on the sample vault**

| Question | Query | Answer |
|---|---|---|
| Notes about reward hacking that link to the GRPO note, outside the sources folder | `"reward hacking" links-to:[[grpo-group-relative-policy-optimization]] -path:10_SOURCES` | 2 notes (above). The GRPO note does not exist yet; 10 notes link to it, 6 of them sources. Without quotes, `reward hacking` also admits the 2 notes that only mention reward |
| Anything within two links of a note, tagged ai-safety | `within:2 of:[[obfuscated-reward-hacking]] tag:ai-safety` | 12 of the vault's 15 ai-safety notes |
| Notes missing a property | `missing:status`, `path:20_ZETTLEKASTEN missing:created` | 3 MOC notes without `Status:`; 15 zettels without `Created:`, 12 of them tagged ai-safety (a batch written from another template) |

**Errors** name the problem, point at it and suggest a fix; the exit code is 2:

```
error: this ( is never closed
  reward (tag:ai-safety OR path:50_MOC
         ^
  add ) after the group
```

Options: `--limit N` (0 = all), `--facets N` (rows of tag/folder counts, 0 = all), `--tree` (print the parsed
query as a syntax tree and stop: `(and (links-to "…") (not (path "10_SOURCES")) (any reward hacking))`),
`--explain` (the numbers behind each score), `--json`, `--no-fuzzy`, `--color auto|always|never`
(`NO_COLOR` is respected). In a terminal, matches are highlighted in colour instead of `«»`.
The grammar and every choice behind it are documented at the top of `vaultengine/querylang.py`,
`vaultengine/search.py`, `vaultengine/properties.py` and `vaultengine/spelling.py`.

## Related notes, links to add, notes to write

`vault related <vault> <note>` lists the notes most related to a note, including the ones it does not link to yet,
and says why for each:

```bash
vault related /tmp/sample-vault kl-penalty-in-rlhf --limit 5
```
```
Related to kl-penalty-in-rlhf
  20_ZETTLEKASTEN/kl-penalty-in-rlhf.md · index 2026-11-07T06:01:37+00:00

  score = 0.5 × graph + 0.3 × text + 0.2 × tags · 179 notes compared

   1. direct-preference-optimization-dpo                                            0.59      linked
      linked from it · similar text (dpo, rlhf, ppo)

   2. RLHF                                                                          0.50  not linked
      2 hops via direct-preference-optimization-dpo +1 · similar text (kl, dpo, rlhf)

   3. trust-region-policy-optimization-monotonic-improvement-guarantee              0.42      linked
      linked from it · similar text (kl, divergence, penalty)

   4. covariance-targeted-entropy-intervention                                      0.33  not linked
      2 hops via trust-region-policy-optimization-monoto… +3 · similar text (kl, penalize, penalty)

   5. Cameron R. Wolfe - GRPO Tricks (Making RL Actually Work)                      0.31  not linked
      2 hops via trust-region-policy-optimization-monoto… +2 · similar text (kl, token, penalty)
```

Three signals, each scaled to 0–1 by its best value, blended 0.5 / 0.3 / 0.2:

- **graph**: personalized PageRank (a random walk that follows links in either direction and jumps back to the
  note with probability 0.15) over the link graph, divided by each note's number of links so hubs do not come
  first for everything. Links to notes that do not exist yet are part of the graph: two notes that both link
  `[[rl-environment-design]]` are two hops apart through it (shown as "(no note yet)");
- **text**: the note's 30 most distinctive terms run as a search, scored with the same BM25F as `vault search`.
  Property keys (`Status:`), tags and the words of template headings ("Related zettels", in 3+ notes) do not count;
- **tags**: shared tags weighted by rarity; tags on more than a third of the notes (`#baby`) are ignored.

Reasons are facts you can check: "linked from it" (this note links to the result), "links to it", "linked both
ways", "2 hops via X +2 more", "shares #tag", "similar text (terms)". Results without any link between the two
notes are marked `not linked`. `--unlinked` shows only those (link candidates), `--explain` the three signals,
`--json` everything. The note can be a name, path or alias, or the name of a missing note (then: what the notes
that link to it are related to).

Where it is convincing and where it is not (checked on the sample vault):

- concept notes with a few specific links (`kl-penalty-in-rlhf`): the unlinked suggestions are real
  (`RLHF`, `ppo-as-trpo-simplification`, `GRPO Tricks` all discuss the KL penalty). Dividing by degree costs
  something here: the broad parent concept, `reinforcement-learning-from-human-feedback-rlhf`, is linked but
  only eighth, because it is a hub and its text is general;
- dense, well-linked clusters (`obfuscated-reward-hacking`): the top results are right but already linked; the
  unlinked ones are generic neighbours ("2 hops via … · shares #ai-safety"), not new insight;
- notes made from a shared template (the `50_MOC/discourse` claims): structure dominates. Sibling claims come up
  through a shared missing question note, the status tag `#supported` and naming-convention words (`que`, `evd`),
  not because their content is related. Treat those results as "same part of the map", not "same idea".

**Connecting notes.** `vault suggest <vault>` starts with links to add: pairs of notes with no link between them
that probably belong linked, with every kind of evidence in the reason:

```bash
vault suggest /tmp/sample-vault --links --path 20_ZETTLEKASTEN --limit 3
```
```
  Links to add  (notes with no link between them; score = 0.4 × shared neighbours + 0.3 × cited together + 0.3 × similar text)

    1. in-context-scheming-capability  ↔  sandbagging-strategic-underperformance                          0.24
       shared neighbours: 2412.04984 - Frontier Models Capable of In-c…, ai-control-vs-alignment +1 · cited together by scheming-as-threat-model · similar text (scheming, goal, sonnet)

    2. in-context-scheming-capability  ↔  scheming-inability-safety-case-closed                           0.24
       shared neighbours: 2412.04984 - Frontier Models Capable of In-c…, ai-control-vs-alignment +1 · cited together by scheming-as-threat-model · similar text (scheming, empirical)

    3. cot-monitor-beats-action-monitor  ↔  cot-monitorability-vs-faithfulness                            0.23
       shared neighbours: 2503.11926 - Monitoring Reasoning Models (Co… · cited together by monitorability-tax, obfuscated-reward-hacking · similar text (cot, intent)
```

- **shared neighbours**: notes (or missing notes) both are linked with, weighted by rarity (a note linked from 2
  notes counts much more than a topic linked from 52) and normalised by how many links each note has, so two
  notes that each link 60 things are not matched just for being big;
- **cited together**: notes that link to both, weighted and normalised the same way (a note that links to both
  counts here, not also as a shared neighbour);
- **similar text**: the cosine of the two notes' distinctive-term profiles.

Each is a similarity from 0 to 1 (`--explain` shows them); score = 0.4 × shared + 0.3 × cited + 0.3 × text.
`--for <note>` lists pairs with one note, `--path <text>` pairs whose paths both contain the text, `--links` /
`--notes` one section only. On the sample the zettel pairs are convincing (the three scheming notes, the two CoT
monitor notes, the rubric notes); across the whole vault the top is dominated by evidence notes (`EVD - …`) of
the discourse graph: small notes under the same claim or source, which score high on thin but consistent
evidence. Some belong linked (the two AppWorld notes share a source, a claim and their vocabulary), many are just
siblings, and in a discourse graph evidence links to claims rather than to other evidence.

`vault unlinked <vault> [note]` finds every plain-text mention of a note's title or alias that is not a link yet
(all notes, or one):

```bash
vault unlinked /tmp/sample-vault monitorability-tax
```
```
  4 plain-text mentions of monitorability-tax in 1 note that does not link to it yet

  monitorability-tax  (1 note)
    10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md:22  …ur own detector*. They call the cost of avoiding this the **«monitorability tax»**.
    10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md:81  ## The «monitorability tax»
    10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md:85  This costs something, hence **"«monitorability tax»"**: developers may have to ship "slightly less performant m…
    10_SOURCES/2503.11926 - Monitoring Reasoning Models (CoT Obfuscation).md:113  …ly safe policy is to leave the CoT unoptimized and pay the "«monitorability tax»" of keeping it honest.
```

Titles and aliases match as words, the way search splits them, so the kebab-case `reward-hacking` is mentioned
by "reward hacking", "Reward Hacking" and "reward-hacking" (even across a line break), but not by "reward
hackings" or "reward_hacking"; where names overlap the longest one wins ("reward hacking as specification gaming"
is not also a mention of `reward`). One-word titles match every use of the word (`reward`: 512 mentions in 71
notes on the sample); `--min-words 2` leaves them out.

Neither command ever suggests a link that already exists (a link or an embed, with or without `#heading`, by
title, path or alias, or an ambiguous link that could mean the note; for pairs in either direction), a note to
itself, or a mention inside code, a comment, frontmatter, a tag, a `[[link]]` or `![[embed]]` (display text
included), a markdown link or image, or a bare URL. A note that already links the target is skipped entirely,
even where it mentions it again. The tests plant each of these cases, and in a copy of the sample vault a
planted mention of `zero-gradient-prompt-filtering` is found while nine decoys around it, the same words in a
note that already links it and in the note itself are not.

The second part of `vault suggest`, **notes to write**, turns the links to notes that do not exist into a writing list: which missing notes the
most notes link to, where those notes live, what they are about, how many other notes mention the name without
linking it, names that overlap (the same idea twice, or a topic and a narrower one), and links that are
probably meant for an existing note (`--notes` shows only this part):

```
Suggestions: /tmp/sample-vault
  index 2026-11-07T06:36:27+00:00 (up to date)

  Notes to write  (612 link targets have no note; 3 most wanted, linked from 2+ notes)

    1. rl-environment-design
       52 notes link here · 1 more mention it
       in: 50_MOC/discourse/rl-environment-design (40) · 20_ZETTLEKASTEN (11) · 50_MOC/topics (1)
       about: evd, appworld, que, database, clm, state

    2. discourse-graph
       40 notes link here
       in: 50_MOC/discourse/rl-environment-design (40)
       about: evd, database, clm, que, appworld, state

    3. evaluation-methodology
```

On the sample it finds, among others, that `[[on-policy-vs-off-policy-learning]]` (5 notes) is the existing
`on-policy-versus-off-policy-learning`, that the PPO paper is linked under two spellings, and that
`llm-as-judge`, `llm-as-a-judge-framework` and `llm-as-judge-evaluation-pattern` (27 different notes link to one of them) have
no note. `--draft "<name>"` prints a starting note in this project's note format, built only from the vault:
what each linking note says around its link, the terms they share, the related notes `vault related` finds for
it (so it starts connected) and the linking sources as references; `--write` creates it (never overwriting),
after which `vault index` resolves every link to it:

```bash
vault suggest /tmp/sample-vault --draft reward-model-overoptimization-hump
```
```
Created: 07-11-2026 06:04

Tags: [[reward]] [[reinforcement-learning-in-language-models]] [[reward-hacking-as-specification-gaming]] [[evaluation-methodology]]

# reward-model-overoptimization-hump

> Drafted by `vault suggest` from the 12 notes that link here. Every line below comes from them: rewrite it in your own words, then delete this line.

## What the linking notes say

- [[2507.17746 - Rubrics as Rewards (RaR)]]: … "latches onto surface artifacts — response length, formatting quirks, annotator idiosyncrasies. That is the [[reward-model-overoptimization-hump]] and the length-bias literature; it is also the mechanism behind [[rlhf-amplifies-sycophancy]]. You optimize" …
- [[filtering-to-the-competence-frontier-as-free-curriculum]]: … "[[rlvr-reinforcement-learning-with-verifiable-rewards]] and would be fragile under a learned reward model ([[reward-model-overoptimization-hump]]). Compare [[consensus-filtering-for-process-supervision-data]], which filters *labels* by agreement rather" …
…
```

The "about" terms are weakest for broad topic hubs, whose linking notes share little vocabulary
(`experimental-design` gets "prime, composition, hub…"), and drafts lean toward what the linking notes are
mostly about (the overoptimization draft's "See Also" is rubric-heavy because several linking notes are about
rubrics). The draft is a starting point to rewrite, not a note.

## Over JSON-RPC

`vault serve` reads one JSON request per line on stdin and writes one response per line on stdout.

| Method | Does |
|---|---|
| `index.build`, `index.status` | update the index; is it up to date |
| `vault.stats`, `health.report` | the numbers of `vault stats` and `vault health` |
| `notes.pick`, `note.get` | fuzzy-find notes by title/alias; one note's properties, outline and links |
| `search.query` | the query language of `vault search`: ranked hits with where they matched, a snippet and marks, tag and folder counts |
| `note.preview` | a note's lines with a query's matches marked (what the app's preview shows) |
| `note.related` | `vault related` |
| `links.suggest`, `notes.suggest` | the two parts of `vault suggest` |
| `mentions.unlinked` | `vault unlinked`, grouped by the note mentioned |

Params, results and errors are defined once as Pydantic models (`vaultengine/rpc/models.py`); `vault rpc-schema`
prints the JSON Schema they produce (`vaultengine/rpc/schema.json`, from which the app's TypeScript types are
generated). Errors: the JSON-RPC codes, plus -32001 index not built, -32002 note not found, -32003/-32004/-32005/-32007 format, write and baseline errors (see [api.md](api.md)), and -32006 invalid
query, whose `data` says where the problem is (`start`/`end` offsets in the query) and how to fix it (`hint`).

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":1,"method":"notes.pick","params":{"query":"klpen","limit":1}}' \
  '{"jsonrpc":"2.0","id":2,"method":"notes.delete"}' \
  '{"jsonrpc":"1.0","id":3,"method":"index.status"}' \
  | vault serve /tmp/sample-vault
```
```
{"jsonrpc":"2.0","id":1,"result":{"query":"klpen","total":3,"results":[{"path":"20_ZETTLEKASTEN/kl-penalty-in-rlhf.md","title":"kl-penalty-in-rlhf","matched_field":"title","matched_text":"kl-penalty-in-rlhf","score":141,"positions":[0,1,3,4,5]}]}}
{"jsonrpc":"2.0","id":2,"error":{"code":-32601,"message":"Method not found","data":{"method":"notes.delete","available":["findings.list","health.report","idea.new","idea.set","index.build","index.status","links.suggest","mentions.unlinked","note.facts","note.get","note.ingest","note.link","note.merge","note.move","note.new","note.preview","note.related","note.set","notes.pick","notes.suggest","page.check","page.exclude","page.new","page.refresh","search.query","vault.check","vault.coverage","vault.fix","vault.next","vault.pipeline","vault.stats"]}}}
{"jsonrpc":"2.0","id":3,"error":{"code":-32600,"message":"Invalid Request","data":[{"loc":["jsonrpc"],"msg":"Input should be '2.0'","type":"literal_error"}]}}
```

```bash
printf '%s\n' \
  '{"jsonrpc":"2.0","id":4,"method":"search.query","params":{"query":"\"monitorability tax\" -path:10_SOURCES","limit":1,"facets":2}}' \
  '{"jsonrpc":"2.0","id":5,"method":"search.query","params":{"query":"reward (tag:ai-safety OR path:50_MOC"}}' \
  | vault serve /tmp/sample-vault
```
```
{"jsonrpc":"2.0","id":4,"result":{"query":"\"monitorability tax\" -path:10_SOURCES","tree":"(and (phrase \"monitorability tax\") (not (path \"10_SOURCES\")))","notes":180,"total":4,"order":"score","warnings":[],"clauses":[{"label":"\"monitorability tax\"","kind":"phrase","required":true,"notes":5,"variants":[]}],"hits":[{"path":"20_ZETTLEKASTEN/monitorability-tax.md","title":"monitorability-tax","score":6.714024840380222,"exact":1,"matches":[{"clause":"\"monitorability tax\"","exact":true,"term":null,"edits":0,"fields":[{"field":"title","count":1},{"field":"body","count":1}]}],"title_marks":[{"start":0,"end":18,"kind":"exact"}],"snippet":{"field":"body","text":"The monitorability tax is the cost you pay to keep a reasoning model's chain-of-thought legible as a","marks":[{"start":4,"end":22,"kind":"exact"}],"truncated_start":false,"truncated_end":true}}],"facets":{"tags":[{"name":"ai-safety","notes":4},{"name":"baby","notes":4}],"folders":[{"name":"20_ZETTLEKASTEN","notes":4}]}}}
{"jsonrpc":"2.0","id":5,"error":{"code":-32006,"message":"this ( is never closed","data":{"query":"reward (tag:ai-safety OR path:50_MOC","start":7,"end":8,"hint":"add ) after the group"}}}
```
On stderr it also prints `vaultengine: serving /tmp/sample-vault (index …) on stdio`.

## In the terminal app

| Type | You should see |
|---|---|
| `/note @klpen`, then Enter twice | the picker shows `❯ kl-penalty-in-rlhf  20_ZETTLEKASTEN`; the first Enter inserts it, the second shows the note: `Links out (10: 4 resolved)`, `Links in (1)` |
| `/search reward hacking` Enter | `⏺ Search “reward hacking”  106 of 180 notes · exact matches first, then score` and a list: each hit with its score, where it matched and a snippet with the matches highlighted. ↑ ↓ (PgUp PgDn, Home End) move |
| then Enter | a preview of the selected note, opened at its first match, every match highlighted (`N matches · lines …`); ↑ ↓ PgUp PgDn scroll, `n` / `N` jump to the next / previous match |
| then `o` | opens the note in Obsidian (`obsidian://open?vault=sample-vault&file=20_ZETTLEKASTEN%2F…md`) and says `↗ Opened … in Obsidian`; works in the list too |
| then Esc, Esc | back to the list (same note selected), then closed: the top results stay in the transcript, the prompt is back |
| `/search reward (tag:ai-safety` Enter | `⏺ Invalid query: this ( is never closed`, the query, a `^` under the `(`, and `add ) after the group` |
| `/related @klpen` (pick, Enter) Enter | `⏺ Related to kl-penalty-in-rlhf`, each note with its score, `linked` / `not linked` and the reason (`2 hops via direct-preference-optimization-dpo +1 · similar text (kl, dpo, rlhf)`) |
| `/suggest` Enter | `⏺ Links to add` (pairs with their evidence) and `⏺ Notes to write`; `/suggest links`, `/suggest notes`, `/suggest @note` for one part |
| `/unlinked @monitorability` (pick, Enter) Enter | `⏺ Unlinked mentions of monitorability-tax  4 mentions in 1 note…`, each with the mention highlighted; `/unlinked` alone lists the most mentioned notes |

`o` hands the `obsidian://` URL to `open` (macOS), `xdg-open` (Linux) or `start` (Windows); set `VAULT_TUI_OPENER` to
any program that takes the URL as its argument to use something else. The vault name in the URL is the vault
folder's name, which is what Obsidian calls the vault unless you renamed it there.
