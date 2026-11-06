Tags: #ai-safety #prompt-engineering #agentic-ai-systems
Status: #baby
## What This Is
The structural root cause behind prompt injection: an LLM-integrated app dumps the user's request *and* retrieved external text into the **same context window, on the same footing**. The model has no architecturally-enforced way to tell "this is content I should reason *about*" from "this is an instruction I should *obey*." Trusted instructions and untrusted data are one undifferentiated stream of tokens, so any text in context can act as a command (per source).

## Why It Matters
It explains why you cannot prompt-engineer your way out. The vulnerability is plumbing, not prompt wording. As long as apps splice untrusted retrieved text and trusted instructions into one channel, the attack class is **inherent — not a bug to be patched**. The realistic fix is architectural: separate trusted instructions from untrusted data, least-privilege the tools, sandbox effects, and monitor actions (per source).

## Key Details
- RLHF/safety tuning is "Whack-A-Mole": individual jailbreaks get patched but the underlying data-vs-instruction confusion remains, so new bypasses keep appearing (per source).
- Input filtering is a dilemma: a *weak* filter is beaten by encoded payloads (Base64/ROT13); a *capable* instruction-tuned filter can decode them but is then **itself injectable**, inheriting the exact vulnerability it was meant to catch (per source).
- More promising but mostly unbuilt: dedicated LLM supervisors that look for *attack signatures* (not just harmful output), and interpretability-based outlier detection on the model's internal prediction trajectory to flag "this looks hijacked" (per source).
- Verdict from the authors: "it is currently hard to imagine a foolproof solution" — posture is defense-in-depth, none sufficient alone (per source).

## Tensions
- This is the mechanism behind [[indirect-prompt-injection-threat-class]] and the same root the multi-agent notes name ([[prompt-injection-in-multi-agent-natural-language-communication]]) — but framed at the single-app architecture level.
- The "monitor between model and world" prescription is exactly an AI-control posture — see [[control-evaluations-red-team-blue-team]] and resampling-as-veto in [[Ctrl-Z - Controlling AI Agents via Resampling]].
- The interpretability-detector idea connects to [[sparse-autoencoders-for-interpretability]] and probe-based monitoring like [[coup-probes-off-policy-detection]].

## References
1. [[2302.12173 - Not What Youve Signed Up For (Indirect Prompt Injection)]]
