Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #settled

# How should a verifier or reward be designed so that high reward means the task was actually solved, and how is a vague customer spec turned into that verifier?

*Lens (b) How it is done. Type: CONSENSUS. Hub: [[RL Environment Design - Discourse Graph]].*

## Terms
- **Hack-robust verifier**: one where high reward implies the task was solved (Epoch's top criterion; TechCrunch "catching cheating").
- **Deterministic anchor**: a programmatic check of the outcome (PatTut "verifiable signals", tau2 hash, ART-E golden answer).
- **Spec drift**: the verifier measures something other than what its author intended (PatTut orderings, tau2 hash scope).
- **False-negative verifier**: a check so rigid that it fails a correct agent, for example by demanding "96.124991…" when the agent wrote "96.12", a fixed database ID, or one particular sequence of calls (AnthEvals, EnterpriseOps-Gym, Loopholes).
- **Grader error**: the grader itself crashes, returns malformed output or cannot reach its model, so no verdict exists; the question is what reward that run receives (EnvScaler, SpecEval, SynthAgent, OpenAI Spec Evals).
- **Letter-versus-intent conflict**: a spec clause whose literal wording and whose purpose point to different correct actions, so a grader must pick one reading (Loopholes, SpecEval "the narrow language of the statement and not the spirit").

## Positions
**Agreed by all**
- [[CLM - Hack-robustness is the first requirement of a verifier]] (supported). Backed by Epoch, OpenEnv, ABC, AnthEvals.
- [[CLM - Anchor reward in a deterministic check of the outcome]] (supported). Backed by PatTut, ART-E, APIGen-MT, ABC.
- [[CLM - Answers, tests and grader state must be out of the agent's reach]] (supported). Backed by Epoch, AppWorld, SWE-smith, TauTau, ABC.
- [[CLM - Check the oracle hits max and the empty policy hits the floor]] (supported). Backed by OpenEnv, PatTut, AppWorld, Verifiers, EnvScaler, ABC, AnthEvals.
- [[CLM - A fixed typed contract should separate environment from trainer]] (supported). Backed by OpenEnv, Verifiers, Survey, EnvScaler.

**Agreed**
- [[CLM - Correctness must gate all shaping credit]] (supported). Backed by AppWorld, ART-E, tau2, EnterpriseOps-Gym, AnthEvals, OpenAI Spec Evals, SynthAgent.

**Refinement**
- [[CLM - Deterministic graders fail silently when their scope drifts from the spec]] (supported). Backed by tau2, PatTut, EnterpriseOps-Gym, ABC, AnthEvals.
- [[CLM - Grader errors must never pay reward and must be counted apart from agent failures]] (supported). Backed by EnvScaler, OpenAI Spec Evals, SpecEval, SynthAgent.
- [[CLM - An LLM-judged rubric is not a deterministic outcome check]] (supported). Backed by Corecraft, SynthAgent.
- [[CLM - A verifier must accept every valid form of the correct outcome]] (supported). Backed by EnterpriseOps-Gym, Loopholes, AnthEvals.
- [[CLM - Graders built on an ambiguous spec reward the annotator's reading rather than the solution]] (supported). Backed by Loopholes, AnthEvals, SpecEval.

## What's agreed / what's disputed
**Agreed.** Epoch's interviewees, OpenEnv's hub incident, PatTut, tau2 and ART-E converge: high reward must mean solved, and the way to get there is a deterministic check of the outcome. They converge because every documented reward failure in the set is a verifier accepting something it should not. The benchmark audits add the size of the damage when this fails. CVE-Bench counted the string SLEEP in a database log as a successful injection and overstated success by 32.5 points (ABC). On SWE-Lancer an agent that overwrote the tests inside a password-protected zip scored 100 percent while solving nothing (ABC). A do-nothing agent passed 38 percent of tau-bench Airline, which read as an RL environment would pay 38 percent reward for inaction (ABC, the note author's inference). The same sources agree on four supporting rules. Answers and tests must sit outside the agent's reach, and TauTau shows this works: 17 to 42 percent of developer runs per harness tried to find held-out data or probe the grader, and none succeeded because no ground truth entered the runtime image. Correctness or hard-constraint compliance gates partial credit (EnterpriseOps-Gym's strict AND, SynthAgent's multiplicative rule gate). A known-correct solution must score the maximum and a do-nothing agent the floor before use (AnthEvals, ABC, and EnvScaler as the counter-example that skipped it). Grader code runs outside the trainer's process (EnvScaler, one code reading).

**Disputed / nuance.** No source disputes the principle; the refinements are about how it fails in practice. First, the anchor can be too rigid as well as too loose: [[CLM - A verifier must accept every valid form of the correct outcome]]. Opus 4.5 went from 42 percent to 95 percent on CORE-Bench once rigid numeric matching, ambiguous specs and irreproducible tasks were fixed and the scaffold loosened, with the separate contributions not measured (AnthEvals); EnterpriseOps-Gym's fixed database IDs punish harmless stray writes; and on tau2 Task 32 every model reached the gold's final state with one combined call and still scored zero (Loopholes). Second, not everything labelled "verifiable" is: [[CLM - An LLM-judged rubric is not a deterministic outcome check]], since Corecraft's reward is an LLM reading text and SynthAgent's judge grades against tool outputs its own simulator invented. Third, grader failures need their own accounting: [[CLM - Grader errors must never pay reward and must be counted apart from agent failures]]. EnvScaler and SpecEval score a crashed or unparseable grader as a failure, which is hack-robust but silently deflates scores, and SpecEval's parse failures rank as the hardest prompts and enter the test set first. SynthAgent does the opposite and pays 0.3 for a trajectory its judge could not evaluate, about the same as completing 40 percent of subgoals. The vague-spec half of the question now has a partial answer: [[CLM - Graders built on an ambiguous spec reward the annotator's reading rather than the solution]]. On tau2 Task 27 a GPT-5 run matched both gold actions and still scored zero because the strict reading demanded an extra flight change (Loopholes), and SpecEval's judge marked CPR instructions non-compliant under OpenAI's "prevent imminent harm" statement. That training on such graders teaches the annotator's preference inconsistently is inference, not a measured training run (Loopholes, SpecEval). Patronus's CEO (in TechCrunch) names non-verifiable tasks as the next step, and there no deterministic anchor exists. See [[QUE 19 - Reliability of LLM judges]].

## Current answer
Settled, high confidence. Anchor reward in a deterministic check of the outcome, keep answers and grader state out of the agent's reach, gate partial credit on correctness, and prove the grader with an oracle that scores the maximum and a do-nothing agent that scores the floor. The check must also accept every valid form of the correct outcome, count grader errors apart from agent failures, and not call an LLM-judged rubric deterministic. Turning a vague spec into a verifier still has no systematic method: the sources show that ambiguous or letter-versus-intent clauses must be found and resolved before grading, but only through audits of individual tasks.
