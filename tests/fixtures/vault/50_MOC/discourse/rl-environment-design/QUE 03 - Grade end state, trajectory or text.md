Created: 24-09-2026 19:20

Tags: [[discourse-graph]] [[rl-environment-design]]

Status: #partly-settled

# What should the grader inspect: final world state, the action trajectory, or text judged against a reference, and how are 'correct answer is do nothing' and collateral-damage cases handled?

*Lens (b) How it is done. Type: CONTESTED. Hub: [[RL Environment Design - Discourse Graph]].*

## Terms
- **State grading**: check the final environment state (DB diff, hash). AppWorld, tau2; tau2's phrase is "grade the world, not the words".
- **Trajectory grading**: inspect the sequence of actions or the trace (TRAIL's span-level view; tau2's process violations).
- **Text grading**: compare the output with a reference (ART-E's judge against a golden answer).
- **Scope**: which side effects the state check includes (tau2 hash scope, AppWorld allow-list).
- **No-op-correct task**: one where refusing or doing nothing is the right answer (tau2 refusal tasks, AppWorld `no_op_pass`).
- **Collateral check**: a check that state which should not change did not change, so the grader diffs before and after instead of reading only the final state (ABC's "check both the states that should change and the ones that should not", EnvScaler).
- **Required actions**: a per-task list of lookups the trajectory must contain for the reward to be non-zero, matched by exact name and arguments (ABC's tau-bench fix).
- **Subgoal checklist**: a per-task list of subgoals, required interactions and forbidden behaviours, "stated at a level that allows several valid paths", used to grade a trajectory when there is no real state to inspect (SynthAgent).

## Positions
**State grading**
- [[CLM - Grade the observed end state rather than the trajectory]] (supported). Backed by AppWorld, tau2, EnvScaler, APIGen-MT, Anthropic. Opposed by tau2.

**Trajectory grading**
- [[CLM - Process violations require checking the trajectory]] (contested). Backed by tau2, ABC. Opposed by TRAIL.

**Refinement**
- [[CLM - Deterministic graders fail silently when their scope drifts from the spec]] (supported). Backed by tau2, PatTut, EntOps, ABC, Anthropic.
- [[CLM - Graders must handle do-nothing-correct tasks explicitly]] (supported). Backed by tau2, AppWorld, EntOps, ABC.
- [[CLM - A verifier must accept every valid form of the correct outcome]] (supported). Backed by EntOps, Loopholes, Anthropic.
- [[CLM - State graders should check what must not change as well as what must]] (supported). Backed by EnvScaler, ABC.
- [[CLM - Refusal tasks must require evidence that the agent checked the policy]] (supported). Backed by ABC.
- [[CLM - Trajectory checks belong as a layer on top of outcome checks]] (supported). Backed by Anthropic, Surge.

**Nuance**
- [[CLM - Exact-action trajectory checks penalise valid alternative paths]] (supported). Backed by Loopholes, ABC.
- [[CLM - Information tasks need the answer checked against the data, not only state]] (weak). Backed by Surge.

**Alternative route**
- [[CLM - Without a real end state, grade the trajectory against path-agnostic subgoals]] (supported). Backed by SynthAgent.

**Resolution**
- [[CLM - Grading end state lets the action space open safely]] (supported). Backed by AppWorld, tau2, TauTau, APIGen-MT.

## The issue
The disagreement is less sharp than it looks. [[CLM - Grade the observed end state rather than the trajectory]] and [[CLM - Process violations require checking the trajectory]] are **different-context**: state grading covers what changed; trajectory checks are needed only for violations that leave state unchanged. tau2 itself holds both positions. The real opposition is about feasibility: TRAIL shows current judges cannot yet do the trajectory job reliably.

The added sources narrow the gap from both sides. On the state side, end-state grading is the stated default of EnvScaler, APIGen-MT (whose ground truth lists only write calls, never lookups), tau^tau-bench and Anthropic's eval guide, which calls fixed tool-call sequences "too rigid" because agents find valid routes nobody anticipated. Two refinements make state grading stronger: [[CLM - State graders should check what must not change as well as what must]] (EnvScaler's check functions receive the initial state; ABC's O.g.2) and [[CLM - A verifier must accept every valid form of the correct outcome]]. The do-nothing problem is now measured: on tau-bench, refusal tasks have an empty gold action list, so an agent that changes nothing passes 38 percent of airline and 6 percent of retail tasks. EnterpriseOps-Gym handles the same case by giving each of its 30 infeasible tasks about 10 checks that nothing changed.

On the trajectory side, ABC's tau-bench fix is the first concrete trajectory check in the set. It adds required lookups per task and zeroes the reward if any is missing, which backs [[CLM - Process violations require checking the trajectory]] and [[CLM - Refusal tasks must require evidence that the agent checked the policy]]. The same fix shows the cost: it matches lookups by exact arguments, so a valid alternative lookup path scores 0, and the Loopholes author concedes that procedural checks still need one unambiguous reference procedure. [[CLM - Exact-action trajectory checks penalise valid alternative paths]] **refines** the trajectory claim: a trajectory check is sound only where the spec fixes one procedure.

The shape that emerges is layered, and this is inference from the set rather than a measured result. Grade state first. Add [[CLM - Trajectory checks belong as a layer on top of outcome checks]] for what state cannot show, such as Surge's agents searching all 31 August days one at a time, or how the user was treated. Check the final answer against the data for read-only tasks ([[CLM - Information tasks need the answer checked against the data, not only state]]). Where no real state exists, grade the trajectory against [[CLM - Without a real end state, grade the trajectory against path-agnostic subgoals]].

**Crux.** A process-aware grader validated against humans on span-level localisation over long traces. Nothing in the set provides one. A cheaper experiment would settle the narrower question of how to grade refusals: score the same tau-bench refusal trajectories under end-state grading, exact-argument required lookups and a path-tolerant lookup check, and compare each against human verdicts to count false passes and false fails.

## Current answer
Partly settled, high confidence on the default, medium on the layers above it. Grade the end state wherever it can show success, check what must not change as well as what must, accept every valid end state, and make do-nothing-correct tasks require evidence that the agent checked the policy. Trajectory checks work as a second layer only where the spec fixes one procedure, and process violations that leave state unchanged remain unsolved at scale.
