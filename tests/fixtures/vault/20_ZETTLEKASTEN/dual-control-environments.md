Created: 24-09-2026 16:29

Status: #baby

Tags: [[multi-agent-systems]] [[agentic-ai-systems]] [[rl-environment-design]]

# Dual-Control Environments

## What This Is
In most tool-use benchmarks, only the agent can change the world. Real tech support isn't like that. The agent can't touch your phone; it can only tell you what to tap and then read back what you report. A dual-control environment gives the **simulated user** their own tools and their own state, so task success depends on the agent coordinating another actor, not just calling APIs.

τ²-bench's telecom domain is the reference build. The user simulator has 30 tools on a simulated phone (`check_status_bar`, `toggle_airplane_mode`, `reseat_sim_card`, `toggle_roaming`, `run_speed_test`, ...). The agent has only its own backend tools plus the chat channel.

## Why It Matters
It tests a capability that single-control tasks can't: instructing someone else precisely and verifying their report. It also lets you separate failure causes. τ² runs `LLMSoloAgent`, which gets a ticket and no user at all, to split "can't reason about the fix" from "can't coordinate the user through it".

## Key Details
- Two states are graded: the agent's DB and the user's device. Both hashes must match the gold replay.
- Telecom success uses `env_assertions` run on the **user's** phone, e.g. `assert_mobile_data_status(expected_status=True)`, `assert_internet_speed(expected_speed=200)`. This checks "is the customer's problem actually fixed?", not "did the agent follow the script?"
- `banking_knowledge` extends dual control through documents: `give_discoverable_user_tool(name, args)` hands the customer a tool (e.g. "apply for the card yourself").
- The Gym wrapper exposes both sides: `AgentGymEnv` (train the agent against a simulated user) and `UserGymEnv` (play the user against an automated agent).

## Tensions
- Adds a second LLM to the loop, so scores mix agent error with simulator error. See [[llm-user-simulator-design-and-auditing]].
- Interface design questions overlap [[multi-agent-environment-interface-design]] and [[agents-environment-interface-types]].
- Coordination failure vs reasoning failure is a confound in the sense of [[partner-quality-confounding]]: a weak user simulator lowers the agent's score.

# References
1. [[sierra-research-tau2-bench - Agent Evaluation Harness]] (Layer 1, Layer 2 "How the DB check actually works", Layer 3, Layer 6)
