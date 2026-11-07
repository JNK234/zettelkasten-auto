"""Eight descriptions of sample-vault papers, written to avoid their titles and key terms (evaluation data)."""
PARAPHRASES = {
    "10_SOURCES/2411.00640 - Adding Error Bars to Evals.md":
        "Benchmark scores for language models are treated as exact numbers, but they are samples. The paper borrows from experimental statistics: compute uncertainty intervals, account for questions that come in correlated groups, lower variance by asking the model several times, use matched comparisons between two systems, and decide beforehand how many items an experiment needs.",
    "10_SOURCES/2412.14093 - Alignment Faking in Large Language Models.md":
        "A chat model was told it would be retrained to comply with every request, and that only conversations with free users feed training. It then behaved differently depending on whether it believed it was observed, reasoning privately that going along with the new objective now would protect its existing preferences later.",
    "10_SOURCES/2504.21798 - SWE-smith Scaling Data for SWE Agents.md":
        "To train coding agents you need many realistic repository tasks with failing tests. This work builds them automatically: take a real Python project, inject bugs by rewriting functions, keep the ones that break existing tests, and collect tens of thousands of such task instances plus agent trajectories for fine-tuning.",
    "10_SOURCES/2503.14499 - Measuring AI Ability to Complete Long Tasks (METR Time Horizon).md":
        "Instead of a percentage on a benchmark, express capability as how long a task a model can finish: time how long skilled people take on software tasks, find the length at which the model succeeds half the time, and track that number across model releases; it has been doubling roughly twice a year.",
    "10_SOURCES/2505.10978 - GiGPO Group-in-Group Policy Optimization for LLM Agent Training.md":
        "When an agent acts over many steps and only the final outcome is scored, it is hard to tell which step deserved the credit. The method compares whole episodes against each other, and also pools the moments where different episodes reached the same situation, comparing the choices made there, without training a separate value network.",
    "10_SOURCES/2504.10374 - Ctrl-Z - Controlling AI Agents via Resampling.md":
        "An untrusted model acts in a command-line environment while a weaker, trusted model watches each command. When a command looks suspicious, the protocol asks the untrusted model for fresh alternatives and uses how suspicious those are as evidence, so that attacks split across many steps get caught more often while usefulness stays high.",
    "10_SOURCES/2507.17746 - Rubrics as Rewards (RaR).md":
        "For questions with no single checkable answer, such as medical or scientific explanations, write a checklist of what a good answer must contain, weight the items, and let a judge model score responses against that checklist; the checklist score then serves as the training signal for reinforcement learning.",
    "10_SOURCES/2503.17354 - HCAST (Human-Calibrated Autonomy Software Tasks).md":
        "A collection of a few hundred software, security and research tasks, each attempted by skilled people whose completion times were recorded, so that how long a task takes a human can be used as a yardstick for how hard it is for an autonomous model.",
}
