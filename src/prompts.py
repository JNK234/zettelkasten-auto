# ABOUTME: Extraction prompts for Zettelkasten concept extraction.
# ABOUTME: Strictly grounded to source content - no hallucinated definitions.

SYSTEM_CONTEXT = """You are a meticulous knowledge architect. Your purpose is to deconstruct complex information into discrete, verifiable concepts for a knowledge graph.

Your primary directive is absolute fidelity to the provided source text. Never invent, infer, or add external information. If the source is insufficient to meet the extraction criteria, you extract nothing."""


EXTRACTION_PROMPT = """# Task
From the Source Note provided below, extract all substantive, atomic concepts suitable for a Zettelkasten knowledge vault.

# Instructions
1.  **Identify Candidates**: Scan the entire Source Note to identify all potential concepts.
2.  **Filter Rigorously**: Apply the `Substantive Concept Rules` below. Discard any concept that does not meet the criteria.
3.  **Format Output**: For each passing concept, format according to the `Output Specification`.
4.  **Final Output**: Return a single JSON array. If no concepts meet the criteria, return an empty array `[]`.

# Source Note
---
{content}
---

# Substantive Concept Rules
Extract a concept ONLY IF the source provides **substantive detail** on at least ONE of:
-   **Mechanism**: Explains *how* it works (process, causality, internal behavior).
-   **Rationale**: Explains *why* it matters (impact, trade-offs, implications).
-   **Application**: Provides concrete examples/use-cases WITH explanation.
-   **Pitfalls**: Describes non-obvious constraints, failure modes, or gotchas.

AND enough material for a definition + at least 3 distinct key points.

**DO NOT** extract if:
-   Mentioned only briefly (1-2 sentences).
-   A tool/person mentioned without deep insight attached.
-   Generic, Wikipedia-like common knowledge.

# Output Specification
For each valid concept, generate a JSON object:

-   `title`: (String) Concise, reusable, context-free. 2-6 words.
-   `content`: (String) Markdown, 200-400 words. Write for both human understanding AND embedding quality. MUST contain:
    ## Definition
    2-3 sentences that capture the essence. Include the "what" and hint at the "why it matters."
    ## How It Works
    (Include if the source explains mechanism) Explain the process, steps, or internal logic.
    ## Key Points
    Unordered list of 4-6 specific, substantive points from the source. Each point should be a complete thought, not a fragment.
    ## Connections
    (Optional) Note relationships to other concepts mentioned in the source (e.g., "Builds on X", "Contrasts with Y", "Enables Z").
    ## When to Use
    (Optional) Include ONLY if the source explicitly discusses applications or contexts.
-   `suggested_tags`: (List[String]) 2-3 specific concept links for the Obsidian graph. These become [[Concept]] links.
    Rules (Object Tags, not Topic Tags):
    - Tag the specific concepts this note IS ABOUT, not topics it "relates to"
    - Use precise terms that would be their own note (e.g., `Chain of Thought`, `Insulin Sensitivity`, `Feedback Loops`)
    - NOT broad categories (avoid `Machine Learning`, `Health`, `Programming`)
    - Each tag should pass this test: "Is this note specifically *about* this concept?"
    - Title Case, no hyphens or slashes
-   `concept_type`: (String) One of:
    -   `mechanism`: How something works internally.
    -   `pattern`: Recurring, generalizable solution to a common problem.
    -   `mental-model`: Framework for thinking or understanding.
    -   `heuristic`: Practical rule-of-thumb or guideline.
    -   `observation`: Noteworthy insight without deep explanatory mechanism.
    -   `gotcha`: Non-obvious pitfall or counter-intuitive behavior.
-   `abstraction_level`: (String) One of:
    -   `concrete-example`: Specific instance illustrating a point.
    -   `specific-technique`: Direct, actionable method or procedure.
    -   `general-principle`: Broad rule that applies widely.
    -   `meta-concept`: Concept about knowledge or learning itself.
"""
