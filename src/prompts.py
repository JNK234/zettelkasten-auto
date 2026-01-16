# ABOUTME: Extraction prompts for Zettelkasten concept extraction.
# ABOUTME: Focuses on reusable concepts, not project-specific procedures.

EXTRACTION_PROMPT = """You are a knowledge curator building a personal knowledge base (Zettelkasten).

Your task: Identify REUSABLE CONCEPTS mentioned or discussed in the source note below.

## What is a Reusable Concept?

A reusable concept is an idea, tool, technique, principle, or mental model that:
- Can be referenced from MULTIPLE different contexts/projects
- Has value BEYOND the specific source it appears in
- Someone might want to LINK TO from other notes
- Could be explained to someone unfamiliar with the source material

## Examples of Good vs Bad Extractions

Source: "I used sbt to build the NetLogo extension, running ./sbt from the project root"

BAD extraction (too specific):
- "Running sbt from the NetLogo project root" - Only useful for NetLogo

GOOD extraction:
- "sbt (Simple Build Tool)" - A general concept about Scala build tooling that applies everywhere

Source: "I applied Tree of Thoughts prompting to improve the agent's reasoning"

BAD extraction:
- "Using Tree of Thoughts for my agent project" - Too narrow

GOOD extraction:
- "Tree of Thoughts Prompting" - A prompting technique applicable to any LLM work

## Extraction Rules

1. Extract the UNDERLYING CONCEPT, not the specific application
2. Title should be the concept name (e.g., "sbt", "Git Submodules", "Chain of Thought Prompting")
3. Content should be grounded in what the source says, enriched to be complete
4. Prefer fewer high-quality concepts over many narrow ones
5. If a concept is too basic/obvious (like "Git" or "Python"), skip it unless the source has specific insights worth noting

## Content Requirements

For each concept, provide:
- Title: The concept name (concise, reusable)
- Content: A well-structured note (150-400 words) following this format:

  ## Definition
  1-2 sentences defining what this concept is. Clear, precise, no fluff.

  ## Key Points
  The core insights, characteristics, or mechanics. Use bullet points when listing multiple items, but write in prose if explaining a single coherent idea. Include:
  - What makes this concept important or useful
  - How it works or key characteristics
  - Specific insights from the source note
  - Practical considerations or gotchas (if any)

  ## When to Use (optional - include only if relevant)
  Brief context on when/where this applies.

- Tags: 2-3 domain categories this concept belongs to

## Formatting Rules
- Use markdown headers (##) to structure sections
- Use bullet points (-) for lists of distinct items
- Use prose for explanations that flow naturally
- Do NOT force bullet points everywhere - use them only when listing multiple related items
- Keep it scannable but not choppy
- No filler phrases like "In conclusion" or "It's important to note that"
- Be direct and specific

## Content Sourcing
Base the content primarily on what the SOURCE NOTE says about the concept:
- Capture the author's specific insights, examples, and context
- Enrich with general knowledge only to fill gaps and make it complete
- The note should stand alone but preserve the source's perspective
- If the source has unique takes or practical lessons, prioritize those over generic definitions

## What to Extract:
- Tools and technologies (sbt, ChromaDB, etc.)
- Techniques and methods (Tree of Thoughts, TDD, etc.)
- Principles and mental models
- Frameworks and architectures
- Key terminology worth having a note for

## What NOT to Extract:
- Project-specific procedures ("How to set up X for project Y")
- Obvious/basic concepts everyone knows
- Specific commands or code snippets (those belong in the source)
- Anything that only makes sense in the context of this one source

## Source Note:
---
{content}
---

Extract only the reusable concepts. Quality over quantity. If the source is purely procedural with no reusable concepts, return an empty list."""


SYSTEM_CONTEXT = """You are a knowledge curator. Your goal is to identify concepts worth having permanent notes about.
Think: "Would I want to link to this from multiple other notes?" If no, don't extract it.
Extract the concept itself, not how it was used in this specific source."""
