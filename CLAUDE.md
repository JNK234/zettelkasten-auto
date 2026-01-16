# Claude Development Instructions

## Mission

Build a powerful yet simple system for automating Zettelkasten note creation. The goal is to transform raw source notes into a rich, interconnected knowledge network while keeping the user in full control of their data.

## Core Principles

### Simplicity First
- Prefer fewer files over many modules
- Avoid abstractions until they prove necessary
- Each function should do one thing well
- If a feature can be removed without breaking core functionality, consider removing it

### Incremental Development
- Add features only when needed, not speculatively
- Test each change before adding the next
- Small commits with clear purpose
- Refactor only when complexity becomes a problem

### Local and Secure
- All data stays on the user's machine
- No telemetry or external data transmission beyond LLM API calls
- ChromaDB runs locally with persistent storage
- API keys stored in .env, never committed

### User Control
- Source files are never deleted, only marked as processed
- Generated Zettels can be manually edited without breaking the system
- Configuration is explicit and readable (YAML)
- No magic or hidden behavior

## Code Guidelines

### What to Avoid
- Over-engineering: No factories, abstract base classes, or dependency injection for a 200-line script
- Speculative features: Do not add config options that are not implemented
- Dead code: Remove unused imports, functions, and variables immediately
- Mock modes: Always use real APIs and data

### What to Prefer
- Direct function calls over class hierarchies
- Dicts over dataclasses unless IDE autocompletion is needed
- Explicit paths over clever path resolution
- Print statements for progress over logging frameworks (for now)

### File Operations
- Use pathlib.Path throughout
- Always check if file exists before reading
- Never overwrite without explicit intent
- Preserve file encoding (UTF-8)

### LLM Interaction
- LLM outputs JSON only, never markdown or file content directly
- All file writing happens in Python, not by LLM
- Prompts live in prompts.py for easy iteration
- Use structured outputs for reliable parsing

## Architecture Constraints

### Current Structure (Keep It)
```
src/
  db.py       # ChromaDB operations only
  llm.py      # LLM calls only
  prompts.py  # Prompt strings only
  main.py     # Everything else
```

### When to Add a New File
- Only if it represents a truly distinct concern
- Only if the current file exceeds 200 lines
- Only if multiple other files would import from it

### When NOT to Add a New File
- For "organization" or "cleanliness"
- For a single function
- For config or constants (keep in existing files or config.yaml)

## Future Enhancements (Do Not Implement Yet)

These features may be added later but should not be built speculatively:
- Batch processing with progress bars
- Multiple LLM provider support (Anthropic, local)
- MOC (Map of Content) generation
- Bidirectional link updates
- Web UI or TUI
- Database migrations

## Testing Philosophy

- Manual testing is acceptable for a tool this size
- Add automated tests only for complex logic (embedding similarity, tag parsing)
- Do not add tests for trivial functions
- Integration tests are more valuable than unit tests here

## Commit Messages

- Describe what changed, not why (the PR or issue has context)
- Keep under 72 characters for the first line
- No AI attribution or generated-by messages
