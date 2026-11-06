/**
 * Prompt history with shell semantics: ↑ goes back, ↓ goes forward, and the
 * line you were typing (the draft) comes back when you move past the newest
 * entry. Consecutive duplicates are stored once. In memory for the POC.
 */
export class PromptHistory {
	private readonly entries: string[] = [];
	private position: number | undefined; // index into entries while browsing
	private draft = '';

	constructor(private readonly limit = 500) {}

	get size(): number {
		return this.entries.length;
	}

	add(line: string): void {
		const value = line.trim();
		if (value && this.entries.at(-1) !== value) {
			this.entries.push(value);
			this.entries.splice(0, Math.max(0, this.entries.length - this.limit));
		}

		this.position = undefined;
		this.draft = '';
	}

	/** The entry to show after ↑, or undefined when there is nothing older. */
	previous(current: string): string | undefined {
		if (this.entries.length === 0) {
			return undefined;
		}

		if (this.position === undefined) {
			this.draft = current;
			this.position = this.entries.length - 1;
		} else if (this.position > 0) {
			this.position -= 1;
		} else {
			return undefined;
		}

		return this.entries[this.position];
	}

	/** The entry to show after ↓ (the draft when leaving the history), or undefined. */
	next(): string | undefined {
		if (this.position === undefined) {
			return undefined;
		}

		if (this.position < this.entries.length - 1) {
			this.position += 1;
			return this.entries[this.position];
		}

		this.position = undefined;
		return this.draft;
	}
}
