/**
 * Shell-like splitting of a command's arguments: words, "quoted words", and
 * @mentions (kept as tokens of kind "mention", in either @path or @"path" form).
 */

export type Token = {kind: 'word' | 'mention'; value: string};

const TOKEN = /@"([^"]*)"|@(\S+)|"([^"]*)"|'([^']*)'|(\S+)/g;

export function tokenize(text: string): Token[] {
	const tokens: Token[] = [];
	for (const match of text.matchAll(TOKEN)) {
		if (match[1] !== undefined || match[2] !== undefined) {
			tokens.push({kind: 'mention', value: match[1] ?? match[2] ?? ''});
		} else {
			tokens.push({kind: 'word', value: match[3] ?? match[4] ?? match[5] ?? ''});
		}
	}

	return tokens;
}

/** Removes `flag` (e.g. "--dry-run") from the tokens; returns whether it was there. */
export function takeFlag(tokens: Token[], ...names: string[]): boolean {
	const index = tokens.findIndex((token) => token.kind === 'word' && names.includes(token.value));
	if (index < 0) {
		return false;
	}

	tokens.splice(index, 1);
	return true;
}

/** A positive whole number, or undefined; throws with `usage` otherwise. */
export function parseLimit(value: string | undefined, usage: string): number | undefined {
	if (value === undefined) {
		return undefined;
	}

	const limit = Number.parseInt(value, 10);
	if (!Number.isInteger(limit) || limit < 0 || String(limit) !== value) {
		throw new Error(`Usage: ${usage} (got "${value}")`);
	}

	return limit === 0 ? undefined : limit;
}
