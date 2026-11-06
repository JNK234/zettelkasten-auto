/**
 * @-mentions of notes in the prompt, the way Claude Code mentions files.
 *
 * A completed mention is `@path` (no spaces) or `@"path with spaces"`. The
 * picker inserts the quoted form when needed. Any command can read the
 * mentions of its input; `/note` uses the first one.
 *
 * While typing, the "active" mention is an `@` at the start of the line or
 * after whitespace, followed by non-space characters up to the cursor. The
 * engine's fuzzy search is subsequence-based, so spaces are never needed in
 * the query: "klpen" finds "kl-penalty-in-rlhf".
 */

export type Mention = {path: string; start: number; end: number};

const MENTION = /(^|\s)@(?:"([^"]*)"|([^\s"]+))/g;

export function parseMentions(text: string): {mentions: Mention[]; rest: string} {
	const mentions: Mention[] = [];
	for (const match of text.matchAll(MENTION)) {
		const start = match.index + (match[1]?.length ?? 0);
		const path = match[2] ?? match[3] ?? '';
		if (path) {
			mentions.push({path, start, end: match.index + match[0].length});
		}
	}

	let rest = text;
	for (const mention of [...mentions].reverse()) {
		rest = rest.slice(0, mention.start) + rest.slice(mention.end);
	}

	return {mentions, rest: rest.replaceAll(/\s+/g, ' ').trim()};
}

export function formatMention(path: string): string {
	return /\s/.test(path) ? `@"${path}"` : `@${path}`;
}

export type ActiveMention = {start: number; query: string};

export function activeMention(text: string, cursor: number): ActiveMention | undefined {
	const before = text.slice(0, cursor);
	const at = before.lastIndexOf('@');
	if (at < 0 || (at > 0 && !/\s/.test(before[at - 1] ?? ''))) {
		return undefined;
	}

	const query = before.slice(at + 1);
	if (/[\s"]/.test(query)) {
		return undefined; // finished (followed by a space) or a quoted, completed mention
	}

	return {start: at, query};
}
