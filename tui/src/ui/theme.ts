export const theme = {
	accent: '#D97757',
	ok: 'green',
	warn: 'yellow',
	error: 'red',
	link: 'cyan',
} as const;

export const formatNumber = (value: number): string => value.toLocaleString('en-US');

export const plural = (count: number, word: string, many = `${word}s`): string =>
	`${formatNumber(count)} ${count === 1 ? word : many}`;

/** "20_ZETTLEKASTEN/kl-penalty.md" -> {title: "kl-penalty", folder: "20_ZETTLEKASTEN"} */
export function splitPath(path: string): {title: string; folder: string} {
	const slash = path.lastIndexOf('/');
	const name = path.slice(slash + 1);
	return {title: name.toLowerCase().endsWith('.md') ? name.slice(0, -3) : name, folder: slash < 0 ? '' : path.slice(0, slash)};
}
