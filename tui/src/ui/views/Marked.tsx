import {Text} from 'ink';
import type {Mark} from '../../rpc/schema.js';

/**
 * Whether to show matches in colour. Without colour (NO_COLOR, a pipe, a dumb
 * terminal, and the frame tests) a match is shown as «match», like the CLI.
 */
export const colorEnabled: boolean =
	!process.env.NO_COLOR &&
	(process.env.FORCE_COLOR ? process.env.FORCE_COLOR !== '0' : Boolean(process.stdout.isTTY && process.stdout.hasColors?.()));

/**
 * Text with marked spans. Offsets count Unicode code points, as the engine
 * (Python) does, so the slices stay right after an emoji or other astral character.
 */
export function Marked({text, marks, dim = false}: {text: string; marks: Mark[]; dim?: boolean}) {
	const chars = [...text];
	const parts = [];
	let position = 0;
	for (const [index, mark] of [...marks].sort((a, b) => a.start - b.start).entries()) {
		if (mark.start < position) {
			continue;
		}

		if (mark.start > position) {
			parts.push(
				<Text key={`t${index}`} dimColor={dim}>
					{chars.slice(position, mark.start).join('')}
				</Text>,
			);
		}

		const marked = chars.slice(mark.start, mark.end).join('');
		parts.push(
			colorEnabled ? (
				<Text key={`m${index}`} bold color="yellow" underline={mark.kind === 'fuzzy'}>
					{marked}
				</Text>
			) : (
				<Text key={`m${index}`}>«{marked}»</Text>
			),
		);
		position = mark.end;
	}

	if (position < chars.length) {
		parts.push(
			<Text key="rest" dimColor={dim}>
				{chars.slice(position).join('')}
			</Text>,
		);
	}

	return <Text>{parts}</Text>;
}
