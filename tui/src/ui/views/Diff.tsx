import {Box, Text} from 'ink';
import {theme} from '../theme.js';

/** A unified diff, coloured: + green, - red, @@ hunks cyan, file headers dim. */
export function DiffBlock({diff, maxLines = 40}: {diff: string; maxLines?: number}) {
	const lines = diff.replace(/\n$/, '').split('\n');
	const shown = lines.slice(0, maxLines);
	return (
		<Box flexDirection="column">
			{shown.map((line, index) => {
				const key = `${index}`;
				if (line.startsWith('+++') || line.startsWith('---')) {
					return (
						<Text key={key} dimColor>
							{line}
						</Text>
					);
				}

				if (line.startsWith('@@')) {
					return (
						<Text key={key} color={theme.link}>
							{line}
						</Text>
					);
				}

				if (line.startsWith('+')) {
					return (
						<Text key={key} color={theme.ok}>
							{line}
						</Text>
					);
				}

				if (line.startsWith('-')) {
					return (
						<Text key={key} color={theme.error}>
							{line}
						</Text>
					);
				}

				return (
					<Text key={key} dimColor wrap="truncate">
						{line}
					</Text>
				);
			})}
			{lines.length > shown.length ? <Text dimColor>… {lines.length - shown.length} more diff lines</Text> : null}
		</Box>
	);
}

/** The whole text of a file that a write created, in a box. */
export function FileContent({content, maxLines = 40}: {content: string; maxLines?: number}) {
	const lines = content.replace(/\n$/, '').split('\n');
	const shown = lines.slice(0, maxLines);
	return (
		<Box flexDirection="column" borderStyle="round" borderDimColor paddingX={1}>
			{shown.map((line, index) => (
				<Text key={`${index}`}>
					{line === '' ? ' ' : line}
				</Text>
			))}
			{lines.length > shown.length ? <Text dimColor>… {lines.length - shown.length} more lines</Text> : null}
		</Box>
	);
}
