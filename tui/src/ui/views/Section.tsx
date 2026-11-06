import {Box, Text} from 'ink';
import type {ReactNode} from 'react';
import {theme} from '../theme.js';

/** Claude Code style block: "⏺ title" then "⎿" and indented body. */
export function Headline({title, detail}: {title: string; detail?: string}) {
	return (
		<Text>
			<Text color={theme.accent}>⏺ </Text>
			<Text bold>{title}</Text>
			{detail ? <Text dimColor>  {detail}</Text> : null}
		</Text>
	);
}

export function Body({children}: {children: ReactNode}) {
	return (
		<Box>
			<Text dimColor>{'  ⎿  '}</Text>
			<Box flexDirection="column">{children}</Box>
		</Box>
	);
}

export function Group({title, children}: {title: string; children: ReactNode}) {
	return (
		<Box flexDirection="column" marginTop={1} paddingLeft={5}>
			<Text bold>{title}</Text>
			<Box flexDirection="column" paddingLeft={2}>
				{children}
			</Box>
		</Box>
	);
}

export function NotePath({path}: {path: string}) {
	const slash = path.lastIndexOf('/');
	const name = path.slice(slash + 1).replace(/\.md$/i, '');
	return (
		<Text>
			<Text color={theme.link}>{name}</Text>
			{slash >= 0 ? <Text dimColor>  {path.slice(0, slash)}</Text> : null}
		</Text>
	);
}
