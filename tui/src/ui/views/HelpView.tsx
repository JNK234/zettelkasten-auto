import {Box, Text} from 'ink';
import type {Command} from '../../commands/registry.js';
import {theme} from '../theme.js';
import {Body, Group, Headline} from './Section.js';

const KEYS: Array<[string, string]> = [
	['↑ / ↓', 'prompt history (or move in a list)'],
	['/', 'command suggestions · Tab completes · Enter runs'],
	['@', 'pick a note · Tab/Enter inserts · Esc closes'],
	['ctrl+a / ctrl+e', 'start / end of line · ctrl+u clears'],
	['ctrl+c · ctrl+d', 'quit'],
];

export function HelpView({commands}: {commands: Command[]}) {
	const width = Math.max(...commands.map((command) => (command.usage ?? `/${command.name}`).length)) + 2;
	return (
		<Box flexDirection="column">
			<Headline title="Commands" />
			<Body>
				{commands.map((command) => (
					<Text key={command.name}>
						<Text color={theme.accent}>{(command.usage ?? `/${command.name}`).padEnd(width)}</Text>
						{command.summary}
					</Text>
				))}
			</Body>
			<Group title="Keys">
				{KEYS.map(([key, what]) => (
					<Text key={key}>
						<Text bold>{key.padEnd(18)}</Text>
						<Text dimColor>{what}</Text>
					</Text>
				))}
			</Group>
		</Box>
	);
}
