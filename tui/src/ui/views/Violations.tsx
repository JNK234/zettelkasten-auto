import {Box, Text} from 'ink';
import type {Violation} from '../../rpc/schema.js';
import {theme} from '../theme.js';
import {NotePath} from './Section.js';

/** Codes `vault fix` repairs by itself; shown in the warning colour with a /fix hint. */
export const FIXABLE = new Set(['missing-mirror', 'missing-heading']);

export function ViolationLine({violation}: {violation: Violation}) {
	const fixable = FIXABLE.has(violation.code);
	// Fixed columns, so a long message wraps under itself, not under the line number.
	return (
		<Box>
			<Box width={6} flexShrink={0}>
				<Text dimColor>L{violation.line}</Text>
			</Box>
			<Box width={22} flexShrink={0}>
				<Text color={fixable ? theme.warn : theme.error}>{violation.code}</Text>
			</Box>
			<Box flexGrow={1} flexShrink={1}>
				<Text>{violation.message}</Text>
			</Box>
		</Box>
	);
}

/** Violations grouped by note: the note, its type, then one line per violation. */
export function ViolationsByNote({
	notes,
}: {
	notes: Array<{path: string; type?: string | null; note?: string; violations: Violation[]}>;
}) {
	return (
		<Box flexDirection="column">
			{notes.map((note) => (
				<Box key={note.path} flexDirection="column" marginTop={1}>
					<Text>
						<NotePath path={note.path} />
						{note.type ? <Text dimColor>  ({note.type})</Text> : null}
						{note.note ? <Text color={theme.warn}>  {note.note}</Text> : null}
					</Text>
					<Box flexDirection="column" paddingLeft={2}>
						{note.violations.map((violation, index) => (
							<ViolationLine key={`${index}:${violation.line}:${violation.code}`} violation={violation} />
						))}
					</Box>
				</Box>
			))}
		</Box>
	);
}

/** Group a flat list of violations by note, keeping path order. */
export function groupByNote(violations: Violation[]): Array<{path: string; violations: Violation[]}> {
	const groups = new Map<string, Violation[]>();
	for (const violation of violations) {
		groups.set(violation.path, [...(groups.get(violation.path) ?? []), violation]);
	}

	return [...groups.entries()].map(([path, list]) => ({path, violations: list}));
}
