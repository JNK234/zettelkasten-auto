import {Box, Text} from 'ink';
import type {VaultCheckResult} from '../../rpc/schema.js';
import {formatNumber, plural, theme} from '../theme.js';
import {Body, Group, Headline, NotePath} from './Section.js';
import {FIXABLE, ViolationsByNote} from './Violations.js';

export function CheckView({data}: {data: VaultCheckResult}) {
	const {scope, summary} = data;
	const fixable = summary.by_code.filter((row) => FIXABLE.has(row.code)).reduce((total, row) => total + row.count, 0);
	const scopeText =
		scope.mode === 'all'
			? `${data.format.name} · all ${plural(scope.checked, 'note')}`
			: scope.reason
				? `${data.format.name} · all ${plural(scope.checked, 'note')} (index needs a rebuild)`
				: `${data.format.name} · ${plural(scope.checked, 'note')} changed since the last index`;
	const hidden = summary.notes - data.notes.length;
	return (
		<Box flexDirection="column">
			<Headline title="Format check" detail={scopeText} />
			<Body>
				{data.ok ? (
					<Text>
						<Text color={theme.ok}>✓</Text> no violations in {plural(scope.checked, 'note')} checked
					</Text>
				) : (
					<Text>
						<Text color={theme.error}>✗</Text> {plural(summary.violations, 'violation')} in {plural(summary.notes, 'note')}
					</Text>
				)}
				{summary.by_code.length > 0 ? (
					<Text dimColor>{summary.by_code.map((row) => `${formatNumber(row.count)} ${row.code}`).join(' · ')}</Text>
				) : null}
				{fixable > 0 ? (
					<Text>
						<Text color={theme.warn}>{formatNumber(fixable)}</Text> can be repaired automatically: <Text color={theme.accent}>/fix</Text>{' '}
						<Text dimColor>shows the diff first</Text>
					</Text>
				) : null}
				{scope.mode === 'changed' && (scope.deleted.length > 0 || (scope.notes?.length ?? 0) > 0) ? (
					<Text dimColor>
						{[...(scope.notes ?? []).map((path) => `~ ${path}`), ...scope.deleted.map((path) => `- ${path}`)].join('   ')}
					</Text>
				) : null}
			</Body>
			{data.notes.length > 0 ? (
				<Box flexDirection="column" paddingLeft={5}>
					<ViolationsByNote
						notes={data.notes.map((note) => ({
							...note,
							note: note.checked || scope.mode === 'all' ? undefined : 'not changed: affected by a change',
						}))}
					/>
					{hidden > 0 ? (
						<Box marginTop={1}>
							<Text dimColor>
								… {plural(hidden, 'more note')} with violations · <Text color={theme.accent}>/check all</Text> lists every note
							</Text>
						</Box>
					) : null}
				</Box>
			) : null}
			{scope.mode === 'changed' && scope.deleted.length > 0 ? (
				<Group title="Deleted since the last index">
					{scope.deleted.map((path) => (
						<NotePath key={path} path={path} />
					))}
				</Group>
			) : null}
		</Box>
	);
}
