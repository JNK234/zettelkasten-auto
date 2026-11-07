import {Box, Text} from 'ink';
import type {VaultFixResult} from '../../rpc/schema.js';
import {formatNumber, plural, theme} from '../theme.js';
import {Body, Group} from './Section.js';
import {CheckLines, FileChange} from './WriteView.js';

/** /fix: the diff each repair would apply (dry run), or what was written. */
export function FixView({result, maxDiffs = 3}: {result: VaultFixResult; maxDiffs?: number}) {
	const preview = result.status === 'dry-run';
	const count = plural(result.files.length, 'note');
	const remaining = result.remaining.map((row) => `${formatNumber(row.count)} ${row.code}`).join(' · ');
	const withDiff = result.files.slice(0, maxDiffs);
	const rest = result.files.slice(maxDiffs);
	return (
		<Box flexDirection="column">
			<Text>
				<Text color={theme.accent}>⏺ </Text>
				<Text bold>
					{result.status === 'unchanged' ? 'Nothing to fix automatically' : preview ? `Fix preview: ${count} would change` : `Fixed ${count}`}
				</Text>
				<Text dimColor>{preview ? '  nothing written yet' : ''}</Text>
			</Text>
			<Body>
				{result.status === 'unchanged' ? null : <CheckLines result={result} />}
				{remaining ? <Text dimColor>left {preview ? 'after this fix' : 'now'}, for you: {remaining}</Text> : <Text color={theme.ok}>no violations left</Text>}
				{preview && result.files.length > 0 ? (
					<Text>
						<Text color={theme.accent}>/fix apply</Text> writes these changes
					</Text>
				) : null}
			</Body>
			{withDiff.length > 0 ? (
				<Box flexDirection="column" paddingLeft={5}>
					{withDiff.map((file) => (
						<FileChange key={file.path} file={file} maxLines={24} />
					))}
				</Box>
			) : null}
			{rest.length > 0 ? (
				<Group title={`${preview ? 'Also changes' : 'Also fixed'} (${formatNumber(rest.length)})`}>
					{rest.map((file) => (
						<Text key={file.path}>
							<Text color={theme.warn}>~ </Text>
							{file.path}
							<Text dimColor>  {file.changes.join('; ')}</Text>
						</Text>
					))}
					{preview ? <Text dimColor>/fix all shows every diff</Text> : null}
				</Group>
			) : null}
			{result.skipped.length > 0 ? (
				<Group title={`Not fixed, needs a decision (${formatNumber(result.skipped.length)})`}>
					{result.skipped.map((skip) => (
						<Text key={`${skip.path}:${skip.line}:${skip.code}`}>
							{skip.path}
							<Text dimColor>:{skip.line}  </Text>
							<Text color={theme.warn}>{skip.code}</Text>
							<Text dimColor>  {skip.reason}</Text>
						</Text>
					))}
				</Group>
			) : null}
		</Box>
	);
}
