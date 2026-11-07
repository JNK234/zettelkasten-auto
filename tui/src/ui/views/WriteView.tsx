import {Box, Text} from 'ink';
import type {WriteResult, WrittenFile} from '../../rpc/schema.js';
import {formatNumber, plural, splitPath, theme} from '../theme.js';
import {DiffBlock, FileContent} from './Diff.js';
import {Body, Group, Headline, NotePath} from './Section.js';
import {groupByNote, ViolationsByNote} from './Violations.js';

const WHAT: Record<WriteResult['operation'], string> = {
	new: 'new note',
	link: 'link',
	set: 'change',
	fix: 'fix',
	ingest: 'ingest',
};

function headline(result: WriteResult): {title: string; detail?: string} {
	const created = result.files.find((file) => file.action === 'create');
	const count = plural(result.files.length, 'note');
	switch (result.status) {
		case 'refused': {
			return {title: `Refused: the ${WHAT[result.operation]} would add ${plural(result.new_violations.length, 'violation')}`, detail: 'nothing was written'};
		}

		case 'dry-run': {
			return {title: `Dry run: would write ${count}`, detail: 'nothing was written'};
		}

		case 'unchanged': {
			return {title: 'Nothing to write', detail: 'already so'};
		}

		default: {
			return created && result.files.length === 1
				? {title: `Created ${splitPath(created.path).title}`, detail: splitPath(created.path).folder}
				: {title: `Wrote ${count}`, detail: result.operation};
		}
	}
}

export function FileChange({file, maxLines}: {file: WrittenFile; maxLines?: number}) {
	return (
		<Box flexDirection="column" marginTop={1}>
			<Text>
				<Text color={file.action === 'create' ? theme.ok : theme.warn}>{file.action === 'create' ? '+ ' : '~ '}</Text>
				<NotePath path={file.path} />
			</Text>
			<Box flexDirection="column" paddingLeft={2}>
				{file.changes.map((change) => (
					<Text key={change} dimColor>
						{change}
					</Text>
				))}
				{file.action === 'create' && file.content !== null ? (
					<FileContent content={file.content} maxLines={maxLines} />
				) : (
					<DiffBlock diff={file.diff} maxLines={maxLines} />
				)}
			</Box>
		</Box>
	);
}

export function CheckLines({result}: {result: WriteResult}) {
	const tolerated = new Map<string, number>();
	for (const violation of result.tolerated) {
		tolerated.set(violation.code, (tolerated.get(violation.code) ?? 0) + 1);
	}

	const index = result.index;
	return (
		<>
			{result.status === 'refused' ? null : (
				<Text>
					<Text color={theme.ok}>✓</Text> check: no new violations
					{tolerated.size > 0 ? (
						<Text dimColor>
							{'  '}(already there, not blocking: {[...tolerated].map(([code, n]) => `${formatNumber(n)} ${code}`).join(', ')})
						</Text>
					) : null}
				</Text>
			)}
			{index?.status === 'updated' ? (
				<Text>
					<Text color={theme.ok}>✓</Text> index: updated {plural(index.synced.length, 'note')}
					{index.pending.length > 0 ? <Text color={theme.warn}>; left pending: {index.pending.join(', ')}</Text> : null}
				</Text>
			) : null}
			{index?.status === 'no-index' ? <Text dimColor>index: none (run vault index to create one)</Text> : null}
			{index?.status === 'needs-rebuild' ? <Text color={theme.warn}>index not updated: it needs a full rebuild ({index.reason})</Text> : null}
			{result.already.map((line) => (
				<Text key={line} dimColor>
					= {line}
				</Text>
			))}
			{result.recovered.length > 0 ? <Text color={theme.warn}>recovered an interrupted write: restored {result.recovered.join(', ')}</Text> : null}
		</>
	);
}

export function WriteView({result, maxFiles = 6, maxLines = 40}: {result: WriteResult; maxFiles?: number; maxLines?: number}) {
	const {title, detail} = headline(result);
	const refused = result.status === 'refused';
	const files = result.files.slice(0, maxFiles);
	return (
		<Box flexDirection="column">
			<Text>
				<Text color={refused ? theme.error : theme.accent}>⏺ </Text>
				<Text bold color={refused ? theme.error : undefined}>
					{title}
				</Text>
				{detail ? <Text dimColor>  {detail}</Text> : null}
			</Text>
			<Body>
				<CheckLines result={result} />
				{result.status === 'dry-run' ? <Text dimColor>run it again without --dry-run to write</Text> : null}
			</Body>
			{refused ? (
				<Group title="Violations it would add">
					<ViolationsByNote notes={groupByNote(result.new_violations)} />
				</Group>
			) : null}
			{files.length > 0 ? (
				<Box flexDirection="column" paddingLeft={5}>
					{refused ? (
						<Box marginTop={1}>
							<Text bold>The refused change</Text>
						</Box>
					) : null}
					{files.map((file) => (
						<FileChange key={file.path} file={file} maxLines={maxLines} />
					))}
					{result.files.length > files.length ? (
						<Box marginTop={1}>
							<Text dimColor>… {plural(result.files.length - files.length, 'more note')}</Text>
						</Box>
					) : null}
				</Box>
			) : null}
		</Box>
	);
}

