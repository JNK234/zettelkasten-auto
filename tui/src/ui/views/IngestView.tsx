import {Box, Text} from 'ink';
import type {DuplicateCandidate, NoteIngestResult, WriteResult} from '../../rpc/schema.js';
import {formatMention} from '../../commands/mentions.js';
import {theme} from '../theme.js';
import {Body, Group, NotePath} from './Section.js';
import {WriteView} from './WriteView.js';

const TITLE_THRESHOLD = 0.9;
const OVERLAP_THRESHOLD = 0.8;

function Bar({value, threshold}: {value: number; threshold: number}) {
	const width = 20;
	const filled = Math.round(value * width);
	const mark = Math.round(threshold * width);
	const cells = Array.from({length: width}, (_, index) => (index < filled ? '█' : index === mark ? '┊' : '·'));
	return <Text color={value >= threshold ? theme.error : theme.ok}>{cells.join('')}</Text>;
}

function Score({label, value, threshold, detail}: {label: string; value: number | null; threshold: number; detail: string}) {
	if (value === null) {
		return (
			<Text>
				<Text dimColor>{label.padEnd(18)}</Text>
				<Text dimColor>-      too short to compare</Text>
			</Text>
		);
	}

	const duplicate = value >= threshold;
	return (
		<Text>
			<Text dimColor>{label.padEnd(18)}</Text>
			<Text bold={duplicate}>{value.toFixed(3)}</Text> <Bar value={value} threshold={threshold} />{' '}
			<Text color={duplicate ? theme.error : undefined}>{duplicate ? `duplicate (≥ ${threshold})` : `below ${threshold}`}</Text>
			<Text dimColor>  {detail}</Text>
		</Text>
	);
}

export function CandidateScores({candidate}: {candidate: DuplicateCandidate}) {
	const [mine, theirs] = candidate.titles_compared;
	return (
		<Box flexDirection="column">
			<Score label="title similarity" value={candidate.title_similarity} threshold={TITLE_THRESHOLD} detail={`"${mine ?? ''}" ~ "${theirs ?? ''}"`} />
			<Score
				label="content overlap"
				value={candidate.content_overlap}
				threshold={OVERLAP_THRESHOLD}
				detail={`${candidate.shared_shingles} of ${candidate.shingles_compared} five-word sequences shared`}
			/>
			{candidate.same_hash ? (
				<Text>
					<Text dimColor>{'same file'.padEnd(18)}</Text>
					<Text color={theme.error} bold>
						yes
					</Text>
					<Text dimColor>    identical SHA-256</Text>
				</Text>
			) : null}
		</Box>
	);
}

export function IngestView({result}: {result: NoteIngestResult}) {
	const document = result.document;
	const about = `${document.kind} · sha256 ${document.sha256.slice(0, 12)}…${document.url ? ` · ${document.url}` : ''}`;
	if (result.status === 'duplicate') {
		return (
			<Box flexDirection="column">
				<Text>
					<Text color={theme.error}>⏺ </Text>
					<Text bold color={theme.error}>
						Not ingested: {document.file} duplicates {result.duplicates.length === 1 ? 'an existing note' : `${result.duplicates.length} existing notes`}
					</Text>
					<Text dimColor>  nothing was written</Text>
				</Text>
				<Body>
					<Text>
						“{document.title}”<Text dimColor>  {about}</Text>
					</Text>
				</Body>
				{result.duplicates.map((candidate) => (
					<Box key={candidate.path} flexDirection="column" marginTop={1} paddingLeft={5}>
						<Text>
							<Text color={theme.error}>≡ </Text>
							<NotePath path={candidate.path} />
						</Text>
						<Box paddingLeft={2} flexDirection="column">
							<CandidateScores candidate={candidate} />
							<Text dimColor>
								open it: <Text color={theme.accent}>/note {formatMention(candidate.path)}</Text>
							</Text>
						</Box>
					</Box>
				))}
			</Box>
		);
	}

	const closest = result.candidates.find((candidate) => !candidate.duplicate);
	const written: WriteResult = {...result, status: result.status};
	return (
		<Box flexDirection="column">
			<WriteView result={written} maxLines={30} />
			<Group title="Document">
				<Text>
					“{document.title}”<Text dimColor>  from {document.file} · {about}</Text>
				</Text>
				{document.warnings.map((warning) => (
					<Text key={warning} color={theme.warn}>
						note: {warning}
					</Text>
				))}
				{closest ? (
					<Text dimColor>
						closest existing note: {closest.path} (title {closest.title_similarity.toFixed(3)}, content overlap{' '}
						{closest.content_overlap === null ? '-' : closest.content_overlap.toFixed(3)}): not a duplicate
					</Text>
				) : null}
			</Group>
		</Box>
	);
}
