import {MoveView} from '../../ui/views/MoveView.js';
import {takeFlag, tokenize} from '../args.js';
import {withFormat} from '../format-errors.js';
import type {Command} from '../registry.js';

const MOVE_USAGE = '/move @note <folder/ | new-name | new/path> [write]';
const MERGE_USAGE = '/merge @survivor @other [write]   (other may be a name links use, with no note)';

export const move: Command = {
	name: 'move',
	usage: MOVE_USAGE,
	summary: 'Move or rename a note; every link that reached it still does (a dry run first; add write)',
	async run(input, {api}) {
		const tokens = tokenize(input.args);
		const write = takeFlag(tokens, 'write', '--write');
		const [note, to, ...rest] = tokens;
		if (!note || note.kind !== 'mention' || !to || to.kind !== 'word' || rest.length > 0) {
			throw new Error(`Usage: ${MOVE_USAGE}`);
		}

		const result = await withFormat(async () => api.move({note: note.value, to: to.value, dry_run: !write}));
		return <MoveView result={result} />;
	},
};

export const merge: Command = {
	name: 'merge',
	usage: MERGE_USAGE,
	summary: 'Merge note B into A: links to B reach A, links that would point at A itself are dropped (dry run first)',
	async run(input, {api}) {
		const tokens = tokenize(input.args);
		const write = takeFlag(tokens, 'write', '--write');
		const [survivor, other, ...rest] = tokens;
		if (!survivor || !other || rest.length > 0) {
			throw new Error(`Usage: ${MERGE_USAGE}`);
		}

		const result = await withFormat(async () => api.merge({survivor: survivor.value, other: other.value, dry_run: !write}));
		return <MoveView result={result} />;
	},
};
