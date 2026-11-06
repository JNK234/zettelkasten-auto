import {NoteView} from '../../ui/views/NoteView.js';
import type {Command} from '../registry.js';

export const note: Command = {
	name: 'note',
	aliases: ['n'],
	usage: '/note @note | /note <search>',
	summary: 'Properties, outline, links out and backlinks of a note',
	async run(input, {api}) {
		const mention = input.mentions[0];
		if (mention) {
			return <NoteView note={await api.note(mention.path)} />;
		}

		if (!input.text) {
			throw new Error('Usage: /note @note (type @ to pick a note) or /note <search>');
		}

		const best = (await api.pick({query: input.text, limit: 1})).results[0];
		if (!best) {
			throw new Error(`No note matches "${input.text}"`);
		}

		return <NoteView note={await api.note(best.path)} matchedFrom={input.text} />;
	},
};
