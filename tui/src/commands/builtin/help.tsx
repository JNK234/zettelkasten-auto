import {HelpView} from '../../ui/views/HelpView.js';
import type {Command} from '../registry.js';

export const help: Command = {
	name: 'help',
	aliases: ['h'],
	summary: 'List commands and keys',
	async run(_input, {registry}) {
		return <HelpView commands={registry.list()} />;
	},
};
