import type {Command} from '../registry.js';
import {check} from './check.js';
import {coverage} from './coverage.js';
import {config, explore, kept, map} from './explore.js';
import {findings} from './findings.js';
import {fix} from './fix.js';
import {related, suggest, unlinked} from './connect.js';
import {health} from './health.js';
import {help} from './help.js';
import {idea} from './idea.js';
import {ingest} from './ingest.js';
import {link} from './link.js';
import {merge, move} from './move.js';
import {newNote} from './new.js';
import {next} from './next.js';
import {note} from './note.js';
import {page} from './page.js';
import {pipeline} from './pipeline.js';
import {search} from './search.js';
import {stats} from './stats.js';

/** Register new commands by adding them here (or calling registry.register elsewhere). */
export const builtinCommands: Command[] = [
	help,
	stats,
	health,
	note,
	search,
	related,
	suggest,
	unlinked,
	explore,
	kept,
	map,
	config,
	check,
	fix,
	newNote,
	link,
	ingest,
	coverage,
	pipeline,
	page,
	findings,
	idea,
	move,
	merge,
	next,
];
