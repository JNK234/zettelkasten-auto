import type {Command} from '../registry.js';
import {related, suggest, unlinked} from './connect.js';
import {health} from './health.js';
import {help} from './help.js';
import {note} from './note.js';
import {search} from './search.js';
import {stats} from './stats.js';

/** Register new commands by adding them here (or calling registry.register elsewhere). */
export const builtinCommands: Command[] = [help, stats, health, note, search, related, suggest, unlinked];
