import type {Command} from '../registry.js';
import {check} from './check.js';
import {coverage} from './coverage.js';
import {fix} from './fix.js';
import {health} from './health.js';
import {help} from './help.js';
import {ingest} from './ingest.js';
import {link} from './link.js';
import {newNote} from './new.js';
import {note} from './note.js';
import {stats} from './stats.js';

/** Register new commands by adding them here (or calling registry.register elsewhere). */
export const builtinCommands: Command[] = [help, stats, health, note, check, fix, newNote, link, ingest, coverage];
