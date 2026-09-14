/**
 * Run every test file in one process.
 *
 * `node --test` spawns a child process per file, which the DSH file sandbox
 * denies (piped stdio needs a named pipe). Importing the files here runs the
 * same `node:test` suites in-process, so the suite works in both environments.
 *
 * Usage: `node test/all.test.mjs`
 */
import './parse-primitives.test.mjs';
import './protobuf.test.mjs';
import './pipeline.test.mjs';
import './analysis.test.mjs';
import './compare.test.mjs';
import './http.test.mjs';
import './web-dom.test.mjs';
import './client-bundle.test.mjs';
