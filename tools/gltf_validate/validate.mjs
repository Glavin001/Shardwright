#!/usr/bin/env node
// Validate a .glb/.gltf file with the Khronos glTF Validator.
// Usage: node validate.mjs <file.glb>
// Prints the JSON validation report to stdout; exits 1 when the report has
// errors (issues.numErrors > 0), 2 on usage / IO failure.
import { readFile } from 'node:fs/promises';
import path from 'node:path';
import validator from 'gltf-validator';

const file = process.argv[2];
if (!file) {
  console.error('usage: node validate.mjs <file.glb|file.gltf>');
  process.exit(2);
}

try {
  const bytes = new Uint8Array(await readFile(file));
  const dir = path.dirname(path.resolve(file));
  const report = await validator.validateBytes(bytes, {
    uri: path.basename(file),
    maxIssues: 1000,
    externalResourceFunction: async (uri) => new Uint8Array(await readFile(path.join(dir, decodeURIComponent(uri)))),
  });
  // exit only after the (possibly large) report is flushed to the pipe
  const code = report.issues.numErrors > 0 ? 1 : 0;
  process.stdout.write(JSON.stringify(report, null, 2) + '\n', () => process.exit(code));
} catch (e) {
  console.error('validation failed:', e && e.stack ? e.stack : e);
  process.exit(2);
}
