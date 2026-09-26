import path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));

export const SHOT_DIR = path.resolve(__dirname, '../../../docs/evidence/product_v2');
