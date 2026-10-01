import { Stub } from './stub.js';

export class Tokens extends Stub {
  constructor(app, R, mode) { super(app, R, 'Tokens ' + (mode || '')); }
}
