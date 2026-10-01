import { Stub } from './stub.js';

export class Loop extends Stub {
  constructor(app, R, mode) { super(app, R, 'Loop ' + (mode || '')); }
}
