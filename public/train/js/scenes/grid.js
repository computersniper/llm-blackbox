import { Stub } from './stub.js';

export class Grid extends Stub {
  constructor(app, R, mode) { super(app, R, 'Grid ' + (mode || '')); }
}
