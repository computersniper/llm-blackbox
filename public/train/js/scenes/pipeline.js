import { Stub } from './stub.js';

export class Pipeline extends Stub {
  constructor(app, R, mode) { super(app, R, 'Pipeline ' + (mode || '')); }
}
