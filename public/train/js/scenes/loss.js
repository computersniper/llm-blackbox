import { Stub } from './stub.js';

export class LossView extends Stub {
  constructor(app, R, mode) { super(app, R, 'LossView ' + (mode || '')); }
}
