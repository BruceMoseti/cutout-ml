import '@testing-library/jest-dom/vitest';

/**
 * jsdom does not implement object URLs or `Element.setPointerCapture`, both of which the
 * studio uses. Stubbing them here rather than in each test keeps the components free of
 * environment checks that only exist for tests.
 */
if (!globalThis.URL.createObjectURL) {
  globalThis.URL.createObjectURL = () => 'blob:stub';
  globalThis.URL.revokeObjectURL = () => undefined;
}

if (!Element.prototype.setPointerCapture) {
  Element.prototype.setPointerCapture = () => undefined;
  Element.prototype.releasePointerCapture = () => undefined;
  Element.prototype.hasPointerCapture = () => false;
}
