import { describe, it, expect } from 'vitest';
import fc from 'fast-check';
import { validateFile } from '../components/ImageUploader';

/**
 * Feature: age-invariant-face-recognition
 * Property 11: Frontend client-side file validation
 *
 * For any file selected in the frontend, if the file type is not in
 * {image/jpeg, image/png, image/webp} or the file size exceeds 10 MB,
 * the frontend SHALL display a validation error and prevent submission.
 *
 * Validates: Requirements 10.6
 */

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE = 10 * 1024 * 1024; // 10 MB

// Arbitrary for generating valid MIME types
const validTypeArb = fc.constantFrom(...ALLOWED_TYPES);

// Arbitrary for generating invalid MIME types
const invalidTypeArb = fc.constantFrom(
  'application/pdf',
  'text/plain',
  'image/gif',
  'image/bmp',
  'image/tiff',
  'video/mp4',
  'application/octet-stream',
  'text/html'
);

// Arbitrary for valid file sizes (0 to 10 MB inclusive)
const validSizeArb = fc.integer({ min: 0, max: MAX_SIZE });

// Arbitrary for oversized files (just above 10 MB up to 20 MB)
const oversizedArb = fc.integer({ min: MAX_SIZE + 1, max: 20 * 1024 * 1024 });

// Helper to create a mock File object
function createMockFile(size, type) {
  // Use a minimal buffer; File constructor accepts size via ArrayBuffer
  const buffer = new ArrayBuffer(size);
  return new File([buffer], 'test-file', { type });
}

describe('Property 11: Frontend client-side file validation', () => {
  it('valid type + valid size => no validation error', () => {
    fc.assert(
      fc.property(
        validTypeArb,
        validSizeArb,
        (type, size) => {
          const file = createMockFile(size, type);
          const result = validateFile(file);
          expect(result).toBeNull();
        }
      ),
      { numRuns: 100 }
    );
  });

  it('invalid type => validation error regardless of size', () => {
    fc.assert(
      fc.property(
        invalidTypeArb,
        fc.integer({ min: 0, max: 20 * 1024 * 1024 }),
        (type, size) => {
          const file = createMockFile(size, type);
          const result = validateFile(file);
          expect(result).not.toBeNull();
          expect(result).toBe('Invalid file type. Allowed: JPEG, PNG, WebP.');
        }
      ),
      { numRuns: 100 }
    );
  });

  it('valid type + oversized => size validation error', () => {
    fc.assert(
      fc.property(
        validTypeArb,
        oversizedArb,
        (type, size) => {
          const file = createMockFile(size, type);
          const result = validateFile(file);
          expect(result).toBe('File size exceeds 10 MB limit.');
        }
      ),
      { numRuns: 100 }
    );
  });

  it('type check takes precedence over size check for invalid type + oversized', () => {
    fc.assert(
      fc.property(
        invalidTypeArb,
        oversizedArb,
        (type, size) => {
          const file = createMockFile(size, type);
          const result = validateFile(file);
          // Type error should be returned first, not size error
          expect(result).toBe('Invalid file type. Allowed: JPEG, PNG, WebP.');
        }
      ),
      { numRuns: 100 }
    );
  });
});
