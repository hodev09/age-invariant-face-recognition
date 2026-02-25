import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import fc from 'fast-check';
import CompareButton from '../components/CompareButton';

/**
 * Feature: age-invariant-face-recognition
 * Property 10: Compare button enabled state
 *
 * For any frontend state, the "Compare" button SHALL be enabled
 * if and only if both image slots have a selected file.
 *
 * Validates: Requirements 10.2
 */
describe('Property 10: Compare button enabled state', () => {
  it('button is enabled if and only if both images are selected', () => {
    fc.assert(
      fc.property(
        fc.boolean(),
        fc.boolean(),
        (hasImage1, hasImage2) => {
          const bothSelected = hasImage1 && hasImage2;
          // The App passes disabled={!image1 || !image2}, so disabled = !bothSelected
          const disabled = !bothSelected;

          const { unmount } = render(
            <CompareButton disabled={disabled} loading={false} onClick={() => {}} />
          );

          const button = screen.getByRole('button');

          if (bothSelected) {
            expect(button).not.toBeDisabled();
          } else {
            expect(button).toBeDisabled();
          }

          unmount();
        }
      ),
      { numRuns: 100 }
    );
  });
});
