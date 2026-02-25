import { describe, it, expect, vi } from 'vitest';
import { render, screen, fireEvent } from '@testing-library/react';
import ImageUploader, { validateFile } from './ImageUploader';

// Helper to create a mock File
function createMockFile(name, size, type) {
  const buffer = new ArrayBuffer(size);
  return new File([buffer], name, { type });
}

describe('validateFile', () => {
  it('returns null for a valid JPEG file', () => {
    const file = createMockFile('photo.jpg', 1024, 'image/jpeg');
    expect(validateFile(file)).toBeNull();
  });

  it('returns null for a valid PNG file', () => {
    const file = createMockFile('photo.png', 1024, 'image/png');
    expect(validateFile(file)).toBeNull();
  });

  it('returns null for a valid WebP file', () => {
    const file = createMockFile('photo.webp', 1024, 'image/webp');
    expect(validateFile(file)).toBeNull();
  });

  it('returns error for an invalid file type', () => {
    const file = createMockFile('doc.pdf', 1024, 'application/pdf');
    expect(validateFile(file)).toBe('Invalid file type. Allowed: JPEG, PNG, WebP.');
  });

  it('returns error for a GIF file', () => {
    const file = createMockFile('anim.gif', 1024, 'image/gif');
    expect(validateFile(file)).toBe('Invalid file type. Allowed: JPEG, PNG, WebP.');
  });

  it('returns error when file exceeds 10 MB', () => {
    const file = createMockFile('big.jpg', 10 * 1024 * 1024 + 1, 'image/jpeg');
    expect(validateFile(file)).toBe('File size exceeds 10 MB limit.');
  });

  it('returns null for a file exactly 10 MB', () => {
    const file = createMockFile('exact.jpg', 10 * 1024 * 1024, 'image/jpeg');
    expect(validateFile(file)).toBeNull();
  });

  it('returns type error before size error for invalid type and oversized', () => {
    const file = createMockFile('big.pdf', 11 * 1024 * 1024, 'application/pdf');
    expect(validateFile(file)).toBe('Invalid file type. Allowed: JPEG, PNG, WebP.');
  });
});

describe('ImageUploader component', () => {
  it('renders with the provided label', () => {
    render(<ImageUploader label="Image 1" onImageSelect={() => {}} />);
    expect(screen.getByText('Image 1')).toBeInTheDocument();
  });

  it('shows placeholder text when no image is selected', () => {
    render(<ImageUploader label="Image 1" onImageSelect={() => {}} />);
    expect(screen.getByText(/drag & drop an image here/i)).toBeInTheDocument();
  });

  it('shows validation error for invalid file type on input change', () => {
    const onImageSelect = vi.fn();
    render(<ImageUploader label="Image 1" onImageSelect={onImageSelect} />);

    const input = document.querySelector('input[type="file"]');
    const invalidFile = createMockFile('doc.pdf', 1024, 'application/pdf');

    fireEvent.change(input, { target: { files: [invalidFile] } });

    expect(screen.getByRole('alert')).toHaveTextContent('Invalid file type. Allowed: JPEG, PNG, WebP.');
    expect(onImageSelect).toHaveBeenCalledWith(null);
  });

  it('shows validation error for oversized file on input change', () => {
    const onImageSelect = vi.fn();
    render(<ImageUploader label="Image 1" onImageSelect={onImageSelect} />);

    const input = document.querySelector('input[type="file"]');
    const bigFile = createMockFile('big.jpg', 11 * 1024 * 1024, 'image/jpeg');

    fireEvent.change(input, { target: { files: [bigFile] } });

    expect(screen.getByRole('alert')).toHaveTextContent('File size exceeds 10 MB limit.');
    expect(onImageSelect).toHaveBeenCalledWith(null);
  });

  it('calls onImageSelect with the file for a valid selection', () => {
    const onImageSelect = vi.fn();
    render(<ImageUploader label="Image 1" onImageSelect={onImageSelect} />);

    const input = document.querySelector('input[type="file"]');
    const validFile = createMockFile('photo.jpg', 1024, 'image/jpeg');

    fireEvent.change(input, { target: { files: [validFile] } });

    expect(onImageSelect).toHaveBeenCalledWith(validFile);
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });

  it('shows validation error on drop of invalid file', () => {
    const onImageSelect = vi.fn();
    render(<ImageUploader label="Image 1" onImageSelect={onImageSelect} />);

    const dropzone = screen.getByRole('button', { name: /upload image 1/i });
    const invalidFile = createMockFile('doc.txt', 1024, 'text/plain');

    fireEvent.drop(dropzone, {
      dataTransfer: { files: [invalidFile] },
    });

    expect(screen.getByRole('alert')).toHaveTextContent('Invalid file type. Allowed: JPEG, PNG, WebP.');
    expect(onImageSelect).toHaveBeenCalledWith(null);
  });

  it('calls onImageSelect on drop of valid file', () => {
    const onImageSelect = vi.fn();
    render(<ImageUploader label="Image 1" onImageSelect={onImageSelect} />);

    const dropzone = screen.getByRole('button', { name: /upload image 1/i });
    const validFile = createMockFile('photo.png', 2048, 'image/png');

    fireEvent.drop(dropzone, {
      dataTransfer: { files: [validFile] },
    });

    expect(onImageSelect).toHaveBeenCalledWith(validFile);
  });

  it('applies dragging class on dragOver and removes on dragLeave', () => {
    render(<ImageUploader label="Image 1" onImageSelect={() => {}} />);

    const dropzone = screen.getByRole('button', { name: /upload image 1/i });

    fireEvent.dragOver(dropzone);
    expect(dropzone.className).toContain('image-uploader__dropzone--dragging');

    fireEvent.dragLeave(dropzone);
    expect(dropzone.className).not.toContain('image-uploader__dropzone--dragging');
  });

  it('has accessible dropzone with keyboard support', () => {
    render(<ImageUploader label="Image 1" onImageSelect={() => {}} />);
    const dropzone = screen.getByRole('button', { name: /upload image 1/i });
    expect(dropzone).toHaveAttribute('tabindex', '0');
  });
});
