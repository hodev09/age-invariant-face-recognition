import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import App from './App';
import axios from 'axios';

vi.mock('axios');

function createFakeFile(name = 'face.jpg', type = 'image/jpeg', size = 1024) {
  const file = new File(['x'.repeat(size)], name, { type });
  return file;
}

describe('App', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('renders the title and two upload areas', () => {
    render(<App />);
    expect(screen.getByText('Age-Invariant Face Recognition')).toBeInTheDocument();
    expect(screen.getByText('Image 1')).toBeInTheDocument();
    expect(screen.getByText('Image 2')).toBeInTheDocument();
  });

  it('renders Compare button disabled initially', () => {
    render(<App />);
    expect(screen.getByRole('button', { name: 'Compare' })).toBeDisabled();
  });

  it('enables Compare button when both images are selected', async () => {
    const user = userEvent.setup();
    render(<App />);

    const inputs = document.querySelectorAll('input[type="file"]');
    const file1 = createFakeFile('face1.jpg');
    const file2 = createFakeFile('face2.jpg');

    await user.upload(inputs[0], file1);
    await user.upload(inputs[1], file2);

    expect(screen.getByRole('button', { name: 'Compare' })).not.toBeDisabled();
  });

  it('posts FormData to /compare-faces on compare click and shows result', async () => {
    const user = userEvent.setup();
    const mockResult = {
      age1: 25,
      age2: 55,
      age_group1: 'adult',
      age_group2: 'senior',
      similarity_score: 0.72,
      confidence: 0.57,
      result: 'same_person',
      message: 'Faces likely belong to the same person.',
    };
    axios.post.mockResolvedValueOnce({ data: mockResult });

    render(<App />);

    const inputs = document.querySelectorAll('input[type="file"]');
    await user.upload(inputs[0], createFakeFile('a.jpg'));
    await user.upload(inputs[1], createFakeFile('b.jpg'));
    await user.click(screen.getByRole('button', { name: 'Compare' }));

    await waitFor(() => {
      expect(screen.getByText('Same Person')).toBeInTheDocument();
    });

    expect(axios.post).toHaveBeenCalledOnce();
    const [url, formData, config] = axios.post.mock.calls[0];
    expect(url).toBe('/compare-faces');
    expect(formData).toBeInstanceOf(FormData);
    expect(config.headers['Content-Type']).toBe('multipart/form-data');
  });

  it('displays error from API error response', async () => {
    const user = userEvent.setup();
    axios.post.mockRejectedValueOnce({
      response: { data: { error: 'No face detected in the image' } },
    });

    render(<App />);

    const inputs = document.querySelectorAll('input[type="file"]');
    await user.upload(inputs[0], createFakeFile('a.jpg'));
    await user.upload(inputs[1], createFakeFile('b.jpg'));
    await user.click(screen.getByRole('button', { name: 'Compare' }));

    await waitFor(() => {
      expect(screen.getByText('No face detected in the image')).toBeInTheDocument();
    });
  });

  it('displays network error message on connection failure', async () => {
    const user = userEvent.setup();
    axios.post.mockRejectedValueOnce(new Error('Network Error'));

    render(<App />);

    const inputs = document.querySelectorAll('input[type="file"]');
    await user.upload(inputs[0], createFakeFile('a.jpg'));
    await user.upload(inputs[1], createFakeFile('b.jpg'));
    await user.click(screen.getByRole('button', { name: 'Compare' }));

    await waitFor(() => {
      expect(screen.getByText('Connection error. Please try again.')).toBeInTheDocument();
    });
  });

  it('clears previous error on successful compare', async () => {
    const user = userEvent.setup();
    // First call fails
    axios.post.mockRejectedValueOnce({
      response: { data: { error: 'Some error' } },
    });

    render(<App />);

    const inputs = document.querySelectorAll('input[type="file"]');
    await user.upload(inputs[0], createFakeFile('a.jpg'));
    await user.upload(inputs[1], createFakeFile('b.jpg'));
    await user.click(screen.getByRole('button', { name: 'Compare' }));

    await waitFor(() => {
      expect(screen.getByText('Some error')).toBeInTheDocument();
    });

    // Second call succeeds
    axios.post.mockResolvedValueOnce({
      data: {
        age1: 30, age2: 35, age_group1: 'adult', age_group2: 'adult',
        similarity_score: 0.5, confidence: 0.23, result: 'same_person',
        message: 'Match found.',
      },
    });

    await user.click(screen.getByRole('button', { name: 'Compare' }));

    await waitFor(() => {
      expect(screen.queryByText('Some error')).not.toBeInTheDocument();
      expect(screen.getByText('Same Person')).toBeInTheDocument();
    });
  });

  it('handles API detail field in error response', async () => {
    const user = userEvent.setup();
    axios.post.mockRejectedValueOnce({
      response: { data: { detail: 'Validation failed' } },
    });

    render(<App />);

    const inputs = document.querySelectorAll('input[type="file"]');
    await user.upload(inputs[0], createFakeFile('a.jpg'));
    await user.upload(inputs[1], createFakeFile('b.jpg'));
    await user.click(screen.getByRole('button', { name: 'Compare' }));

    await waitFor(() => {
      expect(screen.getByText('Validation failed')).toBeInTheDocument();
    });
  });
});
