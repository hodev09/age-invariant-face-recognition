import { describe, it, expect } from 'vitest';
import { render, screen } from '@testing-library/react';
import ResultPanel from './ResultPanel';

describe('ResultPanel', () => {
  it('renders nothing when result and error are both null', () => {
    const { container } = render(<ResultPanel result={null} error={null} />);
    expect(container.innerHTML).toBe('');
  });

  it('displays error message when error prop is set', () => {
    render(<ResultPanel result={null} error="Something went wrong" />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Error')).toBeInTheDocument();
    expect(screen.getByText('Something went wrong')).toBeInTheDocument();
  });

  it('displays error even when result is also provided', () => {
    const result = {
      age1: 25, age2: 30, age_group1: 'adult', age_group2: 'adult',
      similarity_score: 0.8, confidence: 0.7, result: 'same_person', message: 'Match',
    };
    render(<ResultPanel result={result} error="Server error" />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
    expect(screen.getByText('Server error')).toBeInTheDocument();
  });

  it('displays rejection with ages and message', () => {
    const result = {
      age1: 2, age2: 35, age_group1: 'infant', age_group2: 'adult',
      result: 'rejected',
      message: 'Cannot reliably compare infant/childhood images with adult images',
    };
    render(<ResultPanel result={result} error={null} />);

    expect(screen.getByText('Comparison Rejected')).toBeInTheDocument();
    expect(screen.getByText('Age: 2')).toBeInTheDocument();
    expect(screen.getByText('Age: 35')).toBeInTheDocument();
    expect(screen.getByText('infant')).toBeInTheDocument();
    expect(screen.getByText('adult')).toBeInTheDocument();
    expect(screen.getByText('Cannot reliably compare infant/childhood images with adult images')).toBeInTheDocument();
  });

  it('displays successful same_person comparison with all fields', () => {
    const result = {
      age1: 25, age2: 55, age_group1: 'adult', age_group2: 'senior',
      similarity_score: 0.82, confidence: 0.72, result: 'same_person',
      message: 'The faces likely belong to the same person.',
    };
    render(<ResultPanel result={result} error={null} />);

    expect(screen.getByText('Same Person')).toBeInTheDocument();
    expect(screen.getByText('Age: 25')).toBeInTheDocument();
    expect(screen.getByText('Age: 55')).toBeInTheDocument();
    expect(screen.getByText('adult')).toBeInTheDocument();
    expect(screen.getByText('senior')).toBeInTheDocument();
    expect(screen.getByText('82.0%')).toBeInTheDocument();
    expect(screen.getByText('72.0%')).toBeInTheDocument();
    expect(screen.getByText('same_person')).toBeInTheDocument();
    expect(screen.getByText('The faces likely belong to the same person.')).toBeInTheDocument();
  });

  it('displays successful different_person comparison', () => {
    const result = {
      age1: 20, age2: 30, age_group1: 'adult', age_group2: 'adult',
      similarity_score: 0.15, confidence: 0.31, result: 'different_person',
      message: 'The faces likely belong to different people.',
    };
    render(<ResultPanel result={result} error={null} />);

    expect(screen.getByText('Different Person')).toBeInTheDocument();
    expect(screen.getByText('15.0%')).toBeInTheDocument();
    expect(screen.getByText('31.0%')).toBeInTheDocument();
    expect(screen.getByText('different_person')).toBeInTheDocument();
  });

  it('uses match styling for same_person result', () => {
    const result = {
      age1: 25, age2: 30, age_group1: 'adult', age_group2: 'adult',
      similarity_score: 0.8, confidence: 0.7, result: 'same_person', message: 'Match',
    };
    render(<ResultPanel result={result} error={null} />);
    const panel = screen.getByRole('region');
    expect(panel.className).toContain('result-panel--match');
  });

  it('uses no-match styling for different_person result', () => {
    const result = {
      age1: 20, age2: 30, age_group1: 'adult', age_group2: 'adult',
      similarity_score: 0.1, confidence: 0.4, result: 'different_person', message: 'No match',
    };
    render(<ResultPanel result={result} error={null} />);
    const panel = screen.getByRole('region');
    expect(panel.className).toContain('result-panel--no-match');
  });

  it('has accessible region for comparison results', () => {
    const result = {
      age1: 25, age2: 30, age_group1: 'adult', age_group2: 'adult',
      similarity_score: 0.8, confidence: 0.7, result: 'same_person', message: 'Match',
    };
    render(<ResultPanel result={result} error={null} />);
    expect(screen.getByRole('region', { name: /comparison result/i })).toBeInTheDocument();
  });

  it('has accessible alert role for error state', () => {
    render(<ResultPanel result={null} error="Network error" />);
    expect(screen.getByRole('alert')).toBeInTheDocument();
  });
});
