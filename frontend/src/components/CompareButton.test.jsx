import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import CompareButton from './CompareButton';

describe('CompareButton', () => {
  it('renders "Compare" text when not loading', () => {
    render(<CompareButton disabled={false} loading={false} onClick={() => {}} />);
    expect(screen.getByRole('button', { name: 'Compare' })).toBeInTheDocument();
  });

  it('is enabled when disabled is false and loading is false', () => {
    render(<CompareButton disabled={false} loading={false} onClick={() => {}} />);
    expect(screen.getByRole('button')).not.toBeDisabled();
  });

  it('is disabled when disabled prop is true', () => {
    render(<CompareButton disabled={true} loading={false} onClick={() => {}} />);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('is disabled when loading prop is true', () => {
    render(<CompareButton disabled={false} loading={true} onClick={() => {}} />);
    expect(screen.getByRole('button')).toBeDisabled();
  });

  it('shows loading spinner when loading is true', () => {
    render(<CompareButton disabled={false} loading={true} onClick={() => {}} />);
    expect(screen.getByRole('status')).toBeInTheDocument();
    expect(screen.getByText('Comparing…')).toBeInTheDocument();
    expect(screen.queryByText('Compare')).not.toBeInTheDocument();
  });

  it('does not show spinner when loading is false', () => {
    render(<CompareButton disabled={false} loading={false} onClick={() => {}} />);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
    expect(screen.getByText('Compare')).toBeInTheDocument();
  });

  it('calls onClick when clicked and enabled', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<CompareButton disabled={false} loading={false} onClick={onClick} />);

    await user.click(screen.getByRole('button'));
    expect(onClick).toHaveBeenCalledOnce();
  });

  it('does not call onClick when disabled', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<CompareButton disabled={true} loading={false} onClick={onClick} />);

    await user.click(screen.getByRole('button'));
    expect(onClick).not.toHaveBeenCalled();
  });

  it('does not call onClick when loading', async () => {
    const user = userEvent.setup();
    const onClick = vi.fn();
    render(<CompareButton disabled={false} loading={true} onClick={onClick} />);

    await user.click(screen.getByRole('button'));
    expect(onClick).not.toHaveBeenCalled();
  });

  it('sets aria-busy when loading', () => {
    render(<CompareButton disabled={false} loading={true} onClick={() => {}} />);
    expect(screen.getByRole('button')).toHaveAttribute('aria-busy', 'true');
  });

  it('does not set aria-busy when not loading', () => {
    render(<CompareButton disabled={false} loading={false} onClick={() => {}} />);
    expect(screen.getByRole('button')).toHaveAttribute('aria-busy', 'false');
  });
});
