import './CompareButton.css';

export default function CompareButton({ disabled, loading, onClick }) {
  return (
    <button
      className="compare-button"
      disabled={disabled || loading}
      onClick={onClick}
      aria-busy={loading}
    >
      {loading ? (
        <span className="compare-button__spinner" role="status">
          <span className="compare-button__spinner-icon" aria-hidden="true" />
          <span className="compare-button__spinner-text">Comparing…</span>
        </span>
      ) : (
        'Compare'
      )}
    </button>
  );
}
