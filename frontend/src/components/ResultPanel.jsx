import './ResultPanel.css';

export default function ResultPanel({ result, error }) {
  if (error) {
    return (
      <div className="result-panel result-panel--error" role="alert">
        <h2 className="result-panel__title">Error</h2>
        <p className="result-panel__message">{error}</p>
      </div>
    );
  }

  if (!result) {
    return null;
  }

  if (result.result === 'rejected') {
    return (
      <div className="result-panel result-panel--rejected" role="region" aria-label="Comparison result">
        <h2 className="result-panel__title">Comparison Rejected</h2>
        <div className="result-panel__ages">
          <div className="result-panel__age-card">
            <span className="result-panel__age-label">Image 1</span>
            <span className="result-panel__age-value">Age: {result.age1}</span>
            <span className="result-panel__age-group">{result.age_group1}</span>
          </div>
          <div className="result-panel__age-card">
            <span className="result-panel__age-label">Image 2</span>
            <span className="result-panel__age-value">Age: {result.age2}</span>
            <span className="result-panel__age-group">{result.age_group2}</span>
          </div>
        </div>
        <p className="result-panel__message">{result.message}</p>
      </div>
    );
  }

  const isSamePerson = result.result === 'same_person';

  return (
    <div
      className={`result-panel ${isSamePerson ? 'result-panel--match' : 'result-panel--no-match'}`}
      role="region"
      aria-label="Comparison result"
    >
      <h2 className="result-panel__title">
        {isSamePerson ? 'Same Person' : 'Different Person'}
      </h2>
      <div className="result-panel__ages">
        <div className="result-panel__age-card">
          <span className="result-panel__age-label">Image 1</span>
          <span className="result-panel__age-value">Age: {result.age1}</span>
          <span className="result-panel__age-group">{result.age_group1}</span>
        </div>
        <div className="result-panel__age-card">
          <span className="result-panel__age-label">Image 2</span>
          <span className="result-panel__age-value">Age: {result.age2}</span>
          <span className="result-panel__age-group">{result.age_group2}</span>
        </div>
      </div>
      <div className="result-panel__details">
        <div className="result-panel__detail">
          <span className="result-panel__detail-label">Similarity</span>
          <span className="result-panel__detail-value">
            {(result.similarity_score * 100).toFixed(1)}%
          </span>
        </div>
        <div className="result-panel__detail">
          <span className="result-panel__detail-label">Confidence</span>
          <span className="result-panel__detail-value">
            {(result.confidence * 100).toFixed(1)}%
          </span>
        </div>
        <div className="result-panel__detail">
          <span className="result-panel__detail-label">Result</span>
          <span className="result-panel__detail-value">{result.result}</span>
        </div>
      </div>
      <p className="result-panel__message">{result.message}</p>
    </div>
  );
}
