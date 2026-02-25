import { useState } from 'react';
import axios from 'axios';
import ImageUploader from './components/ImageUploader';
import CompareButton from './components/CompareButton';
import ResultPanel from './components/ResultPanel';
import './App.css';

function App() {
  const [image1, setImage1] = useState(null);
  const [image2, setImage2] = useState(null);
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);

  const handleCompare = async () => {
    setLoading(true);
    setResult(null);
    setError(null);

    const formData = new FormData();
    formData.append('image1', image1);
    formData.append('image2', image2);

    try {
      const response = await axios.post('/compare-faces', formData, {
        headers: { 'Content-Type': 'multipart/form-data' },
      });
      setResult(response.data);
    } catch (err) {
      if (err.response && err.response.data) {
        const data = err.response.data;
        setError(data.error || data.detail || 'An unexpected error occurred.');
      } else {
        setError('Connection error. Please try again.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app">
      <h1 className="app__title">Age-Invariant Face Recognition</h1>
      <p className="app__subtitle">
        Upload two face images to determine if they belong to the same person,
        even across different ages.
      </p>
      <div className="app__uploaders">
        <ImageUploader label="Image 1" onImageSelect={setImage1} />
        <ImageUploader label="Image 2" onImageSelect={setImage2} />
      </div>
      <CompareButton
        disabled={!image1 || !image2}
        loading={loading}
        onClick={handleCompare}
      />
      <ResultPanel result={result} error={error} />
    </div>
  );
}

export default App;
