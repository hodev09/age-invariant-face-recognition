import { useState, useRef, useCallback } from 'react';
import './ImageUploader.css';

const ALLOWED_TYPES = ['image/jpeg', 'image/png', 'image/webp'];
const MAX_SIZE_BYTES = 10 * 1024 * 1024; // 10 MB

export function validateFile(file) {
  if (!ALLOWED_TYPES.includes(file.type)) {
    return 'Invalid file type. Allowed: JPEG, PNG, WebP.';
  }
  if (file.size > MAX_SIZE_BYTES) {
    return 'File size exceeds 10 MB limit.';
  }
  return null;
}

export default function ImageUploader({ label, onImageSelect }) {
  const [preview, setPreview] = useState(null);
  const [error, setError] = useState(null);
  const [isDragging, setIsDragging] = useState(false);
  const fileInputRef = useRef(null);

  const handleFile = useCallback((file) => {
    setError(null);
    setPreview(null);

    const validationError = validateFile(file);
    if (validationError) {
      setError(validationError);
      onImageSelect(null);
      return;
    }

    const reader = new FileReader();
    reader.onload = (e) => setPreview(e.target.result);
    reader.readAsDataURL(file);

    onImageSelect(file);
  }, [onImageSelect]);

  const handleInputChange = (e) => {
    const file = e.target.files?.[0];
    if (file) handleFile(file);
  };

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = (e) => {
    e.preventDefault();
    setIsDragging(false);
  };

  const handleDrop = (e) => {
    e.preventDefault();
    setIsDragging(false);
    const file = e.dataTransfer.files?.[0];
    if (file) handleFile(file);
  };

  const handleClick = () => {
    fileInputRef.current?.click();
  };

  return (
    <div className="image-uploader">
      <label className="image-uploader__label">{label}</label>
      <div
        className={`image-uploader__dropzone${isDragging ? ' image-uploader__dropzone--dragging' : ''}`}
        role="button"
        tabIndex={0}
        aria-label={`Upload ${label}`}
        onClick={handleClick}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') handleClick(); }}
        onDragOver={handleDragOver}
        onDragLeave={handleDragLeave}
        onDrop={handleDrop}
      >
        {preview ? (
          <img src={preview} alt={`Preview for ${label}`} className="image-uploader__preview" />
        ) : (
          <p className="image-uploader__placeholder">
            Drag &amp; drop an image here, or click to select
          </p>
        )}
        <input
          ref={fileInputRef}
          type="file"
          accept="image/jpeg,image/png,image/webp"
          onChange={handleInputChange}
          className="image-uploader__input"
          aria-hidden="true"
          tabIndex={-1}
        />
      </div>
      {error && (
        <p className="image-uploader__error" role="alert">{error}</p>
      )}
    </div>
  );
}
