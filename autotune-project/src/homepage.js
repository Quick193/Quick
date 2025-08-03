import React, { useState, useRef, useEffect } from 'react';
import './App.css';
import Silk from './Silk';
import ShinyText from './ShinyText';

const API_BASE_URL = 'http://127.0.0.1:5001';

// Slider: min=0, max=1, step=0.01, value in [0.1, 1.0]
// 10% is left, 50% is center, 100% is right
const AutotuneSlider = ({ value, onChange }) => (
  <div className="slider-container">
    <input
      type="range"
      min="0"
      max="1"
      step="0.01"
      value={value}
      onChange={e => {
        const v = parseFloat(e.target.value);
        onChange(v < 0.1 ? 0.1 : v); // Prevent going below 0.1
      }}
      className="slider-input"
    />
    <div className="slider-labels">
      <span>10% - Subtle</span>
      <span>50% - Balanced</span>
      <span>100% - Heavy</span>
    </div>
  </div>
);

function Homepage() {
  const [isUploading, setIsUploading] = useState(false);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);
  const [processedFile, setProcessedFile] = useState(null);
  const [dragActive, setDragActive] = useState(false);
  const [progress, setProgress] = useState(0);
  const [error, setError] = useState(null);
  // Initialize at 0.5 so 50% is center
  const [autotuneStrength, setAutotuneStrength] = useState(0.5);

  const fileInputRef = useRef(null);

  useEffect(() => {
    let interval;
    if (isUploading) {
      setProgress(0);
      interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(interval);
            return 90;
          }
          return prev + 10;
        });
      }, 100);
    } else if (isProcessing) {
      setProgress(0);
      interval = setInterval(() => {
        setProgress((prev) => {
          if (prev >= 90) {
            clearInterval(interval);
            return 90;
          }
          return prev + 5;
        });
      }, 200);
    } else {
      setProgress(0);
    }
    return () => clearInterval(interval);
  }, [isUploading, isProcessing]);

  const handleDrag = (e) => {
    e.preventDefault();
    e.stopPropagation();
    if (e.type === "dragenter" || e.type === "dragover") {
      setDragActive(true);
    } else if (e.type === "dragleave") {
      setDragActive(false);
    }
  };

  const handleDrop = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setDragActive(false);

    if (e.dataTransfer.files && e.dataTransfer.files[0]) {
      handleFile(e.dataTransfer.files[0]);
    }
  };

  const handleFile = (file) => {
    if (
      file.type === 'audio/mp3' ||
      file.type === 'audio/wav' ||
      file.type === 'audio/mpeg' ||
      file.name.toLowerCase().endsWith('.mp3') ||
      file.name.toLowerCase().endsWith('.wav')
    ) {
      setUploadedFile(file);
      setError(null);
    } else {
      setError('Please upload an MP3 or WAV file only');
    }
  };

  const handleFileInput = (e) => {
    if (e.target.files && e.target.files[0]) {
      handleFile(e.target.files[0]);
    }
  };

  const processAudio = async () => {
    if (!uploadedFile) return;

    setIsProcessing(true);
    setError(null);

    const formData = new FormData();
    formData.append('file', uploadedFile);
    // Always send at least 0.1 (10%) to backend
    formData.append('strength', (autotuneStrength < 0.1 ? 0.1 : autotuneStrength).toString());

    try {
      const response = await fetch(`${API_BASE_URL}/upload`, {
        method: 'POST',
        body: formData,
        // Don't set Content-Type header - let the browser set it with boundary for FormData
      });

      if (!response.ok) {
        const errorText = await response.text();
        let errorMessage;
        try {
          const errorData = JSON.parse(errorText);
          errorMessage = errorData.error || 'Failed to process file';
        } catch {
          errorMessage = `Server error: ${response.status} ${response.statusText}`;
        }
        throw new Error(errorMessage);
      }

      const result = await response.json();
      setProgress(100);

      setProcessedFile({
        name: result.processed_name,
        fileId: result.file_id,
        originalName: result.original_name,
        strengthUsed: result.strength_used,
      });
    } catch (error) {
      console.error('Error processing audio:', error);
      setError(error.message || 'Failed to process audio. Please try again.');
    } finally {
      setIsProcessing(false);
    }
  };

  const downloadProcessedFile = async () => {
    if (!processedFile) return;

    try {
      const response = await fetch(`${API_BASE_URL}/download/${processedFile.fileId}`);

      if (!response.ok) {
        throw new Error('Failed to download file');
      }

      const blob = await response.blob();
      const url = window.URL.createObjectURL(blob);
      const link = document.createElement('a');
      link.href = url;
      link.download = processedFile.name;
      document.body.appendChild(link);
      link.click();
      document.body.removeChild(link);
      window.URL.revokeObjectURL(url);
    } catch (error) {
      console.error('Error downloading file:', error);
      setError('Failed to download file. Please try again.');
    }
  };

  const removeFile = (e) => {
    e.preventDefault();
    e.stopPropagation();
    setUploadedFile(null);
    setProcessedFile(null);
    setIsUploading(false);
    setIsProcessing(false);
    setError(null);
    if (fileInputRef.current) {
      fileInputRef.current.value = '';
    }
  };

  return (
    <div className="homepage">
      <div className="silk-background">
        <Silk
          speed={5}
          scale={1}
          color="#374151"
          noiseIntensity={1.5}
          rotation={0}
        />
      </div>
      <div className="homepage-container">
        <ShinyText 
          text="AutoTune Pro" 
          disabled={false} 
          speed={3} 
          className="homepage-title"
        />
        <p className="homepage-subtitle">
          Transform your audio with AI-powered pitch correction
        </p>

        {error && (
          <div
            style={{
              background: 'rgba(239, 68, 68, 0.1)',
              border: '1px solid rgba(239, 68, 68, 0.3)',
              borderRadius: '10px',
              padding: '1rem',
              margin: '1rem 0',
              color: '#ef4444',
              textAlign: 'center',
            }}
          >
            {error}
          </div>
        )}

        <div className="upload-section">
          <div
            className={`upload-area ${dragActive ? 'drag-active' : ''} ${uploadedFile ? 'has-file' : ''}`}
            onDragEnter={handleDrag}
            onDragLeave={handleDrag}
            onDragOver={handleDrag}
            onDrop={handleDrop}
            onClick={() => !uploadedFile && fileInputRef.current?.click()}
          >
            <input
              ref={fileInputRef}
              type="file"
              accept=".mp3,.wav,audio/mp3,audio/wav,audio/mpeg"
              onChange={handleFileInput}
              style={{ display: 'none' }}
            />

            {!uploadedFile && (
              <div className="upload-content">
                <div className="upload-icon">
                  <svg width="48" height="48" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7,10 12,15 17,10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                </div>
                <h3>Upload Audio File</h3>
                <p>Drag and drop your audio file here, or click to browse</p>
                <span className="file-types">Supports: MP3, WAV</span>
              </div>
            )}

            {uploadedFile && (
              <div className="file-info">
                <div className="file-icon">
                  <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
                    <polyline points="14,2 14,8 20,8"/>
                    <line x1="16" y1="13" x2="8" y2="13"/>
                    <line x1="16" y1="17" x2="8" y2="17"/>
                    <polyline points="10,9 9,9 8,9"/>
                  </svg>
                </div>
                <div className="file-details">
                  <h4>{uploadedFile.name}</h4>
                  <p>{(uploadedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                </div>
                <button className="remove-button" onClick={removeFile} title="Remove file">
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <line x1="18" y1="6" x2="6" y2="18"/>
                    <line x1="6" y1="6" x2="18" y2="18"/>
                  </svg>
                </button>
              </div>
            )}
          </div>

          {uploadedFile && !processedFile && (
            <>
              <div className="autotune-slider-section">
                <h3 className="autotune-slider-title">
                  Autotune Strength: {Math.round(autotuneStrength * 100)}%
                </h3>
                <AutotuneSlider
                  value={autotuneStrength}
                  onChange={setAutotuneStrength}
                />
              </div>

              <button
                className="process-button"
                onClick={processAudio}
                disabled={isProcessing}
              >
                {isProcessing ? 'Processing...' : 'AutoTune Audio'}
              </button>

              {isProcessing && (
                <div className="progress-card">
                  <div className="progress-bar">
                    <div
                      className="progress-fill"
                      style={{
                        width: `${progress}%`,
                      }}
                    ></div>
                  </div>
                  <p style={{ color: '#cbd5e1', margin: '0.5rem 0 0 0' }}>
                    Processing Audio... {progress}%
                  </p>
                </div>
              )}
            </>
          )}

          {processedFile && (
            <div className="result-section">
              <div className="result-card">
                <div className="result-header">
                  <h3>✅ Processing Complete</h3>
                  <p>
                    Your audio has been autotuned successfully!
                    <br />
                    <span style={{ color: '#60a5fa' }}>
                      Strength: {Math.round((processedFile.strengthUsed || autotuneStrength) * 100)}%
                    </span>
                  </p>
                </div>
                <div className="result-details">
                  <div className="detail-item">
                    <span className="detail-label">Original:</span>
                    <span className="detail-value">{processedFile.originalName}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Processed:</span>
                    <span className="detail-value">{processedFile.name}</span>
                  </div>
                  <div className="detail-item">
                    <span className="detail-label">Pitch Correction:</span>
                    <span className="detail-value">Applied</span>
                  </div>
                </div>
                <button className="download-button" onClick={downloadProcessedFile}>
                  <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4"/>
                    <polyline points="7,10 12,15 17,10"/>
                    <line x1="12" y1="15" x2="12" y2="3"/>
                  </svg>
                  Download Processed File
                </button>
              </div>
            </div>
          )}
        </div>

        <div className="features-section">
          <h2>How It Works</h2>
          <div className="features-grid">
            <div className="feature-item">
              <div className="feature-number">1</div>
              <h3>Upload</h3>
              <p>Upload your audio file in MP3 or WAV format</p>
            </div>
            <div className="feature-item">
              <div className="feature-number">2</div>
              <h3>Process</h3>
              <p>Our AI analyzes and corrects pitch automatically</p>
            </div>
            <div className="feature-item">
              <div className="feature-number">3</div>
              <h3>Download</h3>
              <p>Get your perfectly tuned audio file instantly</p>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}

export default Homepage;