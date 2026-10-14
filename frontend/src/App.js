import React, { useState, useRef, useEffect } from 'react';
import axios from 'axios';
import './App.css';

const API_BASE = process.env.REACT_APP_API_URL || 'http://localhost:5000';

function App() {
  const [activeTab, setActiveTab] = useState('m1');
  const [m1Loading, setM1Loading] = useState(false);
  const [m1Result, setM1Result] = useState(null);
  const [m1Error, setM1Error] = useState(null);

  const [m2Loading, setM2Loading] = useState(false);
  const [m2Result, setM2Result] = useState(null);
  const [m2Error, setM2Error] = useState(null);

  const [specs, setSpecs] = useState({ m1: null, m2: null });
  const [specsLoading, setSpecsLoading] = useState(false);

  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  
  // Cache the TFLite model so it only loads once per session
  const tfliteModelRef = useRef(null); 

  // Load specs on mount
  useEffect(() => {
    fetchSpecs();
  }, []);

  // Fetch model specifications
  const fetchSpecs = async () => {
    setSpecsLoading(true);
    try {
      const [m1Res, m2Res] = await Promise.all([
        axios.get(`${API_BASE}/m1/spec`),
        axios.get(`${API_BASE}/m2/spec`)
      ]);
      setSpecs({ m1: m1Res.data, m2: m2Res.data });
    } catch (err) {
      console.error('Failed to load specs:', err);
    }
    setSpecsLoading(false);
  };

  // M1 Demo: Run inference on a test motion window
  const runM1Inference = async () => {
    setM1Loading(true);
    setM1Error(null);
    setM1Result(null);

    try {
      const testWindow = generateTestMotionWindow();
      const response = await axios.post(`${API_BASE}/m1/infer`, {
        windows: [testWindow]
      });
      setM1Result(response.data);
    } catch (err) {
      setM1Error(err.response?.data?.error || err.message);
    }
    setM1Loading(false);
  };

  // Generate synthetic test motion data for M1
  const generateTestMotionWindow = () => {
    const window = [];
    for (let i = 0; i < 400; i++) {
      const t = i / 200; 
      const isEating = Math.random() > 0.5;
      const ax = isEating ? Math.sin(t * 6) * 2 : Math.random() - 0.5;
      const ay = isEating ? Math.cos(t * 5) * 1.5 : Math.random() - 0.5;
      const az = isEating ? Math.sin(t * 4) * 1.2 : Math.random() - 0.5;
      const gx = isEating ? Math.sin(t * 8) * 50 : (Math.random() - 0.5) * 10;
      const gy = isEating ? Math.cos(t * 7) * 50 : (Math.random() - 0.5) * 10;
      const gz = isEating ? Math.sin(t * 6) * 50 : (Math.random() - 0.5) * 10;
      window.push([ax, ay, az, gx, gy, gz]);
    }
    return window;
  };

  // M2 Demo: Start camera or select file
  const startCamera = async () => {
    try {
      const stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment' }
      });
      videoRef.current.srcObject = stream;
      videoRef.current.style.display = 'block';
    } catch (err) {
      setM2Error('Camera access denied: ' + err.message);
    }
  };

  const captureAndClassify = async () => {
    if (!videoRef.current || !canvasRef.current) return;

    const context = canvasRef.current.getContext('2d');
    canvasRef.current.width = videoRef.current.videoWidth;
    canvasRef.current.height = videoRef.current.videoHeight;
    context.drawImage(videoRef.current, 0, 0);

    const imageData = canvasRef.current.toDataURL('image/jpeg');
    await classifyImage(imageData);
  };

  const handleFileSelect = async (event) => {
    const file = event.target.files[0];
    if (!file) return;

    const reader = new FileReader();
    reader.onload = async (e) => {
      await classifyImage(e.target.result);
    };
    reader.readAsDataURL(file);
  };

  // REWRITTEN: Client-side classification logic using TensorFlow.js
  const classifyImage = async (base64Image) => {
    setM2Loading(true);
    setM2Error(null);
    setM2Result(null);

    try {
      const tf = window.tf;
      const tflite = window.tflite;

      if (!tf || !tflite) {
        throw new Error("TensorFlow.js not found. Check index.html scripts.");
      }
      if (!specs.m2) {
        throw new Error("M2 specifications not loaded yet.");
      }

      // Explicitly set the WASM path to the stable alpha.8 CDN
      tflite.setWasmPath('https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite@0.0.1-alpha.8/dist/');

      // 1. Load the model (caches in memory after first download)
      if (!tfliteModelRef.current) {
        console.log("Downloading TFLite model...");
        tfliteModelRef.current = await tflite.loadTFLiteModel(`${API_BASE}/m2/model`);
      }
      const model = tfliteModelRef.current;

      // 2. Create HTML Image Object from base64
      const img = new Image();
      img.src = base64Image;
      await new Promise((resolve, reject) => {
        img.onload = resolve;
        img.onerror = reject;
      });

      // 3. Preprocess Image and Predict inside tf.tidy to prevent memory leaks
      let probabilities;
      tf.tidy(() => {
        let tensor = tf.browser.fromPixels(img);
        
        // Resize to match specs (e.g., 224x224)
        const size = specs.m2.image_size;
        tensor = tf.image.resizeBilinear(tensor, [size, size]);
        
        // Add batch dimension and normalize [0, 1]
        tensor = tensor.expandDims(0).toFloat().div(255.0);

        // Run inference
        const outputTensor = model.predict(tensor);
        probabilities = Array.from(outputTensor.dataSync());
      });

      // 4. Map probabilities to species and sort highest to lowest
      const speciesList = specs.m2.species;
      const confidenceMap = {};
      speciesList.forEach((species, idx) => {
        confidenceMap[species] = probabilities[idx];
      });

      const sortedConfidenceMap = Object.fromEntries(
        Object.entries(confidenceMap).sort(([, a], [, b]) => b - a)
      );

      setM2Result({
        species: speciesList,
        confidence: sortedConfidenceMap,
        demo: false
      });

    } catch (err) {
      console.error(err);
      setM2Error(err.message || "Failed to classify image locally.");
    }
    
    setM2Loading(false);
  };

  return (
    <div className="App">
      <header className="app-header">
        <h1>🐐 Goat Collar Demo</h1>
        <p>Motion & Plant Classification System</p>
      </header>

      <div className="tab-navigation">
        <button
          className={`tab-btn ${activeTab === 'm1' ? 'active' : ''}`}
          onClick={() => setActiveTab('m1')}
        >
          M1: Motion Classifier
        </button>
        <button
          className={`tab-btn ${activeTab === 'm2' ? 'active' : ''}`}
          onClick={() => setActiveTab('m2')}
        >
          M2: Plant Classifier
        </button>
        <button
          className={`tab-btn ${activeTab === 'specs' ? 'active' : ''}`}
          onClick={() => setActiveTab('specs')}
        >
          Model Specs
        </button>
      </div>

      <main className="tab-content">
        {/* M1 DEMO TAB */}
        {activeTab === 'm1' && (
          <div className="tab-pane">
            <h2>Motion Classifier (M1)</h2>
            <p className="description">
              Detects eating behavior from 6-axis motion sensor data (accelerometer + gyroscope).
              Extracts 166 temporal and frequency-domain features from 2-second windows at 200 Hz.
            </p>

            <div className="demo-section">
              <button
                className="action-btn primary"
                onClick={runM1Inference}
                disabled={m1Loading}
              >
                {m1Loading ? '⏳ Running Inference...' : '▶ Run M1 Demo'}
              </button>
              <p className="hint">
                Click to run M1 on simulated motion data (400 samples × 6 channels)
              </p>
            </div>

            {m1Error && (
              <div className="error-box">
                <strong>Error:</strong> {m1Error}
              </div>
            )}

            {m1Result && (
              <div className="result-box">
                <h3>Inference Results</h3>
                <div className="result-grid">
                  <div className="result-item">
                    <label>Decision</label>
                    <p className={`decision ${m1Result.decisions[0].toLowerCase()}`}>
                      {m1Result.decisions[0].toUpperCase()}
                    </p>
                  </div>
                  <div className="result-item">
                    <label>P(Eating)</label>
                    <p className="confidence">{(m1Result.p_eating[0] * 100).toFixed(1)}%</p>
                  </div>
                  <div className="result-item">
                    <label>Threshold</label>
                    <p>{m1Result.threshold.toFixed(2)}</p>
                  </div>
                  <div className="result-item">
                    <label>Features Extracted</label>
                    <p>{m1Result.features_extracted}</p>
                  </div>
                </div>
                <div className="progress-bar">
                  <div
                    className="progress-fill"
                    style={{ width: `${m1Result.p_eating[0] * 100}%` }}
                  ></div>
                </div>
              </div>
            )}
          </div>
        )}

        {/* M2 DEMO TAB */}
        {activeTab === 'm2' && (
          <div className="tab-pane">
            <h2>Plant Classifier (M2)</h2>
            <p className="description">
              Identifies 5 plant species: Hibiscus, Aloe Vera, Moringa, Neem, Tulsi.
              Uses MobileNetV2 distilled CNN with INT8 quantization (33 KB TensorFlow Lite model).
            </p>

            <div className="demo-section">
              <button className="action-btn secondary" onClick={startCamera}>
                📷 Start Camera
              </button>
              <button
                className="action-btn secondary"
                onClick={() => fileInputRef.current?.click()}
              >
                📁 Upload Image
              </button>
              <input
                ref={fileInputRef}
                type="file"
                accept="image/*"
                onChange={handleFileSelect}
                style={{ display: 'none' }}
              />
            </div>

            <div className="video-section">
              <video
                ref={videoRef}
                autoPlay
                playsInline
                style={{ display: 'none', width: '100%', maxWidth: '300px' }}
              />
              <canvas ref={canvasRef} style={{ display: 'none' }} />
              {videoRef.current?.srcObject && (
                <button
                  className="action-btn primary"
                  onClick={captureAndClassify}
                  disabled={m2Loading}
                >
                  {m2Loading ? '⏳ Classifying...' : '📸 Capture & Classify'}
                </button>
              )}
            </div>

            {m2Error && (
              <div className="error-box">
                <strong>Error:</strong> {m2Error}
              </div>
            )}

            {m2Result && (
              <div className="result-box">
                <h3>Classification Results</h3>
                {m2Result.demo && (
                  <p className="demo-badge">{m2Result.message}</p>
                )}
                <div className="species-results">
                  {Object.entries(m2Result.confidence || {}).map(([species, conf]) => (
                    <div key={species} className="species-item">
                      <span className="species-name">{species}</span>
                      <div className="confidence-bar">
                        <div
                          className="confidence-fill"
                          style={{ width: `${conf * 100}%` }}
                        ></div>
                      </div>
                      <span className="confidence-text">{(conf * 100).toFixed(1)}%</span>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}

        {/* SPECS TAB */}
        {activeTab === 'specs' && (
          <div className="tab-pane">
            <h2>Model Specifications</h2>

            {specsLoading && <p>Loading specifications...</p>}

            {specs.m1 && (
              <div className="specs-card">
                <h3>M1: Motion Classifier</h3>
                <table className="specs-table">
                  <tbody>
                    <tr>
                      <td>Model</td>
                      <td>{specs.m1.model}</td>
                    </tr>
                    <tr>
                      <td>Setting</td>
                      <td>{specs.m1.setting}</td>
                    </tr>
                    <tr>
                      <td>Test PR-AUC</td>
                      <td>{(specs.m1.test_accuracy * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Test F1 Score</td>
                      <td>{(specs.m1.test_f1 * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Features</td>
                      <td>{specs.m1.features}</td>
                    </tr>
                    <tr>
                      <td>Sample Rate</td>
                      <td>{specs.m1.sample_rate_hz} Hz</td>
                    </tr>
                    <tr>
                      <td>Window Duration</td>
                      <td>{specs.m1.window_ms} ms (2 seconds)</td>
                    </tr>
                    <tr>
                      <td>Decision Threshold</td>
                      <td>{specs.m1.threshold.toFixed(3)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}

            {specs.m2 && (
              <div className="specs-card">
                <h3>M2: Plant Classifier</h3>
                <table className="specs-table">
                  <tbody>
                    <tr>
                      <td>Model</td>
                      <td>{specs.m2.model}</td>
                    </tr>
                    <tr>
                      <td>Setting</td>
                      <td>{specs.m2.setting}</td>
                    </tr>
                    <tr>
                      <td>Test Accuracy</td>
                      <td>{(specs.m2.test_accuracy * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Test Macro F1</td>
                      <td>{(specs.m2.test_macro_f1 * 100).toFixed(2)}%</td>
                    </tr>
                    <tr>
                      <td>Species Count</td>
                      <td>{specs.m2.species.length}</td>
                    </tr>
                    <tr>
                      <td>Species</td>
                      <td>{specs.m2.species.join(', ')}</td>
                    </tr>
                    <tr>
                      <td>Image Size</td>
                      <td>{specs.m2.image_size}×{specs.m2.image_size} px</td>
                    </tr>
                    <tr>
                      <td>Model Size</td>
                      <td>{specs.m2.file_mb.toFixed(3)} MB</td>
                    </tr>
                    <tr>
                      <td>Classification Threshold</td>
                      <td>{specs.m2.theta.toFixed(2)}</td>
                    </tr>
                  </tbody>
                </table>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="app-footer">
        <p>Goat Collar System Demo • Makerere University CSC 3118</p>
        <p>Backend: {API_BASE}</p>
      </footer>
    </div>
  );
}

export default App;