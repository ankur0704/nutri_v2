import { useState } from 'react';
import './App.css';

function App() {
  const [formData, setFormData] = useState({
    text: '',
    age: 25,
    goal: 'Maintain Weight'
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData(prev => ({
      ...prev,
      [name]: name === 'age' ? parseInt(value) || '' : value
    }));
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    setResult(null);

    try {
      const response = await fetch('http://127.0.0.1:8000/recommend_diet', {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.detail || 'Failed to get recommendation.');
      }

      const data = await response.json();
      setResult(data);
    } catch (err) {
      console.error(err);
      // Check if it's a fetch error (Network) or a thrown Error from response
      if (err.message.includes('Failed to fetch')) {
        setError('Network Error: Could not connect to backend (is it running on port 8000?)');
      } else {
        setError(err.message || 'An unexpected error occurred.');
      }
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="app-container">
      <header className="header">
        <h1 className="title">🥗 NutriMate AI</h1>
        <p className="subtitle">Emotion-Aware Personalized Diet Recommendation System</p>
      </header>

      <main>
        <div className="card">
          <form onSubmit={handleSubmit}>
            <div className="form-group">
              <label htmlFor="text" className="form-label">How are you feeling today?</label>
              <textarea
                id="text"
                name="text"
                className="form-textarea"
                placeholder="How are you feeling today? (e.g., I feel stressed and tired)"
                value={formData.text}
                onChange={handleChange}
                required
              />
            </div>

            <div className="form-row">
              <div className="form-col">
                <div className="form-group">
                  <label htmlFor="age" className="form-label">Age</label>
                  <input
                    type="number"
                    id="age"
                    name="age"
                    className="form-input"
                    value={formData.age}
                    onChange={handleChange}
                    min="1"
                    max="120"
                    required
                  />
                </div>
              </div>
              <div className="form-col">
                <div className="form-group">
                  <label htmlFor="goal" className="form-label">Goal</label>
                  <select
                    id="goal"
                    name="goal"
                    className="form-select"
                    value={formData.goal}
                    onChange={handleChange}
                  >
                    <option value="Maintain Weight">Maintain Weight</option>
                    <option value="Weight Loss">Weight Loss</option>
                    <option value="Weight Gain">Weight Gain</option>
                  </select>
                </div>
              </div>
            </div>

            <button type="submit" className="btn-primary" disabled={loading}>
              {loading ? 'Analyzing...' : 'Get Diet Recommendation'}
            </button>
          </form>
        </div>

        {loading && (
          <div className="loading-container">
            <div className="spinner"></div>
            <p className="loading-text">Analyzing your emotion...</p>
          </div>
        )}

        {error && (
          <div className="card" style={{ borderLeft: '4px solid #f44336', color: '#f44336' }}>
            {error}
          </div>
        )}

        {result && (
          <div className="card result-card">
            <div className="result-header">
              <h2 className="section-title">Analysis Result</h2>
              {result.emotion && (
                <div className="emotion-badge">
                  <span>Detected Emotion:</span>
                  <strong>{result.emotion}</strong>
                </div>
              )}
            </div>

            <div className="meal-grid">
              <div className="meal-card">
                <span className="meal-icon">🍳</span>
                <div className="meal-title">Breakfast</div>
                <div className="meal-content">{result.diet_plan?.breakfast || 'N/A'}</div>
              </div>
              <div className="meal-card">
                <span className="meal-icon">🥗</span>
                <div className="meal-title">Lunch</div>
                <div className="meal-content">{result.diet_plan?.lunch || 'N/A'}</div>
              </div>
              <div className="meal-card">
                <span className="meal-icon">🍽️</span>
                <div className="meal-title">Dinner</div>
                <div className="meal-content">{result.diet_plan?.dinner || 'N/A'}</div>
              </div>
            </div>

            {result.notes && (
              <div className="notes-section">
                <div className="notes-title">💡 Why this recommendation?</div>
                <p>{result.notes}</p>
              </div>
            )}
          </div>
        )}
      </main>

      <footer className="footer">
        <p>NutriMate AI – Phase 1 Prototype | Final Year Engineering Project</p>
      </footer>
    </div>
  );
}

export default App;
