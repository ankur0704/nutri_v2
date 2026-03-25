import React, { useState } from 'react';
import API_BASE from '../config';

/** Safely parse JSON from a Response. Falls back to null on failure. */
async function safeJson(response) {
  try {
    return await response.json();
  } catch {
    return null;
  }
}

const MealAdvisorForm = ({ onResult, onEmotionResult, onEmotionStart }) => {
  const [formData, setFormData] = useState({
    text: '',
    age: 25,
    goal: 'Maintain Weight',
  });
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);

  const handleChange = (e) => {
    const { name, value } = e.target;
    setFormData((prev) => ({
      ...prev,
      [name]: name === 'age' ? parseInt(value) || '' : value,
    }));
  };

  const analyzeEmotion = async (text) => {
    if (onEmotionStart) onEmotionStart();
    try {
      const response = await fetch(`${API_BASE}/analyze_emotion`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text }),
      });
      if (response.ok) {
        const data = await safeJson(response);
        if (data) onEmotionResult(data);
        else onEmotionResult(null);
      } else {
        onEmotionResult(null);
      }
    } catch (err) {
      console.error('Emotion analysis failed:', err);
      onEmotionResult(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setLoading(true);
    setError(null);
    onResult(null);
    onEmotionResult(null);

    // Kick off fast emotion analysis in parallel (non-blocking)
    analyzeEmotion(formData.text);

    try {
      const response = await fetch(`${API_BASE}/recommend_diet`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(formData),
      });

      if (!response.ok) {
        // Safely parse error body — it may not be valid JSON
        const errData = await safeJson(response);
        const detail = errData?.detail || `Request failed (${response.status})`;
        throw new Error(detail);
      }

      const data = await safeJson(response);
      if (!data) throw new Error('Server returned an invalid response. Please try again.');
      onResult(data);
    } catch (err) {
      console.error(err);
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
    <div className="bg-white rounded-2xl shadow-lg p-8 mb-6">
      <form onSubmit={handleSubmit}>
        {/* Feeling input */}
        <div className="mb-6">
          <label htmlFor="text" className="block font-semibold text-gray-700 mb-2">
            How are you feeling today?
          </label>
          <textarea
            id="text"
            name="text"
            className="w-full p-4 border border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all resize-none"
            placeholder="Describe your current state... (e.g., I feel stressed and have a lot of work)"
            rows={4}
            value={formData.text}
            onChange={handleChange}
            required
            maxLength={2000}
          />
        </div>

        {/* Age + Goal */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mb-6">
          <div>
            <label htmlFor="age" className="block font-semibold text-gray-700 mb-2">
              Age
            </label>
            <input
              type="number"
              id="age"
              name="age"
              className="w-full p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all"
              value={formData.age}
              onChange={handleChange}
              min="1"
              max="120"
              required
            />
          </div>
          <div>
            <label htmlFor="goal" className="block font-semibold text-gray-700 mb-2">
              Goal
            </label>
            <select
              id="goal"
              name="goal"
              className="w-full p-3 border border-gray-200 rounded-xl focus:ring-2 focus:ring-green-500 focus:border-transparent transition-all bg-white cursor-pointer"
              value={formData.goal}
              onChange={handleChange}
            >
              <option value="Maintain Weight">Maintain Weight</option>
              <option value="Weight Loss">Weight Loss</option>
              <option value="Weight Gain">Weight Gain</option>
            </select>
          </div>
        </div>

        <button
          type="submit"
          className="w-full py-4 bg-gradient-to-r from-green-500 to-green-600 text-white font-semibold rounded-xl shadow-lg hover:shadow-xl hover:from-green-600 hover:to-green-700 transition-all duration-300 disabled:opacity-50 disabled:cursor-not-allowed"
          disabled={loading}
        >
          {loading ? 'Analyzing...' : 'Get Diet Recommendation'}
        </button>
      </form>

      {/* Loading */}
      {loading && (
        <div className="flex flex-col items-center justify-center py-12">
          <div className="w-12 h-12 border-4 border-green-200 border-t-green-500 rounded-full animate-spin mb-4" />
          <p className="text-gray-500 font-medium">Analyzing your emotion and generating diet plan...</p>
        </div>
      )}

      {/* Error */}
      {error && (
        <div className="bg-red-50 border border-red-200 text-red-700 rounded-xl p-4 mt-4">
          {error}
        </div>
      )}
    </div>
  );
};

export default MealAdvisorForm;
