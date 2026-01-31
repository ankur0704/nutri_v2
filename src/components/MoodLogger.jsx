import React, { useState } from 'react';
import { Smile, Zap, Brain, Utensils, Save, CheckCircle } from 'lucide-react';

const MoodLogger = ({ onEntryLogged }) => {
    const [formData, setFormData] = useState({
        mood_score: 3,
        energy_level: 3,
        focus_level: 3,
        food_eaten: '',
        notes: ''
    });
    const [loading, setLoading] = useState(false);
    const [success, setSuccess] = useState(false);
    const [error, setError] = useState(null);

    const handleSubmit = async (e) => {
        e.preventDefault();
        setLoading(true);
        setError(null);
        setSuccess(false);

        try {
            const response = await fetch('http://localhost:8000/log_entry', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify(formData)
            });

            if (!response.ok) {
                const errData = await response.json();
                throw new Error(errData.detail || 'Failed to log entry');
            }

            setSuccess(true);
            setFormData({
                mood_score: 3,
                energy_level: 3,
                focus_level: 3,
                food_eaten: '',
                notes: ''
            });

            // Refresh dashboard data
            console.log('Entry logged, refreshing dashboard...');
            if (onEntryLogged) {
                await onEntryLogged();
                console.log('Dashboard refreshed!');
            }

            setTimeout(() => setSuccess(false), 3000);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const ScoreSelector = ({ label, icon: Icon, value, onChange, color }) => (
        <div className="score-selector">
            <div className="score-label">
                <Icon size={18} style={{ color }} />
                <span>{label}</span>
            </div>
            <div className="score-buttons">
                {[1, 2, 3, 4, 5].map(score => (
                    <button
                        key={score}
                        type="button"
                        className={`score-btn ${value === score ? 'active' : ''}`}
                        style={{ '--active-color': color }}
                        onClick={() => onChange(score)}
                    >
                        {score}
                    </button>
                ))}
            </div>
        </div>
    );

    return (
        <div className="mood-logger card">
            <h3 className="logger-title">📝 Log Today's Wellness</h3>

            {success && (
                <div className="success-banner">
                    <CheckCircle size={18} />
                    Entry logged successfully!
                </div>
            )}

            {error && (
                <div className="error-banner">{error}</div>
            )}

            <form onSubmit={handleSubmit}>
                <ScoreSelector
                    label="Mood Level"
                    icon={Smile}
                    value={formData.mood_score}
                    onChange={(val) => setFormData(p => ({ ...p, mood_score: val }))}
                    color="#4CAF50"
                />

                <ScoreSelector
                    label="Energy Level"
                    icon={Zap}
                    value={formData.energy_level}
                    onChange={(val) => setFormData(p => ({ ...p, energy_level: val }))}
                    color="#FF9800"
                />

                <ScoreSelector
                    label="Focus Level"
                    icon={Brain}
                    value={formData.focus_level}
                    onChange={(val) => setFormData(p => ({ ...p, focus_level: val }))}
                    color="#2196F3"
                />

                <div className="form-group">
                    <label className="form-label">
                        <Utensils size={16} /> What did you eat today?
                    </label>
                    <textarea
                        className="form-textarea"
                        placeholder="e.g., Oatmeal for breakfast, salad for lunch..."
                        value={formData.food_eaten}
                        onChange={(e) => setFormData(p => ({ ...p, food_eaten: e.target.value }))}
                        rows={2}
                    />
                </div>

                <div className="form-group">
                    <label className="form-label">Notes (optional)</label>
                    <input
                        type="text"
                        className="form-input"
                        placeholder="Any additional notes..."
                        value={formData.notes}
                        onChange={(e) => setFormData(p => ({ ...p, notes: e.target.value }))}
                    />
                </div>

                <button type="submit" className="btn-primary" disabled={loading}>
                    <Save size={18} />
                    {loading ? 'Saving...' : 'Log Entry'}
                </button>
            </form>
        </div>
    );
};

export default MoodLogger;
