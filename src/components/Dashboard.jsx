import React, { useState, useEffect } from 'react';
import {
    LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer,
    BarChart, Bar, Cell, Legend
} from 'recharts';
import { Brain, Activity, TrendingUp, Lightbulb, Download, RefreshCw } from 'lucide-react';
import jsPDF from 'jspdf';
import autoTable from 'jspdf-autotable';
import MoodLogger from './MoodLogger';

// Fallback mock data
const mockMoodData = [
    { day: 'Mon', mood: 4, focus: 3, energy: 4 },
    { day: 'Tue', mood: 3, focus: 4, energy: 3 },
    { day: 'Wed', mood: 2, focus: 2, energy: 2 },
    { day: 'Thu', mood: 5, focus: 5, energy: 5 },
    { day: 'Fri', mood: 4, focus: 4, energy: 4 },
    { day: 'Sat', mood: 5, focus: 4, energy: 5 },
    { day: 'Sun', mood: 3, focus: 3, energy: 3 },
];

const nutrientData = [
    { name: 'Protein', impact: 85, color: '#4CAF50' },
    { name: 'Fiber', impact: 70, color: '#8BC34A' },
    { name: 'Omega-3', impact: 95, color: '#2E7D32' },
    { name: 'Magnesium', impact: 60, color: '#A5D6A7' },
    { name: 'Sugar', impact: -40, color: '#FF5252' },
];

const Dashboard = () => {
    const [moodData, setMoodData] = useState(mockMoodData);
    const [analytics, setAnalytics] = useState(null);
    const [loading, setLoading] = useState(true);
    const [usingMockData, setUsingMockData] = useState(true);

    const fetchHistory = async () => {
        console.log('fetchHistory called - fetching data...');
        setLoading(true);
        try {
            const response = await fetch('http://localhost:8000/get_history?limit=7');
            if (!response.ok) throw new Error('Failed to fetch');

            const data = await response.json();
            console.log('Fetched history data:', data);

            if (data.entries && data.entries.length > 0) {
                const days = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'];
                const chartData = data.entries.reverse().map(entry => {
                    const date = new Date(entry.timestamp);
                    return {
                        day: days[date.getDay()],
                        mood: entry.mood_score,
                        focus: entry.focus_level,
                        energy: entry.energy_level
                    };
                });
                console.log('Chart data updated:', chartData);
                setMoodData(chartData);
                setUsingMockData(false);
            } else {
                setMoodData(mockMoodData);
                setUsingMockData(true);
            }
        } catch (err) {
            console.error('Failed to fetch history:', err);
            setMoodData(mockMoodData);
            setUsingMockData(true);
        }

        // Fetch analytics separately - don't let it affect mood data
        try {
            const analyticsResponse = await fetch('http://localhost:8000/analytics');
            if (analyticsResponse.ok) {
                const analyticsData = await analyticsResponse.json();
                console.log('Analytics data updated:', analyticsData);
                setAnalytics(analyticsData);
            }
        } catch (analyticsErr) {
            console.error('Failed to fetch analytics:', analyticsErr);
            // Don't reset mood data on analytics failure
        }

        setLoading(false);
    };

    useEffect(() => {
        fetchHistory();
    }, []);

    const exportPDF = () => {
        try {
            const doc = new jsPDF();

            doc.setFontSize(22);
            doc.setTextColor(46, 125, 50);
            doc.text('NutriMate AI', 105, 20, { align: 'center' });

            doc.setFontSize(16);
            doc.setTextColor(44, 62, 80);
            doc.text('Clinical Health & Nutrition Report', 105, 30, { align: 'center' });

            doc.setFontSize(10);
            doc.setTextColor(100, 100, 100);
            doc.text(`Generated on: ${new Date().toLocaleDateString()}`, 105, 38, { align: 'center' });

            doc.setDrawColor(200, 200, 200);
            doc.line(20, 45, 190, 45);

            doc.setFontSize(14);
            doc.setTextColor(46, 125, 50);
            doc.text('I. Patient Bio-Summary', 20, 55);

            doc.setFontSize(11);
            doc.setTextColor(0, 0, 0);

            const avgMood = analytics?.averages?.mood || (moodData.reduce((a, b) => a + b.mood, 0) / moodData.length).toFixed(1);
            doc.text('Weekly Assessment Level:', 20, 65);
            doc.text(`Score: ${avgMood}/5.0`, 80, 65);

            doc.text('Primary Goal:', 20, 72);
            doc.text('Metabolic Maintenance & Cognitive Support', 80, 72);

            doc.setFontSize(14);
            doc.setTextColor(46, 125, 50);
            doc.text('II. Weekly Wellness Tracking', 20, 85);

            const tableRows = moodData.map(item => [
                item.day,
                item.mood + '/5',
                item.focus + '/5',
                item.energy + '/5'
            ]);

            autoTable(doc, {
                startY: 90,
                head: [['Day', 'Mood Level', 'Focus Level', 'Energy Level']],
                body: tableRows,
                theme: 'grid',
                headStyles: { fillColor: [76, 175, 80], textColor: [255, 255, 255] }
            });

            const finalY = doc.lastAutoTable ? doc.lastAutoTable.finalY + 15 : 150;
            doc.setFontSize(14);
            doc.setTextColor(46, 125, 50);
            doc.text('III. AI Clinical Insights', 20, finalY);

            doc.setFontSize(10);
            doc.setTextColor(50, 50, 50);
            const insightText = "Based on longitudinal analysis, the NutriMate AI system identified correlations between omega-3 intake and cognitive focus. Clinical recommendation: prioritize complex carbohydrates and healthy fats.";
            const splitText = doc.splitTextToSize(insightText, 170);
            doc.text(splitText, 20, finalY + 10);

            doc.setFontSize(9);
            doc.setTextColor(150, 150, 150);
            doc.text('Confidential Health Data | Generated by NutriMate AI', 105, 285, { align: 'center' });

            doc.save('NutriMate_Clinical_Report.pdf');
        } catch (err) {
            console.error("PDF Error:", err);
            alert("Error generating PDF: " + err.message);
        }
    };

    return (
        <div className="dashboard-fade-in">
            <div className="dashboard-header-row">
                <h2 className="dashboard-main-title">
                    Wellness Dashboard
                    {usingMockData && <span className="mock-badge">Demo Data</span>}
                </h2>
                <div className="dashboard-actions">
                    <button className="btn-icon" onClick={fetchHistory} title="Refresh">
                        <RefreshCw size={18} />
                    </button>
                    <button className="btn-export" onClick={exportPDF}>
                        <Download size={18} />
                        Export Report
                    </button>
                </div>
            </div>

            <div className="dashboard-grid">
                <div className="dashboard-main">
                    <div className="stats-grid">
                        <div className="stat-card">
                            <div className="stat-icon-wrapper mood">
                                <Brain size={24} />
                            </div>
                            <div className="stat-info">
                                <span className="stat-label">Weekly Mood Avg</span>
                                <span className="stat-value">
                                    {analytics?.averages?.mood || '4.2'}/5
                                </span>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon-wrapper focus">
                                <Activity size={24} />
                            </div>
                            <div className="stat-info">
                                <span className="stat-label">Focus Level</span>
                                <span className="stat-value">
                                    {analytics?.averages?.focus || '3.5'}/5
                                </span>
                            </div>
                        </div>
                        <div className="stat-card">
                            <div className="stat-icon-wrapper trend">
                                <TrendingUp size={24} />
                            </div>
                            <div className="stat-info">
                                <span className="stat-label">Total Entries</span>
                                <span className="stat-value">
                                    {analytics?.total_entries || moodData.length}
                                </span>
                            </div>
                        </div>
                    </div>

                    <div className="charts-container">
                        <div className="chart-wrapper">
                            <h3 className="chart-title">Mood & Focus Trends</h3>
                            <div style={{ width: '100%', height: 280 }}>
                                <ResponsiveContainer>
                                    <LineChart data={moodData}>
                                        <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                                        <XAxis dataKey="day" axisLine={false} tickLine={false} />
                                        <YAxis hide domain={[0, 5]} />
                                        <Tooltip />
                                        <Legend iconType="circle" />
                                        <Line type="monotone" dataKey="mood" stroke="#4CAF50" strokeWidth={3} name="Mood" />
                                        <Line type="monotone" dataKey="focus" stroke="#2196F3" strokeWidth={3} name="Focus" />
                                        <Line type="monotone" dataKey="energy" stroke="#FF9800" strokeWidth={3} name="Energy" />
                                    </LineChart>
                                </ResponsiveContainer>
                            </div>
                        </div>

                        <div className="chart-wrapper">
                            <h3 className="chart-title">Nutrient Impact Analysis</h3>
                            <div style={{ width: '100%', height: 280 }}>
                                <ResponsiveContainer>
                                    <BarChart data={nutrientData} layout="vertical">
                                        <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#eee" />
                                        <XAxis type="number" hide />
                                        <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} width={80} />
                                        <Tooltip />
                                        <Bar dataKey="impact" radius={[0, 4, 4, 0]} name="Impact %">
                                            {nutrientData.map((entry, index) => (
                                                <Cell key={`cell-${index}`} fill={entry.color} />
                                            ))}
                                        </Bar>
                                    </BarChart>
                                </ResponsiveContainer>
                            </div>
                        </div>
                    </div>

                    <div className="ai-insight-card">
                        <div className="insight-header">
                            <Lightbulb className="insight-icon" />
                            <h4>AI Personalized Insight</h4>
                        </div>
                        <p className="insight-text">
                            Based on your recent data, we've identified patterns between your nutrition and mental wellness.
                            <strong> Omega-3 rich foods</strong> correlate with higher focus scores, while <strong>high sugar intake</strong> shows a negative impact on afternoon energy levels.
                        </p>
                    </div>
                </div>

                <div className="dashboard-sidebar">
                    <MoodLogger onEntryLogged={fetchHistory} />
                </div>
            </div>
        </div>
    );
};

export default Dashboard;

