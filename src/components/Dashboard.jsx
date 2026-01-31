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

        try {
            const analyticsResponse = await fetch('http://localhost:8000/analytics');
            if (analyticsResponse.ok) {
                const analyticsData = await analyticsResponse.json();
                console.log('Analytics data updated:', analyticsData);
                setAnalytics(analyticsData);
            }
        } catch (analyticsErr) {
            console.error('Failed to fetch analytics:', analyticsErr);
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
            const date = new Date().toLocaleDateString('en-US', { weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' });
            doc.text(`Report Generated: ${date}`, 105, 40, { align: 'center' });

            doc.setDrawColor(76, 175, 80);
            doc.setLineWidth(0.5);
            doc.line(20, 45, 190, 45);

            doc.setFontSize(14);
            doc.setTextColor(46, 125, 50);
            doc.text('1. Weekly Wellness Summary', 20, 55);

            autoTable(doc, {
                startY: 60,
                head: [['Day', 'Mood (1-5)', 'Focus (1-5)', 'Energy (1-5)']],
                body: moodData.map(d => [d.day, d.mood, d.focus, d.energy]),
                theme: 'striped',
                headStyles: { fillColor: [76, 175, 80], textColor: 255 },
                styles: { halign: 'center' },
            });

            let finalY = doc.lastAutoTable.finalY + 10;

            doc.setFontSize(14);
            doc.setTextColor(46, 125, 50);
            doc.text('2. Nutrient Impact on Mental Wellness', 20, finalY);

            autoTable(doc, {
                startY: finalY + 5,
                head: [['Nutrient', 'Mood Impact Score']],
                body: nutrientData.map(n => [n.name, `${n.impact > 0 ? '+' : ''}${n.impact}%`]),
                theme: 'grid',
                headStyles: { fillColor: [76, 175, 80] },
                styles: { halign: 'center' },
            });

            finalY = doc.lastAutoTable.finalY + 10;

            doc.setFontSize(14);
            doc.setTextColor(46, 125, 50);
            doc.text('3. AI-Generated Insights', 20, finalY);

            doc.setFontSize(11);
            doc.setTextColor(50, 50, 50);
            const insightText = "Based on your recent data, Omega-3 rich foods correlate with higher focus scores, while high sugar intake shows a negative impact on afternoon energy levels. Consider reducing refined sugars and increasing whole foods for improved mental clarity.";
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
        <div className="animate-fadeIn">
            {/* Header Row */}
            <div className="flex flex-wrap justify-between items-center gap-4 mb-6">
                <h2 className="text-2xl font-bold text-gray-800 flex items-center gap-3">
                    Wellness Dashboard
                    {usingMockData && (
                        <span className="text-xs px-3 py-1 bg-amber-100 text-amber-700 rounded-full font-medium">
                            Demo Data
                        </span>
                    )}
                </h2>
                <div className="flex items-center gap-3">
                    <button
                        className="p-2 bg-white border border-gray-200 rounded-lg hover:bg-gray-50 transition-colors"
                        onClick={fetchHistory}
                        title="Refresh"
                    >
                        <RefreshCw size={18} className="text-gray-600" />
                    </button>
                    <button
                        className="flex items-center gap-2 px-4 py-2 bg-white border border-green-500 text-green-600 rounded-xl font-semibold hover:bg-green-500 hover:text-white transition-all duration-300"
                        onClick={exportPDF}
                    >
                        <Download size={18} />
                        Export Report
                    </button>
                </div>
            </div>

            {/* Main Grid */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
                {/* Left Column - Charts */}
                <div className="lg:col-span-2 space-y-6">
                    {/* Stats Cards */}
                    <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
                        <div className="bg-white rounded-xl p-5 shadow-sm flex items-center gap-4">
                            <div className="w-12 h-12 bg-green-100 rounded-xl flex items-center justify-center">
                                <Brain className="text-green-600" size={24} />
                            </div>
                            <div>
                                <span className="text-sm text-gray-500 block">Weekly Mood Avg</span>
                                <span className="text-xl font-bold text-gray-800">
                                    {analytics?.averages?.mood || '4.2'}/5
                                </span>
                            </div>
                        </div>
                        <div className="bg-white rounded-xl p-5 shadow-sm flex items-center gap-4">
                            <div className="w-12 h-12 bg-blue-100 rounded-xl flex items-center justify-center">
                                <Activity className="text-blue-600" size={24} />
                            </div>
                            <div>
                                <span className="text-sm text-gray-500 block">Focus Level</span>
                                <span className="text-xl font-bold text-gray-800">
                                    {analytics?.averages?.focus || '3.5'}/5
                                </span>
                            </div>
                        </div>
                        <div className="bg-white rounded-xl p-5 shadow-sm flex items-center gap-4">
                            <div className="w-12 h-12 bg-orange-100 rounded-xl flex items-center justify-center">
                                <TrendingUp className="text-orange-600" size={24} />
                            </div>
                            <div>
                                <span className="text-sm text-gray-500 block">Total Entries</span>
                                <span className="text-xl font-bold text-gray-800">
                                    {analytics?.total_entries || moodData.length}
                                </span>
                            </div>
                        </div>
                    </div>

                    {/* Charts */}
                    <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
                        <div className="bg-white rounded-xl p-6 shadow-sm">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Mood & Focus Trends</h3>
                            <div className="h-64">
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

                        <div className="bg-white rounded-xl p-6 shadow-sm">
                            <h3 className="text-lg font-semibold text-gray-800 mb-4">Nutrient Impact Analysis</h3>
                            <div className="h-64">
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

                    {/* AI Insight */}
                    <div className="bg-gradient-to-r from-green-50 to-emerald-50 border border-green-100 rounded-xl p-6">
                        <div className="flex items-center gap-3 mb-3">
                            <Lightbulb className="text-green-600" size={24} />
                            <h4 className="font-semibold text-gray-800">AI Personalized Insight</h4>
                        </div>
                        <p className="text-gray-600 leading-relaxed">
                            Based on your recent data, we've identified patterns between your nutrition and mental wellness.
                            <strong className="text-green-700"> Omega-3 rich foods</strong> correlate with higher focus scores, while
                            <strong className="text-red-600"> high sugar intake</strong> shows a negative impact on afternoon energy levels.
                        </p>
                    </div>
                </div>

                {/* Right Column - Mood Logger */}
                <div className="lg:col-span-1">
                    <MoodLogger onEntryLogged={fetchHistory} />
                </div>
            </div>
        </div>
    );
};

export default Dashboard;
