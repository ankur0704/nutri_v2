import React from 'react';
import { 
  LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, 
  BarChart, Bar, Cell, Legend 
} from 'recharts';
import { Brain, Activity, TrendingUp, Lightbulb } from 'lucide-react';

const moodData = [
  { day: 'Mon', mood: 4, focus: 3 },
  { day: 'Tue', mood: 3, focus: 4 },
  { day: 'Wed', mood: 2, focus: 2 },
  { day: 'Thu', mood: 5, focus: 5 },
  { day: 'Fri', mood: 4, focus: 4 },
  { day: 'Sat', mood: 5, focus: 4 },
  { day: 'Sun', mood: 3, focus: 3 },
];

const nutrientData = [
  { name: 'Protein', impact: 85, color: '#4CAF50' },
  { name: 'Fiber', impact: 70, color: '#8BC34A' },
  { name: 'Omega-3', impact: 95, color: '#2E7D32' },
  { name: 'Magnesium', impact: 60, color: '#A5D6A7' },
  { name: 'Sugar', impact: -40, color: '#FF5252' },
];

const Dashboard = () => {
  return (
    <div className="dashboard-fade-in">
      <div className="stats-grid">
        <div className="stat-card">
          <div className="stat-icon-wrapper mood">
            <Brain size={24} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Weekly Mood Avg</span>
            <span className="stat-value">Good (4.2/5)</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon-wrapper focus">
            <Activity size={24} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Cognitive Clarity</span>
            <span className="stat-value">High (+15%)</span>
          </div>
        </div>
        <div className="stat-card">
          <div className="stat-icon-wrapper trend">
            <TrendingUp size={24} />
          </div>
          <div className="stat-info">
            <span className="stat-label">Best performing food</span>
            <span className="stat-value">Salmon & Walnuts</span>
          </div>
        </div>
      </div>

      <div className="charts-container">
        <div className="chart-wrapper">
          <h3 className="chart-title">Mood & Focus Trends</h3>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <LineChart data={moodData}>
                <CartesianGrid strokeDasharray="3 3" vertical={false} stroke="#eee" />
                <XAxis dataKey="day" axisLine={false} tickLine={false} tick={{fill: '#666'}} />
                <YAxis hide />
                <Tooltip 
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                />
                <Legend iconType="circle" />
                <Line 
                  type="monotone" 
                  dataKey="mood" 
                  stroke="#4CAF50" 
                  strokeWidth={3} 
                  dot={{ r: 4, fill: '#4CAF50' }}
                  activeDot={{ r: 6 }}
                  name="Mood Level"
                />
                <Line 
                  type="monotone" 
                  dataKey="focus" 
                  stroke="#2196F3" 
                  strokeWidth={3} 
                  dot={{ r: 4, fill: '#2196F3' }}
                  name="Focus Level"
                />
              </LineChart>
            </ResponsiveContainer>
          </div>
        </div>

        <div className="chart-wrapper">
          <h3 className="chart-title">Nutrient-Mood Correlation</h3>
          <div style={{ width: '100%', height: 300 }}>
            <ResponsiveContainer>
              <BarChart data={nutrientData} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" horizontal={false} stroke="#eee" />
                <XAxis type="number" hide />
                <YAxis dataKey="name" type="category" axisLine={false} tickLine={false} width={80} />
                <Tooltip 
                  cursor={{fill: 'transparent'}}
                  contentStyle={{ borderRadius: '8px', border: 'none', boxShadow: '0 4px 12px rgba(0,0,0,0.1)' }}
                />
                <Bar dataKey="impact" radius={[0, 4, 4, 0]} name="Impact Score %">
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
          Based on your last 7 days, we found a strong positive correlation (0.85) between <strong>Omega-3</strong> intake and <strong>afternoon focus levels</strong>. Conversely, <strong>high sugar lunches</strong> on Wednesday were followed by a 40% drop in mood scores by 4 PM. We recommend shifting sugar-heavy snacks to natural fiber-rich fruits.
        </p>
      </div>
    </div>
  );
};

export default Dashboard;
