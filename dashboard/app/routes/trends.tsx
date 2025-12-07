import { useCsvData } from "../hooks/useCsvData";
import { DashboardCard } from "../components/DashboardCard";
import { useMemo, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from "recharts";

interface TrendData {
  alternative_id: string;
  nazwa: string;
  ensemble_score: number;
  year: number;
}

export default function Trends() {
  const [dataType, setDataType] = useState<'dzial' | 'sekcja'>('sekcja');
  
  const { data } = useCsvData<TrendData>(
    dataType === 'sekcja' ? "/data/trends_sekcja.csv" : "/data/trends_dzial.csv", 
    {
      delimiter: ";",
      dynamicTyping: (header: string | number) => {
        if (header === 'alternative_id') return false;
        return true;
      },
    }
  );

  const [visibleSections, setVisibleSections] = useState<Record<string, boolean>>({});

  // Reset visibility when data type changes
  useMemo(() => {
    setVisibleSections({});
  }, [dataType]);

  const { chartData, sections } = useMemo(() => {
    if (!data || data.length === 0) return { chartData: [], sections: [] };

    const uniqueSections = Array.from(new Set(data.map((d) => d.alternative_id))).sort();
    const sectionMap = new Map<string, string>();
    const scores = new Map<string, { sum: number; count: number }>();
    const scores2024 = new Map<string, number>();

    data.forEach(d => {
      sectionMap.set(d.alternative_id, d.nazwa);
      const current = scores.get(d.alternative_id) || { sum: 0, count: 0 };
      scores.set(d.alternative_id, {
        sum: current.sum + d.ensemble_score,
        count: current.count + 1,
      });
      if (d.year === 2024) {
        scores2024.set(d.alternative_id, d.ensemble_score);
      }
    });

    const groupedByYear: Record<string, any> = {};

    data.forEach((item) => {
      if (item.year === 2028) return;
      if (!groupedByYear[item.year]) {
        groupedByYear[item.year] = { year: item.year };
      }
      groupedByYear[item.year][item.alternative_id] = item.ensemble_score;
    });

    const chartData = Object.values(groupedByYear).sort((a, b) => a.year - b.year);
    
    const COLORS = [
      "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
      "#aec7e8", "#ffbb78", "#98df8a", "#ff9896", "#c5b0d5", "#c49c94", "#f7b6d2", "#c7c7c7", "#dbdb8d", "#9edae5",
      "#393b79", "#637939", "#8c6d31", "#843c39", "#7b4173"
    ];

    const sections = uniqueSections.map((id, index) => {
      const score = scores.get(id);
      return {
        id,
        name: sectionMap.get(id) || id,
        color: COLORS[index % COLORS.length],
        averageScore: score ? score.sum / score.count : 0,
        score2024: scores2024.get(id) || 0
      };
    });

    return { chartData, sections };
  }, [data]);

  // Initialize visibility state once sections are loaded
  useMemo(() => {
    if (sections.length > 0 && Object.keys(visibleSections).length === 0) {
      const initialVisibility: Record<string, boolean> = {};
      
      // Sort by score2024 descending and take top 3
      const top3Ids = new Set([...sections]
        .sort((a, b) => b.score2024 - a.score2024)
        .slice(0, 3)
        .map(s => s.id));

      sections.forEach(s => {
        initialVisibility[s.id] = top3Ids.has(s.id);
      });
      setVisibleSections(initialVisibility);
    }
  }, [sections]);

  const toggleSection = (id: string) => {
    setVisibleSections(prev => ({
      ...prev,
      [id]: !prev[id]
    }));
  };

  const toggleAll = (show: boolean) => {
    const newVisibility: Record<string, boolean> = {};
    sections.forEach(s => newVisibility[s.id] = show);
    setVisibleSections(newVisibility);
  };

  return (
    <div className="space-y-8">
      <div className="flex justify-between items-start">
        <div>
          <h2 className="text-2xl font-bold text-pko-navy mb-2">
            Trendy {dataType === 'sekcja' ? 'Sektorowe' : 'Działowe'} (Ensemble)
          </h2>
          <p className="text-pko-navy/70">
            Analiza zmian wyników metody Ensemble w czasie dla poszczególnych {dataType === 'sekcja' ? 'sekcji' : 'działów'}.
          </p>
        </div>
        
        <div className="bg-white p-1 rounded-lg border border-gray-200 flex gap-1">
          <button
            onClick={() => setDataType('sekcja')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              dataType === 'sekcja'
                ? 'bg-pko-navy text-white shadow-sm'
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            Sekcje
          </button>
          <button
            onClick={() => setDataType('dzial')}
            className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
              dataType === 'dzial'
                ? 'bg-pko-navy text-white shadow-sm'
                : 'text-gray-600 hover:bg-gray-50'
            }`}
          >
            Działy
          </button>
        </div>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-4 gap-6">
        <div className="lg:col-span-3">
          <DashboardCard title="Wykres Trendów">
            <div className="h-[600px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={chartData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
                  <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
                  <ReferenceArea x1={2024} x2={2027} fill="#fee2e2" fillOpacity={0.3} />
                  <ReferenceLine x={2024} stroke="#ef4444" strokeDasharray="3 3" label={{ value: 'Prognoza', position: 'insideTopRight', fill: '#ef4444' }} />
                  <XAxis dataKey="year" stroke="#1a2f3a" />
                  <YAxis stroke="#1a2f3a" domain={[0, 1]} />
                  <Tooltip 
                    contentStyle={{ 
                      backgroundColor: 'white', 
                      borderRadius: '8px', 
                      border: '1px solid #e5e7eb', 
                      boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                      maxHeight: '400px',
                      overflowY: 'auto',
                      fontSize: '12px'
                    }}
                    formatter={(value: number, name: string) => [
                      value.toFixed(4), 
                      sections.find(s => s.id === name)?.name || name
                    ]}
                  />
                  <Legend />
                  {sections.map((section) => (
                    visibleSections[section.id] && (
                      <Line
                        key={section.id}
                        type="monotone"
                        dataKey={section.id}
                        name={section.id}
                        stroke={section.color}
                        strokeWidth={2}
                        dot={{ r: 3 }}
                        activeDot={{ r: 6 }}
                        connectNulls
                      />
                    )
                  ))}
                </LineChart>
              </ResponsiveContainer>
            </div>
          </DashboardCard>
        </div>

        <div className="lg:col-span-1">
          <DashboardCard title={`Filtry ${dataType === 'sekcja' ? 'Sekcji' : 'Działów'}`}>
            <div className="space-y-4">
              <div className="flex gap-2">
                <button 
                  onClick={() => toggleAll(true)}
                  className="text-xs px-2 py-1 bg-pko-navy text-white rounded hover:bg-pko-navy/90"
                >
                  Pokaż wszystkie
                </button>
                <button 
                  onClick={() => toggleAll(false)}
                  className="text-xs px-2 py-1 bg-gray-200 text-pko-navy rounded hover:bg-gray-300"
                >
                  Ukryj wszystkie
                </button>
              </div>
              <div className="space-y-2 max-h-[500px] overflow-y-auto pr-2">
                {sections.map(section => (
                  <label key={section.id} className="flex items-start gap-2 cursor-pointer text-sm hover:bg-gray-50 p-1 rounded">
                    <input
                      type="checkbox"
                      className="mt-1 form-checkbox h-4 w-4 text-pko-navy rounded border-gray-300 focus:ring-pko-red"
                      checked={!!visibleSections[section.id]}
                      onChange={() => toggleSection(section.id)}
                    />
                    <span className="flex-1 leading-tight">
                      <span className="font-bold mr-1" style={{ color: section.color }}>{section.id}</span>
                      <span className="text-gray-600 text-xs">{section.name}</span>
                    </span>
                  </label>
                ))}
              </div>
            </div>
          </DashboardCard>
        </div>
      </div>
    </div>
  );
}
