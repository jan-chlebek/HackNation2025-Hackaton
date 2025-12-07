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

const SECTION_RANGES: Record<string, { start: number; end: number; name: string }> = {
  'A': { start: 1, end: 3, name: 'Rolnictwo, leśnictwo, łowiectwo i rybactwo' },
  'B': { start: 5, end: 9, name: 'Górnictwo i wydobywanie' },
  'C': { start: 10, end: 33, name: 'Przetwórstwo przemysłowe' },
  'D': { start: 35, end: 35, name: 'Wytwarzanie i zaopatrywanie w energię elektryczną, gaz, parę wodną i gorącą wodę' },
  'E': { start: 36, end: 39, name: 'Dostawa wody; gospodarowanie ściekami i odpadami; rekultywacja' },
  'F': { start: 41, end: 43, name: 'Budownictwo' },
  'G': { start: 45, end: 47, name: 'Handel hurtowy i detaliczny; naprawa pojazdów samochodowych, włączając motocykle' },
  'H': { start: 49, end: 53, name: 'Transport i gospodarka magazynowa' },
  'I': { start: 55, end: 56, name: 'Działalność związana z zakwaterowaniem i usługami gastronomicznymi' },
  'J': { start: 58, end: 63, name: 'Informacja i komunikacja' },
  'K': { start: 64, end: 66, name: 'Działalność finansowa i ubezpieczeniowa' },
  'L': { start: 68, end: 68, name: 'Działalność związana z obsługą rynku nieruchomości' },
  'M': { start: 69, end: 75, name: 'Działalność profesjonalna, naukowa i techniczna' },
  'N': { start: 77, end: 82, name: 'Działalność w zakresie usług administrowania i działalność wspierająca' },
  'O': { start: 84, end: 84, name: 'Administracja publiczna i obrona narodowa; obowiązkowe zabezpieczenia społeczne' },
  'P': { start: 85, end: 85, name: 'Edukacja' },
  'Q': { start: 86, end: 88, name: 'Opieka zdrowotna i pomoc społeczna' },
  'R': { start: 90, end: 93, name: 'Działalność związana z kulturą, rozrywką i rekreacją' },
  'S': { start: 94, end: 96, name: 'Pozostała działalność usługowa' },
  'T': { start: 97, end: 98, name: 'Gospodarstwa domowe zatrudniające pracowników; gospodarstwa domowe produkujące wyroby i świadczące usługi na własne potrzeby' },
  'U': { start: 99, end: 99, name: 'Organizacje i zespoły eksterytorialne' },
};

function getSectionForDivision(divisionId: string): string | null {
  const id = parseInt(divisionId, 10);
  if (isNaN(id)) return null;
  
  for (const [section, range] of Object.entries(SECTION_RANGES)) {
    if (id >= range.start && id <= range.end) {
      return section;
    }
  }
  return null;
}

function calculateLinearRegression(points: { year: number; score: number }[]) {
  const n = points.length;
  if (n < 2) return { slope: 0, intercept: 0 };

  let sumX = 0;
  let sumY = 0;
  let sumXY = 0;
  let sumXX = 0;

  points.forEach((p) => {
    const x = p.year - 2013; // Normalize year
    const y = p.score;
    sumX += x;
    sumY += y;
    sumXY += x * y;
    sumXX += x * x;
  });

  const slope = (n * sumXY - sumX * sumY) / (n * sumXX - sumX * sumX);
  const intercept = (sumY - slope * sumX) / n;
  return { slope, intercept };
}

export default function Trends() {
  const [dataType, setDataType] = useState<'dzial' | 'sekcja'>('sekcja');
  const [searchQuery, setSearchQuery] = useState('');
  const [selectedSectionFilter, setSelectedSectionFilter] = useState<string>('');
  
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
  const [showAllGrowers, setShowAllGrowers] = useState(false);
  const [showAllDecliners, setShowAllDecliners] = useState(false);

  // Reset visibility when data type changes
  useMemo(() => {
    setVisibleSections({});
    setSearchQuery('');
    setSelectedSectionFilter('');
  }, [dataType]);

  const { chartData, sections } = useMemo(() => {
    if (!data || data.length === 0) return { chartData: [], sections: [] };

    const uniqueSections = Array.from(new Set(data.map((d) => d.alternative_id))).sort();
    const sectionMap = new Map<string, string>();
    const scores = new Map<string, { sum: number; count: number }>();
    const scores2024 = new Map<string, number>();
    const scores2013 = new Map<string, number>();
    const seriesData = new Map<string, { year: number; score: number }[]>();

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
      if (d.year === 2013) {
        scores2013.set(d.alternative_id, d.ensemble_score);
      }

      if (!seriesData.has(d.alternative_id)) {
        seriesData.set(d.alternative_id, []);
      }
      // Only include historical data for trend calculation
      if (d.year >= 2013 && d.year <= 2024) {
        seriesData.get(d.alternative_id)!.push({ year: d.year, score: d.ensemble_score });
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
      const s2024 = scores2024.get(id) || 0;
      const s2013 = scores2013.get(id) || 0;
      const { slope, intercept } = calculateLinearRegression(seriesData.get(id) || []);
      
      return {
        id,
        name: sectionMap.get(id) || id,
        color: COLORS[index % COLORS.length],
        averageScore: score ? score.sum / score.count : 0,
        score2024: s2024,
        growth: s2024 - s2013,
        trendSlope: slope,
        trendIntercept: intercept
      };
    });

    return { chartData, sections };
  }, [data]);

  const singleSelectedId = Object.keys(visibleSections).filter(k => visibleSections[k]).length === 1 
    ? Object.keys(visibleSections).find(k => visibleSections[k]) 
    : null;

  const displayData = useMemo(() => {
    if (!singleSelectedId) return chartData;
    
    const section = sections.find(s => s.id === singleSelectedId);
    if (!section) return chartData;

    return chartData.map(item => ({
      ...item,
      trend: section.trendSlope * (item.year - 2013) + section.trendIntercept
    }));
  }, [chartData, singleSelectedId, sections]);

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

  const showOnly = (ids: string[]) => {
    const newVisibility: Record<string, boolean> = {};
    sections.forEach(s => {
      newVisibility[s.id] = ids.includes(s.id);
    });
    setVisibleSections(newVisibility);
  };

  const sortedGrowers = useMemo(() => {
    return [...sections]
      .filter(s => s.trendSlope > 0)
      .sort((a, b) => b.trendSlope - a.trendSlope);
  }, [sections]);

  const sortedDecliners = useMemo(() => {
    return [...sections]
      .filter(s => s.trendSlope < 0)
      .sort((a, b) => a.trendSlope - b.trendSlope);
  }, [sections]);

  const topGrowers = showAllGrowers ? sortedGrowers : sortedGrowers.slice(0, 5);
  const topDecliners = showAllDecliners ? sortedDecliners : sortedDecliners.slice(0, 5);

  const filteredSections = useMemo(() => {
    return sections.filter(section => {
      const matchesSearch = 
        section.id.toLowerCase().includes(searchQuery.toLowerCase()) || 
        section.name.toLowerCase().includes(searchQuery.toLowerCase());
      
      if (!matchesSearch) return false;

      if (dataType === 'dzial' && selectedSectionFilter) {
        const sectionCode = getSectionForDivision(section.id);
        return sectionCode === selectedSectionFilter;
      }

      return true;
    });
  }, [sections, searchQuery, selectedSectionFilter, dataType]);

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
        <div className="lg:col-span-3 space-y-6">
          <DashboardCard title="Wykres Trendów">
            <div className="h-[600px] w-full">
              <ResponsiveContainer width="100%" height="100%">
                <LineChart data={displayData} margin={{ top: 5, right: 30, left: 20, bottom: 5 }}>
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
                      name === 'trend' ? 'Trend Liniowy' : (sections.find(s => s.id === name)?.name || name)
                    ]}
                  />
                  <Legend />
                  {singleSelectedId && (
                    <Line
                      type="monotone"
                      dataKey="trend"
                      name="Trend Liniowy"
                      stroke="#666"
                      strokeDasharray="5 5"
                      strokeWidth={2}
                      dot={false}
                      activeDot={false}
                    />
                  )}
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
            <div className="h-[600px] flex flex-col gap-4">
              <div className="space-y-2 flex-shrink-0">
                <input
                  type="text"
                  placeholder="Szukaj..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-pko-navy"
                />
                
                {dataType === 'dzial' && (
                  <select
                    value={selectedSectionFilter}
                    onChange={(e) => setSelectedSectionFilter(e.target.value)}
                    className="w-full px-3 py-2 border border-gray-300 rounded-md text-sm focus:outline-none focus:ring-2 focus:ring-pko-navy"
                  >
                    <option value="">Wszystkie sekcje</option>
                    {Object.entries(SECTION_RANGES).map(([code, info]) => (
                      <option key={code} value={code}>
                        Sekcja {code} - {info.name}
                      </option>
                    ))}
                  </select>
                )}
              </div>

              <div className="flex gap-2 flex-shrink-0">
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
              <div className="space-y-2 flex-1 overflow-y-auto pr-2 min-h-0">
                {filteredSections.map(section => (
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
                {filteredSections.length === 0 && (
                  <div className="text-sm text-gray-500 text-center py-4">
                    Brak wyników
                  </div>
                )}
              </div>
            </div>
          </DashboardCard>
        </div>
      </div>

      <div className="space-y-6">
        <DashboardCard 
          title="Największe Wzrosty (Trend Liniowy 2013-2024)"
          action={
            <div className="flex gap-2">
              <button 
                onClick={() => setShowAllGrowers(!showAllGrowers)}
                className="text-xs px-3 py-1.5 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 font-medium transition-colors"
              >
                {showAllGrowers ? "Pokaż mniej" : "Pokaż wszystkie"}
              </button>
              <button 
                onClick={() => showOnly(topGrowers.map(g => g.id))}
                className="text-xs px-3 py-1.5 bg-green-100 text-green-700 rounded-md hover:bg-green-200 font-medium transition-colors"
              >
                Pokaż na wykresie
              </button>
            </div>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {topGrowers.map((grower, index) => (
              <div 
                key={grower.id} 
                onClick={() => showOnly([grower.id])}
                className="bg-gray-50 p-4 rounded-lg border border-gray-100 cursor-pointer hover:bg-green-50 hover:border-green-200 transition-all"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-2xl font-bold text-pko-navy">#{index + 1}</span>
                  <span className="text-sm font-medium text-green-600 bg-green-50 px-2 py-1 rounded">
                    +{grower.trendSlope.toFixed(4)}/rok
                  </span>
                </div>
                <div className="text-sm font-bold text-gray-900 mb-1 truncate" title={grower.name}>
                  {grower.id} - {grower.name}
                </div>
                <div className="text-xs text-gray-500">
                  Score 2024: {grower.score2024.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </DashboardCard>

        <DashboardCard 
          title="Największe Spadki (Trend Liniowy 2013-2024)"
          action={
            <div className="flex gap-2">
              <button 
                onClick={() => setShowAllDecliners(!showAllDecliners)}
                className="text-xs px-3 py-1.5 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 font-medium transition-colors"
              >
                {showAllDecliners ? "Pokaż mniej" : "Pokaż wszystkie"}
              </button>
              <button 
                onClick={() => showOnly(topDecliners.map(d => d.id))}
                className="text-xs px-3 py-1.5 bg-red-100 text-red-700 rounded-md hover:bg-red-200 font-medium transition-colors"
              >
                Pokaż na wykresie
              </button>
            </div>
          }
        >
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-5 gap-4">
            {topDecliners.map((decliner, index) => (
              <div 
                key={decliner.id} 
                onClick={() => showOnly([decliner.id])}
                className="bg-gray-50 p-4 rounded-lg border border-gray-100 cursor-pointer hover:bg-red-50 hover:border-red-200 transition-all"
              >
                <div className="flex items-center justify-between mb-2">
                  <span className="text-2xl font-bold text-pko-navy">#{index + 1}</span>
                  <span className="text-sm font-medium text-red-600 bg-red-50 px-2 py-1 rounded">
                    {decliner.trendSlope.toFixed(4)}/rok
                  </span>
                </div>
                <div className="text-sm font-bold text-gray-900 mb-1 truncate" title={decliner.name}>
                  {decliner.id} - {decliner.name}
                </div>
                <div className="text-xs text-gray-500">
                  Score 2024: {decliner.score2024.toFixed(4)}
                </div>
              </div>
            ))}
          </div>
        </DashboardCard>
      </div>
    </div>
  );
}
