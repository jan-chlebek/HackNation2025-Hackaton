import { useMemo, useState } from "react";
import { useCsvData } from "../hooks/useCsvData";
import { DashboardCard } from "../components/DashboardCard";
import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell
} from "recharts";

interface ResultData {
  alternative_id: string;
  topsis_rank: number;
  vikor_rank: number;
  [key: string]: string | number | undefined;
}

const DATASETS = [
  { id: 'results', label: 'Ogólne', color: '#1a2f3a' },
  { id: 'results-credit', label: 'Kredytowe', color: '#d93026' },
  { id: 'results-development', label: 'Rozwojowe', color: '#c9a961' },
  { id: 'results-effectivity', label: 'Efektywność', color: '#16a34a' },
];

const SECTION_MAPPING: Record<string, { label: string, range: [number, number][] }> = {
  'A': { label: 'Rolnictwo, leśnictwo, łowiectwo i rybactwo', range: [[1, 3]] },
  'B': { label: 'Górnictwo i wydobywanie', range: [[5, 9]] },
  'C': { label: 'Przetwórstwo przemysłowe', range: [[10, 33]] },
  'D': { label: 'Wytwarzanie i zaopatrywanie w energię elektryczną, gaz, parę wodną i gorącą wodę', range: [[35, 35]] },
  'E': { label: 'Dostawa wody; gospodarowanie ściekami i odpadami oraz działalność związana z rekultywacją', range: [[36, 39]] },
  'F': { label: 'Budownictwo', range: [[41, 43]] },
  'G': { label: 'Handel hurtowy i detaliczny; naprawa pojazdów samochodowych, włączając motocykle', range: [[45, 47]] },
  'H': { label: 'Transport i gospodarka magazynowa', range: [[49, 53]] },
  'I': { label: 'Działalność związana z zakwaterowaniem i usługami gastronomicznymi', range: [[55, 56]] },
  'J': { label: 'Informacja i komunikacja', range: [[58, 63]] },
  'K': { label: 'Działalność finansowa i ubezpieczeniowa', range: [[64, 66]] },
  'L': { label: 'Działalność związana z obsługą rynku nieruchomości', range: [[68, 68]] },
  'M': { label: 'Działalność profesjonalna, naukowa i techniczna', range: [[69, 75]] },
  'N': { label: 'Działalność w zakresie usług administrowania i działalność wspierająca', range: [[77, 82]] },
  'O': { label: 'Administracja publiczna i obrona narodowa; obowiązkowe zabezpieczenia społeczne', range: [[84, 84]] },
  'P': { label: 'Edukacja', range: [[85, 85]] },
  'Q': { label: 'Opieka zdrowotna i pomoc społeczna', range: [[86, 88]] },
  'R': { label: 'Działalność związana z kulturą, rozrywką i rekreacją', range: [[90, 93]] },
  'S': { label: 'Pozostała działalność usługowa', range: [[94, 96]] },
  'T': { label: 'Gospodarstwa domowe zatrudniające pracowników; gospodarstwa domowe produkujące wyroby i świadczące usługi na własne potrzeby', range: [[97, 98]] },
  'U': { label: 'Organizacje i zespoły eksterytorialne', range: [[99, 99]] },
};

function getSectionForDivision(divisionId: string): string | undefined {
  const id = parseInt(divisionId, 10);
  if (isNaN(id)) return undefined;
  
  for (const [section, { range }] of Object.entries(SECTION_MAPPING)) {
    for (const [start, end] of range) {
      if (id >= start && id <= end) return section;
    }
  }
  return undefined;
}

function calculateSpearmanCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n !== y.length || n === 0) return 0;

  // Calculate mean
  const meanX = x.reduce((a, b) => a + b, 0) / n;
  const meanY = y.reduce((a, b) => a + b, 0) / n;

  // Calculate numerator and denominator for Pearson correlation on ranks
  let numerator = 0;
  let denomX = 0;
  let denomY = 0;

  for (let i = 0; i < n; i++) {
    const diffX = x[i] - meanX;
    const diffY = y[i] - meanY;
    numerator += diffX * diffY;
    denomX += diffX * diffX;
    denomY += diffY * diffY;
  }

  const denominator = Math.sqrt(denomX * denomY);
  if (denominator === 0) return 0;
  return numerator / denominator;
}

function YearComparison({ year, dataType, selectedSection }: { year: string, dataType: 'dzial' | 'sekcja', selectedSection: string }) {
  // Fetch data for all datasets
  const { data: resultsData } = useCsvData<ResultData>(`/data/results/${year}/${dataType === 'dzial' ? 'dział' : 'sekcja'}/complete.csv`, { delimiter: ";", dynamicTyping: false });
  const { data: creditData } = useCsvData<ResultData>(`/data/results-credit/${year}/${dataType === 'dzial' ? 'dział' : 'sekcja'}/complete.csv`, { delimiter: ";", dynamicTyping: false });
  const { data: developmentData } = useCsvData<ResultData>(`/data/results-development/${year}/${dataType === 'dzial' ? 'dział' : 'sekcja'}/complete.csv`, { delimiter: ";", dynamicTyping: false });
  const { data: effectivityData } = useCsvData<ResultData>(`/data/results-effectivity/${year}/${dataType === 'dzial' ? 'dział' : 'sekcja'}/complete.csv`, { delimiter: ";", dynamicTyping: false });

  const correlationData = useMemo(() => {
    const datasets = [
      { id: 'results', data: resultsData },
      { id: 'results-credit', data: creditData },
      { id: 'results-development', data: developmentData },
      { id: 'results-effectivity', data: effectivityData },
    ];

    return datasets.map(ds => {
      const datasetInfo = DATASETS.find(d => d.id === ds.id)!;
      
      let filteredData = ds.data || [];

      if (dataType === 'dzial' && selectedSection) {
        filteredData = filteredData.filter(d => getSectionForDivision(d.alternative_id) === selectedSection);
      }
      
      if (filteredData.length === 0) {
        return {
          name: datasetInfo.label,
          correlation: 0,
          color: datasetInfo.color
        };
      }

      const topsisRanks = filteredData.map(d => Number(d.topsis_rank));
      const vikorRanks = filteredData.map(d => Number(d.vikor_rank));

      const correlation = calculateSpearmanCorrelation(topsisRanks, vikorRanks);

      return {
        name: datasetInfo.label,
        correlation: correlation,
        color: datasetInfo.color
      };
    });
  }, [resultsData, creditData, developmentData, effectivityData, dataType, selectedSection]);

  return (
    <DashboardCard title={`Rok ${year}`}>
      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        <div className="lg:col-span-2 h-[250px]">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={correlationData} margin={{ top: 20, right: 30, left: 40, bottom: 25 }}>
              <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" />
              <XAxis 
                dataKey="name" 
                stroke="#1a2f3a" 
                fontSize={12} 
                label={{ value: 'Obszar analizy', position: 'insideBottom', offset: -5, fontSize: 12, fill: '#666' }}
              />
              <YAxis 
                stroke="#1a2f3a" 
                domain={[-1, 1]} 
                ticks={[-1, -0.5, 0, 0.5, 1]} 
                fontSize={12} 
                label={{ value: 'Korelacja Spearmana', angle: -90, position: 'insideLeft', fontSize: 12, fill: '#666', style: { textAnchor: 'middle' } }}
              />
              <Tooltip 
                cursor={{ fill: 'transparent' }}
                contentStyle={{ 
                  backgroundColor: 'white', 
                  borderRadius: '8px', 
                  border: '1px solid #e5e7eb', 
                  boxShadow: '0 4px 6px -1px rgb(0 0 0 / 0.1)',
                  fontSize: '12px'
                }}
                formatter={(value: number) => [value.toFixed(2), 'Korelacja Spearmana']}
              />
              <Bar dataKey="correlation" name="Korelacja Spearmana" radius={[4, 4, 0, 0]}>
                {correlationData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={entry.color} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="grid grid-cols-2 gap-2 content-center">
          {correlationData.map((item) => (
            <div key={item.name} className="bg-gray-50 p-3 rounded border border-gray-100">
              <div className="text-xs text-gray-500 mb-1">{item.name}</div>
              <div className="text-xl font-bold" style={{ color: item.color }}>
                {item.correlation.toFixed(2)}
              </div>
            </div>
          ))}
        </div>
      </div>
    </DashboardCard>
  );
}

export default function Comparison() {
  const [dataType, setDataType] = useState<'dzial' | 'sekcja'>('sekcja');
  const [selectedSection, setSelectedSection] = useState<string>('');

  const YEARS = Array.from({ length: 2028 - 2013 + 1 }, (_, i) => (2013 + i).toString());

  return (
    <div className="space-y-8">
      <div className="flex flex-col md:flex-row justify-between items-start md:items-center gap-4">
        <div>
          <h2 className="text-2xl font-bold text-pko-navy mb-2">TOPSIS vs VIKOR</h2>
          <p className="text-pko-navy/70">
            Analiza korelacji Spearmana między rankingami TOPSIS i VIKOR dla różnych zestawów danych w czasie.
          </p>
        </div>
        
        <div className="flex gap-2">
          {dataType === 'dzial' && (
            <div className="bg-white p-1 rounded-lg border border-gray-200">
              <select 
                value={selectedSection} 
                onChange={(e) => setSelectedSection(e.target.value)}
                className="px-4 py-2 rounded-md text-sm font-medium text-gray-600 bg-transparent border-none focus:ring-0 cursor-pointer outline-none"
              >
                <option value="">Wszystkie sekcje</option>
                {Object.entries(SECTION_MAPPING).map(([key, { label }]) => (
                  <option key={key} value={key}>{key} - {label}</option>
                ))}
              </select>
            </div>
          )}

          <div className="bg-white p-1 rounded-lg border border-gray-200 flex gap-1">
            <button
              onClick={() => { setDataType('sekcja'); setSelectedSection(''); }}
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
      </div>

      <DashboardCard title="Interpretacja Analizy Porównawczej">
        <div className="space-y-6 text-sm text-gray-700">
          {/* Main Concept */}
          <div className="flex items-start gap-4 p-4 bg-blue-50 rounded-lg border border-blue-100">
            <div className="text-2xl">💡</div>
            <div>
              <h3 className="font-bold text-pko-navy mb-1">Kluczowa zasada</h3>
              <p>
                Wysoka korelacja oznacza zgodność metod rankingowych. 
                <span className="font-semibold"> Niska lub ujemna korelacja</span> sygnalizuje niejednoznaczność oceny sektora i konieczność pogłębionej analizy przez dział ryzyka.
              </p>
            </div>
          </div>

          {/* Detailed Breakdown */}
          <div>
            <h3 className="font-bold text-pko-navy mb-3">Przyczyny rozbieżności (TOPSIS vs VIKOR)</h3>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <div className="p-3 bg-gray-50 rounded border border-gray-100">
                <div className="font-semibold mb-1 text-pko-navy">Konfliktowe parametry</div>
                <p className="text-xs text-gray-600">Np. wysoka rentowność przy ekstremalnym zadłużeniu. VIKOR mocniej "karze" słabości niż TOPSIS.</p>
              </div>
              <div className="p-3 bg-gray-50 rounded border border-gray-100">
                <div className="font-semibold mb-1 text-pko-navy">Wrażliwość na wagi</div>
                <p className="text-xs text-gray-600">Ranking jest niestabilny – niewielka zmiana kryteriów drastycznie zmienia ocenę.</p>
              </div>
              <div className="p-3 bg-gray-50 rounded border border-gray-100">
                <div className="font-semibold mb-1 text-pko-navy">Trade-off: Ryzyko vs Zysk</div>
                <p className="text-xs text-gray-600">Sektory o wysokiej dynamice, ale dużej zmienności (np. technologie, budownictwo).</p>
              </div>
              <div className="p-3 bg-gray-50 rounded border border-gray-100">
                <div className="font-semibold mb-1 text-pko-navy">Niespójność danych</div>
                <p className="text-xs text-gray-600">Wartości ekstremalne w danych branżowych są różnie interpretowane przez obie metody.</p>
              </div>
            </div>
          </div>

          {/* Critical Warning */}
          <div className="flex items-center gap-3 p-3 bg-red-50 rounded-lg border border-red-100 text-red-800">
            <span className="text-xl">🚨</span>
            <p>
              <span className="font-bold">Ujemna korelacja</span> to sygnał alarmowy: metody oceniają sektory w sposób przeciwstawny. Wymagana weryfikacja manualna.
            </p>
          </div>
        </div>
      </DashboardCard>

      <div className="space-y-6">
        {YEARS.map(year => (
          <YearComparison key={year} year={year} dataType={dataType} selectedSection={selectedSection} />
        ))}
      </div>
    </div>
  );
}
