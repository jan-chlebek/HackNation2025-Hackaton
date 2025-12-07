import { useCsvData } from "../hooks/useCsvData";
import { DashboardCard } from "../components/DashboardCard";
import { AnalysisChart } from "../components/AnalysisChart";
import { useMemo, useState } from "react";

interface ResultData {
  alternative_id: string;
  nazwa: string;
  ensemble_rank: number;
  ensemble_score: number;
  topsis_rank: number;
  topsis_score: number;
  vikor_rank: number;
  vikor_score: number;
  monte_carlo_rank: number;
  monte_carlo_score: number;
  name?: string;
  [key: string]: string | number | undefined;
}

interface PkdDictionaryItem {
  PKD_INDEX: string;
  symbol: string;
  nazwa: string;
  TYP_INDEX: string;
}

interface PkdTypDictionaryItem {
  TYP_INDEX: string;
  typ: string;
}

export default function Analysis() {
  const [dataType, setDataType] = useState<'dzial' | 'sekcja'>('sekcja');
  const [selectedYear, setSelectedYear] = useState<string>('2024');
  const [selectedDataset, setSelectedDataset] = useState<string>('results');
  const [selectedPkdTyp, setSelectedPkdTyp] = useState<string>('1'); // Default to DZIAŁ (1)
  const [selectedMethods, setSelectedMethods] = useState<Record<string, boolean>>({
    ensemble: true,
    monte_carlo: true,
    topsis: true,
    vikor: true,
  });

  const YEARS = Array.from({ length: 2028 - 2013 + 1 }, (_, i) => (2013 + i).toString());
  
  const DATASETS = [
    { id: 'results', label: 'Ogólne' },
    { id: 'results-credit', label: 'Kredytowe' },
    { id: 'results-development', label: 'Rozwojowe' },
    { id: 'results-effectivity', label: 'Efektywność' },
  ];

  const { data } = useCsvData<ResultData>(`/data/${selectedDataset}/${selectedYear}/${dataType === 'dzial' ? 'dział' : 'sekcja'}/complete.csv`, { delimiter: ";", dynamicTyping: false });
  const { data: pkdData } = useCsvData<PkdDictionaryItem>("/data/pkd_dictionary.csv", { delimiter: ";", dynamicTyping: false });
  const { data: pkdTypData } = useCsvData<PkdTypDictionaryItem>("/data/pkd_typ_dictionary.csv", { delimiter: ";", dynamicTyping: false });

  const METHODS = [
    { id: 'ensemble', label: 'Ensemble', color: '#1a2f3a', bgClass: 'bg-pko-navy' },
    { id: 'monte_carlo', label: 'Monte Carlo', color: '#d93026', bgClass: 'bg-pko-red' },
    { id: 'topsis', label: 'TOPSIS', color: '#c9a961', bgClass: 'bg-pko-gold' },
    { id: 'vikor', label: 'VIKOR', color: '#16a34a', bgClass: 'bg-green-600' },
  ];

  const toggleMethod = (methodId: string) => {
    setSelectedMethods(prev => ({
      ...prev,
      [methodId]: !prev[methodId]
    }));
  };

  const selectedCount = Object.values(selectedMethods).filter(Boolean).length;

  const processedData = useMemo(() => {
    if (!data || data.length === 0) return [];

    const pkdMap = new Map<string, { symbol: string, typIndex: string }>();
    if (dataType === 'dzial' && pkdData) {
      pkdData.forEach(item => {
        pkdMap.set(String(item.symbol), { symbol: String(item.symbol), typIndex: String(item.TYP_INDEX) });
      });
    }

    let filteredData = data;

    // Filter by PKD Type if in 'dzial' mode
    if (dataType === 'dzial' && pkdData) {
      filteredData = data.filter(item => {
        const pkdInfo = pkdMap.get(String(item.alternative_id));
        return pkdInfo ? String(pkdInfo.typIndex) === String(selectedPkdTyp) : true;
      });
    }

    return filteredData.map(item => {
      let displayName = item.alternative_id;
      if (dataType === 'dzial' && pkdMap.has(String(item.alternative_id))) {
        displayName = pkdMap.get(String(item.alternative_id))?.symbol || item.alternative_id;
      }

      return {
        ...item,
        ensemble_rank: Number(item.ensemble_rank),
        ensemble_score: Number(item.ensemble_score),
        topsis_rank: Number(item.topsis_rank),
        topsis_score: Number(item.topsis_score),
        vikor_rank: Number(item.vikor_rank),
        vikor_score: Number(item.vikor_score),
        monte_carlo_rank: Number(item.monte_carlo_rank),
        monte_carlo_score: Number(item.monte_carlo_score),
        name: displayName,
        original_id: item.alternative_id
      };
    });
  }, [data, pkdData, dataType, selectedPkdTyp]);

  return (
    <div className="space-y-8">
      <div className="flex flex-col gap-4">
        <div className="w-full">
          <h2 className="text-2xl font-bold text-pko-navy mb-2">Analiza Metod Decyzyjnych</h2>
          <div className="text-pko-navy/70 space-y-3 text-sm">
            <p>
              Analiza porównawcza pozwala zweryfikować stabilność rankingu poprzez zestawienie różnych podejść matematycznych.
              Umożliwia to identyfikację sektorów, które są silne niezależnie od przyjętej metodyki oceny.
              Wyniki można filtrować według <strong>kategorii danych</strong> (np. Kredytowe, Rozwojowe), <strong>roku</strong> (historyczne i prognozy) oraz poziomu agregacji (<strong>Sekcje/Działy</strong>).
            </p>
            <div className="bg-blue-50 p-3 rounded-md border border-blue-100 text-pko-navy space-y-2">
              <p>
                <strong className="font-semibold">Ensemble</strong> to metoda hybrydowa, agregująca wyniki trzech pozostałych metod (TOPSIS, VIKOR, Monte Carlo). 
                Generuje ona główny ranking statystyczny, który jest najbardziej wiarygodny i odporny na specyfikę poszczególnych algorytmów.
              </p>
              <ul className="list-disc list-inside text-xs space-y-1 pt-1 border-t border-blue-200">
                <li><strong className="font-semibold">Monte Carlo</strong>: metoda probabilistyczna (losowa), symulująca tysiące scenariuszy wag dla kryteriów.</li>
                <li><strong className="font-semibold">TOPSIS i VIKOR</strong>: metody deterministyczne, optymalizujące ranking zgodnie z zadanym kierunkiem preferencji (odległość od ideału).</li>
              </ul>
            </div>
          </div>
        </div>
        <div className="flex flex-col sm:flex-row gap-4 items-end sm:items-center justify-end">
          <div className="relative">
            <select
              value={selectedDataset}
              onChange={(e) => setSelectedDataset(e.target.value)}
              className="appearance-none bg-white border border-gray-300 text-pko-navy text-sm rounded-lg focus:ring-pko-red focus:border-pko-red block w-full px-4 py-2 pr-8 cursor-pointer hover:bg-gray-50"
            >
              {DATASETS.map(dataset => (
                <option key={dataset.id} value={dataset.id}>{dataset.label}</option>
              ))}
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-pko-navy">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
            </div>
          </div>

          <div className="flex flex-wrap gap-3 bg-white px-3 py-2 rounded-lg shadow-sm border border-gray-100">
            {METHODS.map(method => (
              <label key={method.id} className="inline-flex items-center cursor-pointer select-none">
                <input
                  type="checkbox"
                  className="form-checkbox h-4 w-4 text-pko-navy rounded border-gray-300 focus:ring-pko-red"
                  checked={selectedMethods[method.id]}
                  onChange={() => toggleMethod(method.id)}
                />
                <span className={`ml-2 text-sm font-medium ${selectedMethods[method.id] ? 'text-pko-navy' : 'text-gray-400'}`}>
                  {method.label}
                </span>
              </label>
            ))}
          </div>

          <div className="relative">
            <select
              value={selectedYear}
              onChange={(e) => setSelectedYear(e.target.value)}
              className="appearance-none bg-white border border-gray-300 text-pko-navy text-sm rounded-lg focus:ring-pko-red focus:border-pko-red block w-full px-4 py-2 pr-8 cursor-pointer hover:bg-gray-50"
            >
              <optgroup label="Dane historyczne">
                {YEARS.filter(y => parseInt(y) <= 2024).map(year => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </optgroup>
              <optgroup label="Prognoza">
                {YEARS.filter(y => parseInt(y) > 2024).map(year => (
                  <option key={year} value={year}>{year}</option>
                ))}
              </optgroup>
            </select>
            <div className="pointer-events-none absolute inset-y-0 right-0 flex items-center px-2 text-pko-navy">
              <svg className="fill-current h-4 w-4" xmlns="http://www.w3.org/2000/svg" viewBox="0 0 20 20"><path d="M9.293 12.95l.707.707L15.657 8l-1.414-1.414L10 10.828 5.757 6.586 4.343 8z"/></svg>
            </div>
          </div>

          <div className="inline-flex rounded-md shadow-sm isolate">
            <button
              type="button"
              onClick={() => setDataType('sekcja')}
              className={`relative inline-flex items-center px-4 py-2 text-sm font-medium border focus:z-10 focus:ring-2 focus:ring-pko-red rounded-l-lg ${
                dataType === 'sekcja'
                  ? 'bg-pko-navy text-white border-pko-navy z-10'
                  : 'bg-white text-pko-navy border-gray-300 hover:bg-gray-50'
              }`}
            >
              Sekcje
            </button>
            <button
              type="button"
              onClick={() => setDataType('dzial')}
              className={`relative inline-flex items-center px-4 py-2 text-sm font-medium border focus:z-10 focus:ring-2 focus:ring-pko-red rounded-r-lg -ml-px ${
                dataType === 'dzial'
                  ? 'bg-pko-navy text-white border-pko-navy z-10'
                  : 'bg-white text-pko-navy border-gray-300 hover:bg-gray-50'
              }`}
            >
              Działy
            </button>
          </div>
        </div>
      </div>

      <div className={`grid grid-cols-1 ${selectedCount >= 2 ? 'lg:grid-cols-2' : ''} gap-6`}>
        <DashboardCard title="Porównanie Wyników (Score)">
          <AnalysisChart
            data={processedData}
            xKey="name"
            dataKeys={METHODS.filter(m => selectedMethods[m.id]).map(m => ({
              key: `${m.id}_score`,
              color: m.color,
              name: m.label
            }))}
          />
        </DashboardCard>

        {selectedCount >= 2 && (
          <DashboardCard title="Porównanie Rankingów">
            <AnalysisChart
              data={processedData}
              xKey="name"
              type="bar"
              dataKeys={METHODS.filter(m => selectedMethods[m.id]).map(m => ({
                key: `${m.id}_rank`,
                color: m.color,
                name: `${m.label} Rank`
              }))}
            />
          </DashboardCard>
        )}
      </div>

      <div className={`grid grid-cols-1 gap-4 ${
        selectedCount === 1 ? '' :
        selectedCount === 2 ? 'md:grid-cols-2' :
        selectedCount === 3 ? 'md:grid-cols-2 lg:grid-cols-3' :
        'md:grid-cols-2 lg:grid-cols-4'
      }`}>
        {METHODS.filter(m => selectedMethods[m.id]).map(method => (
          <MethodCard 
            key={method.id}
            title={method.label} 
            color={method.bgClass} 
            data={processedData} 
            prefix={method.id} 
          />
        ))}
      </div>
    </div>
  );
}

function MethodCard({ title, color, data, prefix }: { title: string, color: string, data: ResultData[], prefix: string }) {
  const [isExpanded, setIsExpanded] = useState(false);
  // Create a copy before sorting to avoid mutating the original array
  const sortedData = [...data].sort((a, b) => (Number(a[`${prefix}_rank`]) || 999) - (Number(b[`${prefix}_rank`]) || 999));
  const top3 = sortedData.slice(0, 3);
  const worst3 = sortedData.slice(-3);

  return (
    <div className={`pko-card overflow-hidden flex flex-col transition-all duration-300 ${isExpanded ? 'row-span-2 h-[500px]' : ''}`}>
      <div className={`${color} p-4 text-white flex justify-between items-center`}>
        <h3 className="font-bold text-lg">{title}</h3>
        <button 
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-white/80 hover:text-white text-xs bg-white/20 px-2 py-1 rounded transition-colors"
        >
          {isExpanded ? 'Zwiń' : 'Pełna tabela'}
        </button>
      </div>
      <div className="p-4 space-y-4 flex-1 overflow-hidden flex flex-col">
        {isExpanded ? (
          <div className="overflow-y-auto flex-1 pr-2 custom-scrollbar">
            <table className="w-full text-xs text-left">
              <thead className="sticky top-0 bg-white text-pko-navy font-bold border-b border-gray-200 shadow-sm z-10">
                <tr>
                  <th className="py-2 pl-1">Rank</th>
                  <th className="py-2">Sektor</th>
                  <th className="py-2 text-right pr-1">Score</th>
                </tr>
              </thead>
              <tbody>
                {sortedData.map((item, i) => (
                  <tr key={i} className="border-b border-gray-50 hover:bg-gray-50">
                    <td className="py-2 pl-1 font-bold text-pko-navy/70">#{item[`${prefix}_rank`]}</td>
                    <td className="py-2 px-2">
                      <div className="truncate max-w-[180px]" title={`${item.name || item.alternative_id} - ${item.nazwa}`}>
                        {item.name || item.alternative_id} - {item.nazwa}
                      </div>
                    </td>
                    <td className="py-2 text-right font-mono pr-1">
                      {Number(item[`${prefix}_score`]).toFixed(4)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : (
          <>
            <div>
              <h4 className="text-sm font-semibold text-pko-navy mb-2">Top 3 Sektory:</h4>
              <ul className="space-y-1">
                {top3.map((item, i) => (
                  <li key={i} className="flex items-center justify-between text-xs">
                    <span className="font-medium text-pko-navy truncate flex-1 mr-2" title={`${item.name || item.alternative_id} - ${item.nazwa}`}>
                      {item.name || item.alternative_id} - {item.nazwa}
                    </span>
                    <span className="font-bold text-pko-navy/70">#{item[`${prefix}_rank`]}</span>
                  </li>
                ))}
              </ul>
            </div>
            
            <div className="pt-2 border-t border-gray-100">
              <h4 className="text-sm font-semibold text-pko-navy mb-2">Worst 3 Sektory:</h4>
              <ul className="space-y-1">
                {worst3.map((item, i) => (
                  <li key={i} className="flex items-center justify-between text-xs">
                    <span className="font-medium text-pko-navy truncate flex-1 mr-2" title={`${item.name || item.alternative_id} - ${item.nazwa}`}>
                      {item.name || item.alternative_id} - {item.nazwa}
                    </span>
                    <span className="font-bold text-pko-navy/70">#{item[`${prefix}_rank`]}</span>
                  </li>
                ))}
              </ul>
            </div>
          </>
        )}
      </div>
    </div>
  );
}
