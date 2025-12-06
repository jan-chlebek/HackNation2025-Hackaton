import { useState, useMemo, useEffect } from "react";
import { useCsvData } from "../hooks/useCsvData";
import { DashboardCard } from "../components/DashboardCard";
import { AnalysisChart } from "../components/AnalysisChart";
import { Loader2 } from "lucide-react";

interface KpiData {
  rok: number;
  wartosc: string; // It has spaces and commas, need to parse
  WSKAZNIK_INDEX: number;
  PKD_INDEX: number;
}

interface WskaznikDict {
  WSKAZNIK_INDEX: number;
  WSKAZNIK: string;
}

interface PkdDict {
  PKD_INDEX: number;
  symbol: string;
  nazwa: string;
  TYP_INDEX: number;
}

function parseValue(val: string | number): number {
  if (typeof val === "number") return val;
  if (!val) return 0;
  // Remove spaces (thousands separator) and replace comma with dot
  return parseFloat(val.replace(/\s/g, "").replace(",", "."));
}

function calculateCorrelation(x: number[], y: number[]): number {
  const n = x.length;
  if (n !== y.length || n === 0) return 0;

  const sumX = x.reduce((a, b) => a + b, 0);
  const sumY = y.reduce((a, b) => a + b, 0);
  const sumXY = x.reduce((a, b, i) => a + b * y[i], 0);
  const sumX2 = x.reduce((a, b) => a + b * b, 0);
  const sumY2 = y.reduce((a, b) => a + b * b, 0);

  const numerator = n * sumXY - sumX * sumY;
  const denominator = Math.sqrt((n * sumX2 - sumX * sumX) * (n * sumY2 - sumY * sumY));

  if (denominator === 0) return 0;
  return numerator / denominator;
}

export default function Correlations() {
  const { data: kpiData, loading: kpiLoading } = useCsvData<KpiData>("/data/kpi-value-table.csv");
  const { data: wskaznikDict } = useCsvData<WskaznikDict>("/data/wskaznik_dictionary.csv");
  const { data: pkdDict } = useCsvData<PkdDict>("/data/pkd_dictionary.csv");

  const [selectedPkd, setSelectedPkd] = useState<number | null>(null);

  // Process dictionaries
  const wskaznikMap = useMemo(() => {
    const map = new Map<number, string>();
    wskaznikDict.forEach(w => map.set(w.WSKAZNIK_INDEX, w.WSKAZNIK));
    return map;
  }, [wskaznikDict]);

  const pkdOptions = useMemo(() => {
    return pkdDict.map(p => ({ value: p.PKD_INDEX, label: `${p.symbol} - ${p.nazwa}` }));
  }, [pkdDict]);

  // Set default PKD once loaded
  useEffect(() => {
    if (selectedPkd === null && pkdOptions.length > 0) {
      setSelectedPkd(pkdOptions[0].value);
    }
  }, [pkdOptions, selectedPkd]);

  // Filter and process data for selected PKD
  const chartData = useMemo(() => {
    if (selectedPkd === null || !kpiData.length) return [];

    const filtered = kpiData.filter(d => d.PKD_INDEX === selectedPkd);
    
    // Group by year
    const byYear = new Map<number, any>();
    filtered.forEach(d => {
      const year = d.rok;
      const val = parseValue(d.wartosc);
      const wskaznikName = wskaznikMap.get(d.WSKAZNIK_INDEX) || `W${d.WSKAZNIK_INDEX}`;
      
      const existing = byYear.get(year) || { name: year };
      existing[wskaznikName] = val;
      byYear.set(year, existing);
    });

    return Array.from(byYear.values()).sort((a, b) => a.name - b.name);
  }, [kpiData, selectedPkd, wskaznikMap]);

  // Find exact keys for the chart
  const chartKeys = useMemo(() => {
    if (wskaznikMap.size === 0) return [];
    const indicators = Array.from(wskaznikMap.values());
    
    const findKey = (partial: string) => indicators.find(i => i.includes(partial));

    return [
      { key: findKey("Przychody ogółem") || "GS Przychody ogółem ", color: "#1a2f3a", name: "Przychody" },
      { key: findKey("Koszty ogółem") || "TC Koszty ogółem ", color: "#d93026", name: "Koszty" },
      { key: findKey("Wynik finansowy netto") || "NP Wynik finansowy netto (zysk netto) ", color: "#c9a961", name: "Zysk Netto" },
    ].filter(k => k.key); // Filter out undefined if not found
  }, [wskaznikMap]);

  // Calculate correlations between indicators for the selected PKD
  const correlations = useMemo(() => {
    if (!chartData.length || wskaznikMap.size === 0) return [];

    const indicators = Array.from(wskaznikMap.values());
    const matrix: { indicator1: string, indicator2: string, value: number }[] = [];

    // We only take top 5 indicators by variance or just first few to avoid huge matrix
    const keyIndicators = indicators.filter(i => 
      i.includes("Wynik finansowy netto") || 
      i.includes("Przychody ogółem") || 
      i.includes("Koszty ogółem") ||
      i.includes("Zapasy")
    );

    for (let i = 0; i < keyIndicators.length; i++) {
      for (let j = i + 1; j < keyIndicators.length; j++) {
        const ind1 = keyIndicators[i];
        const ind2 = keyIndicators[j];
        
        const values1 = chartData.map(d => d[ind1] || 0);
        const values2 = chartData.map(d => d[ind2] || 0);
        
        const corr = calculateCorrelation(values1, values2);
        matrix.push({ indicator1: ind1, indicator2: ind2, value: corr });
      }
    }
    return matrix.sort((a, b) => Math.abs(b.value) - Math.abs(a.value));
  }, [chartData, wskaznikMap]);

  if (kpiLoading) {
    return (
      <div className="flex items-center justify-center h-64">
        <Loader2 className="w-8 h-8 animate-spin text-pko-navy" />
        <span className="ml-2 text-pko-navy">Ładowanie danych...</span>
      </div>
    );
  }

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-pko-navy mb-2">Korelacje i Trendy</h2>
        <p className="text-pko-navy/70">Analiza wskaźników finansowych w czasie.</p>
      </div>

      <div className="bg-white p-4 rounded-lg shadow-sm border border-pko-gray-medium/30">
        <label className="block text-sm font-medium text-pko-navy mb-2">Wybierz Sektor (PKD)</label>
        <select 
          className="w-full max-w-md p-2 border border-pko-gray-medium rounded-md focus:outline-none focus:ring-2 focus:ring-pko-red"
          value={selectedPkd ?? ""}
          onChange={(e) => setSelectedPkd(Number(e.target.value))}
        >
          {pkdOptions.map(opt => (
            <option key={opt.value} value={opt.value}>{opt.label}</option>
          ))}
        </select>
      </div>

      <DashboardCard title="Trendy Wskaźników Finansowych">
        <AnalysisChart
          data={chartData}
          xKey="name"
          type="line"
          height={500}
          dataKeys={chartKeys as any}
        />
        <p className="text-xs text-gray-500 mt-2">* Wyświetlono wybrane kluczowe wskaźniki. Nazwy mogą się różnić w zależności od danych.</p>
      </DashboardCard>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DashboardCard title="Macierz Korelacji (Kluczowe Wskaźniki)">
          <div className="overflow-x-auto">
            <table className="w-full text-sm text-left">
              <thead className="text-xs text-pko-navy uppercase bg-pko-gray-light">
                <tr>
                  <th className="px-4 py-2">Wskaźnik 1</th>
                  <th className="px-4 py-2">Wskaźnik 2</th>
                  <th className="px-4 py-2 text-right">Korelacja</th>
                </tr>
              </thead>
              <tbody>
                {correlations.map((c, i) => (
                  <tr key={i} className="border-b border-pko-gray-medium/20 hover:bg-pko-gray-light/50">
                    <td className="px-4 py-2 font-medium">{c.indicator1}</td>
                    <td className="px-4 py-2">{c.indicator2}</td>
                    <td className="px-4 py-2 text-right font-mono">
                      <span className={c.value > 0.7 ? "text-green-600 font-bold" : c.value < -0.7 ? "text-red-600 font-bold" : "text-gray-600"}>
                        {c.value.toFixed(4)}
                      </span>
                    </td>
                  </tr>
                ))}
                {correlations.length === 0 && (
                  <tr>
                    <td colSpan={3} className="px-4 py-4 text-center text-gray-500">Brak danych do obliczenia korelacji dla wybranych wskaźników.</td>
                  </tr>
                )}
              </tbody>
            </table>
          </div>
        </DashboardCard>
      </div>
    </div>
  );
}
