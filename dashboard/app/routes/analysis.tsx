import { useCsvData } from "../hooks/useCsvData";
import { DashboardCard } from "../components/DashboardCard";
import { AnalysisChart } from "../components/AnalysisChart";
import { useMemo } from "react";

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
  [key: string]: string | number;
}

export default function Analysis() {
  const { data } = useCsvData<ResultData>("/data/complete.csv");

  const processedData = useMemo(() => {
    if (!data || data.length === 0) return [];
    return data.map(item => ({
      ...item,
      name: item.alternative_id // Map alternative_id to name for the chart
    }));
  }, [data]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-pko-navy mb-2">Analiza Metod Decyzyjnych</h2>
        <p className="text-pko-navy/70">Porównanie wyników różnych metod wielokryterialnych.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DashboardCard title="Porównanie Wyników (Score)">
          <AnalysisChart
            data={processedData}
            xKey="name"
            dataKeys={[
              { key: "ensemble_score", color: "#1a2f3a", name: "Ensemble" },
              { key: "monte_carlo_score", color: "#d93026", name: "Monte Carlo" },
              { key: "topsis_score", color: "#c9a961", name: "TOPSIS" },
              { key: "vikor_score", color: "#1a1a1a", name: "VIKOR" },
            ]}
          />
        </DashboardCard>

        <DashboardCard title="Porównanie Rankingów">
           <AnalysisChart
            data={processedData}
            xKey="name"
            type="bar"
            dataKeys={[
              { key: "ensemble_rank", color: "#1a2f3a", name: "Ensemble Rank" },
              { key: "monte_carlo_rank", color: "#d93026", name: "Monte Carlo Rank" },
              { key: "topsis_rank", color: "#c9a961", name: "TOPSIS Rank" },
              { key: "vikor_rank", color: "#1a1a1a", name: "VIKOR Rank" },
            ]}
          />
        </DashboardCard>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        <MethodCard title="Ensemble" color="bg-pko-navy" data={processedData} prefix="ensemble" />
        <MethodCard title="Monte Carlo" color="bg-pko-red" data={processedData} prefix="monte_carlo" />
        <MethodCard title="TOPSIS" color="bg-pko-gold" data={processedData} prefix="topsis" />
        <MethodCard title="VIKOR" color="bg-pko-black" data={processedData} prefix="vikor" />
      </div>
    </div>
  );
}

function MethodCard({ title, color, data, prefix }: { title: string, color: string, data: ResultData[], prefix: string }) {
  // Create a copy before sorting to avoid mutating the original array
  const sortedData = [...data].sort((a, b) => (Number(a[`${prefix}_rank`]) || 999) - (Number(b[`${prefix}_rank`]) || 999));
  const top3 = sortedData.slice(0, 3);
  const worst3 = sortedData.slice(-3);

  return (
    <div className="pko-card overflow-hidden">
      <div className={`${color} p-4 text-white`}>
        <h3 className="font-bold text-lg">{title}</h3>
      </div>
      <div className="p-4 space-y-4">
        <div>
          <h4 className="text-sm font-semibold text-pko-navy mb-2">Top 3 Sektory:</h4>
          <ul className="space-y-1">
            {top3.map((item, i) => (
              <li key={i} className="flex items-center justify-between text-xs">
                <span className="font-medium text-pko-navy truncate flex-1 mr-2" title={`${item.alternative_id} - ${item.nazwa}`}>
                  {item.alternative_id} - {item.nazwa}
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
                <span className="font-medium text-pko-navy truncate flex-1 mr-2" title={`${item.alternative_id} - ${item.nazwa}`}>
                  {item.alternative_id} - {item.nazwa}
                </span>
                <span className="font-bold text-pko-navy/70">#{item[`${prefix}_rank`]}</span>
              </li>
            ))}
          </ul>
        </div>
      </div>
    </div>
  );
}
