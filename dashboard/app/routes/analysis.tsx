import { useCsvData } from "../hooks/useCsvData";
import { DashboardCard } from "../components/DashboardCard";
import { AnalysisChart } from "../components/AnalysisChart";
import { useMemo } from "react";

interface ResultData {
  alternative_id: string;
  [key: string]: string | number;
}

export default function Analysis() {
  const { data: ensembleData } = useCsvData<ResultData>("/data/results_ensemble.csv");
  const { data: monteCarloData } = useCsvData<ResultData>("/data/results_monte_carlo.csv");
  const { data: topsisData } = useCsvData<ResultData>("/data/results_topsis.csv");
  const { data: vikorData } = useCsvData<ResultData>("/data/results_vikor.csv");

  const mergedData = useMemo(() => {
    if (!ensembleData.length || !monteCarloData.length || !topsisData.length || !vikorData.length) {
      return [];
    }

    const map = new Map<string, any>();

    const process = (data: ResultData[], prefix: string) => {
      data.forEach((item) => {
        if (!item.alternative_id) return;
        const existing = map.get(item.alternative_id) || { name: item.alternative_id };
        existing[`${prefix}_score`] = item[`${prefix}_score`];
        existing[`${prefix}_rank`] = item[`${prefix}_rank`];
        map.set(item.alternative_id, existing);
      });
    };

    process(ensembleData, "ensemble");
    process(monteCarloData, "monte_carlo");
    process(topsisData, "topsis");
    process(vikorData, "vikor");

    return Array.from(map.values());
  }, [ensembleData, monteCarloData, topsisData, vikorData]);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-pko-navy mb-2">Analiza Metod Decyzyjnych</h2>
        <p className="text-pko-navy/70">Porównanie wyników różnych metod wielokryterialnych.</p>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <DashboardCard title="Porównanie Wyników (Score)">
          <AnalysisChart
            data={mergedData}
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
            data={mergedData}
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
        <MethodCard title="Ensemble" color="bg-pko-navy" data={ensembleData} prefix="ensemble" />
        <MethodCard title="Monte Carlo" color="bg-pko-red" data={monteCarloData} prefix="monte_carlo" />
        <MethodCard title="TOPSIS" color="bg-pko-gold" data={topsisData} prefix="topsis" />
        <MethodCard title="VIKOR" color="bg-pko-black" data={vikorData} prefix="vikor" />
      </div>
    </div>
  );
}

function MethodCard({ title, color, data, prefix }: { title: string, color: string, data: ResultData[], prefix: string }) {
  const top3 = data.sort((a, b) => (Number(a[`${prefix}_rank`]) || 999) - (Number(b[`${prefix}_rank`]) || 999)).slice(0, 3);

  return (
    <div className="pko-card overflow-hidden">
      <div className={`${color} p-4 text-white`}>
        <h3 className="font-bold text-lg">{title}</h3>
      </div>
      <div className="p-4">
        <h4 className="text-sm font-semibold text-pko-navy mb-3">Top 3 Sektory:</h4>
        <ul className="space-y-2">
          {top3.map((item, i) => (
            <li key={i} className="flex items-center justify-between text-sm">
              <span className="text-pko-navy/80">{i + 1}. {item.alternative_id}</span>
              <span className="font-mono font-bold text-pko-red">
                {Number(item[`${prefix}_score`]).toFixed(3)}
              </span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}
