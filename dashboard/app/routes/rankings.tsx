import { useCsvData } from "../hooks/useCsvData";
import { DashboardCard } from "../components/DashboardCard";
import { useMemo, useState } from "react";
import { clsx } from "clsx";

interface ResultData {
  alternative_id: string;
  [key: string]: string | number;
}

export default function Rankings() {
  const { data: ensembleData } = useCsvData<ResultData>("/data/results_ensemble.csv");
  const { data: monteCarloData } = useCsvData<ResultData>("/data/results_monte_carlo.csv");
  const { data: topsisData } = useCsvData<ResultData>("/data/results_topsis.csv");
  const { data: vikorData } = useCsvData<ResultData>("/data/results_vikor.csv");

  const [sortConfig, setSortConfig] = useState<{ key: string; direction: 'asc' | 'desc' }>({ key: 'ensemble_rank', direction: 'asc' });

  const mergedData = useMemo(() => {
    if (!ensembleData.length || !monteCarloData.length || !topsisData.length || !vikorData.length) {
      return [];
    }

    const map = new Map<string, any>();

    const process = (data: ResultData[], prefix: string) => {
      data.forEach((item) => {
        if (!item.alternative_id) return;
        const existing = map.get(item.alternative_id) || { name: item.alternative_id };
        existing[`${prefix}_score`] = Number(item[`${prefix}_score`]);
        existing[`${prefix}_rank`] = Number(item[`${prefix}_rank`]);
        map.set(item.alternative_id, existing);
      });
    };

    process(ensembleData, "ensemble");
    process(monteCarloData, "monte_carlo");
    process(topsisData, "topsis");
    process(vikorData, "vikor");

    return Array.from(map.values());
  }, [ensembleData, monteCarloData, topsisData, vikorData]);

  const sortedData = useMemo(() => {
    const sorted = [...mergedData];
    if (sortConfig.key) {
      sorted.sort((a, b) => {
        if (a[sortConfig.key] < b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? -1 : 1;
        }
        if (a[sortConfig.key] > b[sortConfig.key]) {
          return sortConfig.direction === 'asc' ? 1 : -1;
        }
        return 0;
      });
    }
    return sorted;
  }, [mergedData, sortConfig]);

  const requestSort = (key: string) => {
    let direction: 'asc' | 'desc' = 'asc';
    if (sortConfig.key === key && sortConfig.direction === 'asc') {
      direction = 'desc';
    }
    setSortConfig({ key, direction });
  };

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-pko-navy mb-2">Szczegółowe Rankingi</h2>
        <p className="text-pko-navy/70">Tabela porównawcza wszystkich sektorów.</p>
      </div>

      <DashboardCard title="Tabela Wyników">
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left">
            <thead className="text-xs text-pko-navy uppercase bg-pko-gray-light">
              <tr>
                <th className="px-4 py-3 cursor-pointer hover:bg-gray-200" onClick={() => requestSort('name')}>Sektor</th>
                <th className="px-4 py-3 text-center cursor-pointer hover:bg-gray-200" onClick={() => requestSort('ensemble_rank')}>Ensemble Rank</th>
                <th className="px-4 py-3 text-center cursor-pointer hover:bg-gray-200" onClick={() => requestSort('monte_carlo_rank')}>Monte Carlo Rank</th>
                <th className="px-4 py-3 text-center cursor-pointer hover:bg-gray-200" onClick={() => requestSort('topsis_rank')}>TOPSIS Rank</th>
                <th className="px-4 py-3 text-center cursor-pointer hover:bg-gray-200" onClick={() => requestSort('vikor_rank')}>VIKOR Rank</th>
                <th className="px-4 py-3 text-right cursor-pointer hover:bg-gray-200" onClick={() => requestSort('ensemble_score')}>Ensemble Score</th>
              </tr>
            </thead>
            <tbody>
              {sortedData.map((row, i) => (
                <tr key={i} className="border-b border-pko-gray-medium/20 hover:bg-pko-gray-light/50">
                  <td className="px-4 py-3 font-medium text-pko-navy">{row.name}</td>
                  <td className="px-4 py-3 text-center">
                    <RankBadge rank={row.ensemble_rank} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <RankBadge rank={row.monte_carlo_rank} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <RankBadge rank={row.topsis_rank} />
                  </td>
                  <td className="px-4 py-3 text-center">
                    <RankBadge rank={row.vikor_rank} />
                  </td>
                  <td className="px-4 py-3 text-right font-mono text-pko-navy/80">
                    {row.ensemble_score?.toFixed(4)}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </DashboardCard>
    </div>
  );
}

function RankBadge({ rank }: { rank: number }) {
  return (
    <span className={clsx(
      "inline-flex items-center justify-center w-6 h-6 rounded-full text-xs font-bold",
      rank === 1 ? "bg-pko-gold text-white" :
      rank === 2 ? "bg-gray-400 text-white" :
      rank === 3 ? "bg-orange-400 text-white" :
      "bg-pko-gray-light text-pko-navy"
    )}>
      {rank}
    </span>
  );
}
