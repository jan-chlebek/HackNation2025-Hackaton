import { useMemo } from "react";
import { useCsvData } from "../hooks/useCsvData";
import { DashboardCard } from "../components/DashboardCard";
import {
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  LabelList,
  ReferenceLine,
  Legend
} from "recharts";

interface ResultData {
  alternative_id: string;
  nazwa: string;
  ensemble_score: number;
  [key: string]: string | number | undefined;
}

export default function LoanNeeds() {
  const year = "2024"; // Default to 2024
  const level = "sekcja"; // Default to sekcja for high level view

  const { data: creditData } = useCsvData<ResultData>(`/data/results-credit/${year}/${level}/complete.csv`, { delimiter: ";", dynamicTyping: true });
  const { data: developmentData } = useCsvData<ResultData>(`/data/results-development/${year}/${level}/complete.csv`, { delimiter: ";", dynamicTyping: true });
  const { data: effectivityData } = useCsvData<ResultData>(`/data/results-effectivity/${year}/${level}/complete.csv`, { delimiter: ";", dynamicTyping: true });

  const mergedData = useMemo(() => {
    if (!creditData.length || !developmentData.length || !effectivityData.length) return [];

    const devMap = new Map(developmentData.map(d => [d.alternative_id, d.ensemble_score]));
    const effMap = new Map(effectivityData.map(d => [d.alternative_id, d.ensemble_score]));

    return creditData.map(c => {
      const devScore = devMap.get(c.alternative_id) || 0;
      const effScore = effMap.get(c.alternative_id) || 0;
      const willingness = (devScore + effScore) / 2;

      return {
        id: c.alternative_id,
        name: c.nazwa,
        creditScore: c.ensemble_score, // Loan Needs
        willingnessScore: willingness, // Willingness to Borrow
        developmentScore: devScore,
        effectivityScore: effScore
      };
    }).filter(d => d.creditScore > 0 && d.willingnessScore > 0); // Filter out empty/zero if needed
  }, [creditData, developmentData, effectivityData]);

  return (
    <div className="p-6 space-y-6">
      <h1 className="text-3xl font-bold text-gray-900">Analiza Potrzeb Pożyczkowych Sektorów</h1>
      <p className="text-gray-600">
        Analiza porównawcza potrzeb pożyczkowych (Credit Score) względem potencjału i szybkości rozwoju (Development & Effectivity).
      </p>

      <DashboardCard title="Mapa Potrzeb Pożyczkowych (2024)">
        <div className="h-[600px] w-full">
          <ResponsiveContainer width="100%" height="100%">
            <ScatterChart margin={{ top: 20, right: 30, bottom: 60, left: 60 }}>
              <CartesianGrid />
              <XAxis 
                type="number" 
                dataKey="creditScore" 
                name="Potrzeby Pożyczkowe (Credit)" 
                unit="" 
                domain={[0, 1]} 
                label={{ value: 'Potrzeby Pożyczkowe (Credit Score)', position: 'bottom', offset: 40 }} 
              />
              <YAxis 
                type="number" 
                dataKey="willingnessScore" 
                name="Szybkość Rozwoju" 
                unit="" 
                domain={[0, 1]} 
                label={{ value: 'Szybkość Rozwoju (Development + Effectivity)', angle: -90, position: 'insideLeft', offset: 0, style: { textAnchor: 'middle' } }} 
              />
              <Tooltip cursor={{ strokeDasharray: '3 3' }} content={({ active, payload }) => {
                if (active && payload && payload.length) {
                  const data = payload[0].payload;
                  return (
                    <div className="bg-white p-2 border border-gray-200 shadow-sm rounded text-sm">
                      <p className="font-bold">{data.id} - {data.name}</p>
                      <p>Potrzeby: {data.creditScore.toFixed(3)}</p>
                      <p>Szybkość: {data.willingnessScore.toFixed(3)}</p>
                      <p className="text-xs text-gray-500">Dev: {data.developmentScore.toFixed(3)}, Eff: {data.effectivityScore.toFixed(3)}</p>
                    </div>
                  );
                }
                return null;
              }} />
              <Legend verticalAlign="top" height={36} />
              <ReferenceLine x={0.5} stroke="#374151" strokeWidth={2} />
              <ReferenceLine y={0.5} stroke="#374151" strokeWidth={2} />
              <Scatter name="Sektory" data={mergedData} fill="#8884d8" r={8}>
                 <LabelList dataKey="id" position="top" />
              </Scatter>
            </ScatterChart>
          </ResponsiveContainer>
        </div>
      </DashboardCard>

      <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
        <DashboardCard title="Top Sektory - Największe Potrzeby">
            <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sektor</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {mergedData.sort((a, b) => b.creditScore - a.creditScore).slice(0, 5).map((item) => (
                            <tr key={item.id}>
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.id} - {item.name}</td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.creditScore.toFixed(3)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </DashboardCard>

        <DashboardCard title="Top Sektory - Najszybszy Rozwój">
             <div className="overflow-x-auto">
                <table className="min-w-full divide-y divide-gray-200">
                    <thead className="bg-gray-50">
                        <tr>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Sektor</th>
                            <th className="px-6 py-3 text-left text-xs font-medium text-gray-500 uppercase tracking-wider">Score</th>
                        </tr>
                    </thead>
                    <tbody className="bg-white divide-y divide-gray-200">
                        {mergedData.sort((a, b) => b.willingnessScore - a.willingnessScore).slice(0, 5).map((item) => (
                            <tr key={item.id}>
                                <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">{item.id} - {item.name}</td>
                                <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">{item.willingnessScore.toFixed(3)}</td>
                            </tr>
                        ))}
                    </tbody>
                </table>
            </div>
        </DashboardCard>
      </div>
    </div>
  );
}
