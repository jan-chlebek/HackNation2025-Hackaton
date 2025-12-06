import type { Route } from "./+types/analytics";
import { useLoaderData } from "react-router";
import Papa from "papaparse";
import { Navigation } from "../components/Navigation";
import {
  BarChart,
  Bar,
  LineChart,
  Line,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  Cell,
} from "recharts";

// PKO BP Brand Colors
const PKO_COLORS = {
  navy: "#1a2f3a",
  red: "#d93026",
  gold: "#c9a961",
  black: "#1a1a1a",
  white: "#ffffff",
  grayLight: "#f5f5f5",
  grayMedium: "#d0d0d0",
};

const CHART_COLORS = [
  PKO_COLORS.navy,
  PKO_COLORS.red,
  PKO_COLORS.gold,
  "#4a90e2",
  "#50c878",
  "#ff6b6b",
  "#9b59b6",
  "#f39c12",
  "#1abc9c",
  "#e74c3c",
];

interface AnalysisResult {
  alternative_id: string;
  score: number;
  rank: number;
}

interface LoaderData {
  topsisResults: AnalysisResult[];
  vikorResults: AnalysisResult[];
  monteCarloResults: AnalysisResult[];
  ensembleResults: AnalysisResult[];
  correlations: {
    topsisVikor: number;
    topsisEnsemble: number;
    vikorEnsemble: number;
    topsisMonteCarloRank: number;
    vikorMonteCarloRank: number;
  };
}

export async function loader({ request }: Route.LoaderArgs) {
  // Load CSV files
  const [topsisRes, vikorRes, monteCarloRes, ensembleRes] = await Promise.all([
    fetch(new URL("/data/results_topsis.csv", request.url)),
    fetch(new URL("/data/results_vikor.csv", request.url)),
    fetch(new URL("/data/results_monte_carlo.csv", request.url)),
    fetch(new URL("/data/results_ensemble.csv", request.url)),
  ]);

  const [topsisText, vikorText, monteCarloText, ensembleText] =
    await Promise.all([
      topsisRes.text(),
      vikorRes.text(),
      monteCarloRes.text(),
      ensembleRes.text(),
    ]);

  // Parse CSV data
  const parseTopsis = Papa.parse<{
    alternative_id: string;
    topsis_score: string;
    topsis_rank: string;
  }>(topsisText, { header: true });
  const parseVikor = Papa.parse<{
    alternative_id: string;
    vikor_score: string;
    vikor_rank: string;
  }>(vikorText, { header: true });
  const parseMonteCarlo = Papa.parse<{
    alternative_id: string;
    monte_carlo_score: string;
    monte_carlo_rank: string;
  }>(monteCarloText, { header: true });
  const parseEnsemble = Papa.parse<{
    alternative_id: string;
    ensemble_score: string;
    ensemble_rank: string;
  }>(ensembleText, { header: true });

  const topsisResults = parseTopsis.data
    .filter((d) => d.alternative_id && d.topsis_score)
    .map((d) => ({
      alternative_id: d.alternative_id,
      score: parseFloat(d.topsis_score),
      rank: parseFloat(d.topsis_rank),
    }));

  const vikorResults = parseVikor.data
    .filter((d) => d.alternative_id && d.vikor_score)
    .map((d) => ({
      alternative_id: d.alternative_id,
      score: parseFloat(d.vikor_score),
      rank: parseFloat(d.vikor_rank),
    }));

  const monteCarloResults = parseMonteCarlo.data
    .filter((d) => d.alternative_id && d.monte_carlo_score)
    .map((d) => ({
      alternative_id: d.alternative_id,
      score: parseFloat(d.monte_carlo_score),
      rank: parseFloat(d.monte_carlo_rank),
    }));

  const ensembleResults = parseEnsemble.data
    .filter((d) => d.alternative_id && d.ensemble_score)
    .map((d) => ({
      alternative_id: d.alternative_id,
      score: parseFloat(d.ensemble_score),
      rank: parseFloat(d.ensemble_rank),
    }));

  // Calculate correlations between methods
  const correlations = calculateCorrelations(
    topsisResults,
    vikorResults,
    monteCarloResults,
    ensembleResults
  );

  return {
    topsisResults,
    vikorResults,
    monteCarloResults,
    ensembleResults,
    correlations,
  };
}

function calculateCorrelations(
  topsis: AnalysisResult[],
  vikor: AnalysisResult[],
  monteCarlo: AnalysisResult[],
  ensemble: AnalysisResult[]
) {
  // Pearson correlation coefficient calculation
  const pearsonCorrelation = (x: number[], y: number[]) => {
    if (x.length !== y.length || x.length === 0) return 0;
    
    const n = x.length;
    const sum_x = x.reduce((a, b) => a + b, 0);
    const sum_y = y.reduce((a, b) => a + b, 0);
    const sum_xy = x.reduce((sum, xi, i) => sum + xi * y[i], 0);
    const sum_x2 = x.reduce((sum, xi) => sum + xi * xi, 0);
    const sum_y2 = y.reduce((sum, yi) => sum + yi * yi, 0);

    const numerator = n * sum_xy - sum_x * sum_y;
    const denominator = Math.sqrt(
      (n * sum_x2 - sum_x * sum_x) * (n * sum_y2 - sum_y * sum_y)
    );

    if (denominator === 0) return 0;
    return numerator / denominator;
  };

  // Create maps for alignment by alternative_id
  const vikorMap = new Map(vikor.map((r) => [r.alternative_id, r]));
  const monteCarloMap = new Map(monteCarlo.map((r) => [r.alternative_id, r]));
  const ensembleMap = new Map(ensemble.map((r) => [r.alternative_id, r]));

  // Align all data by TOPSIS order
  const alignedData = topsis
    .filter((t) => vikorMap.has(t.alternative_id) && ensembleMap.has(t.alternative_id))
    .map((t) => ({
      topsisScore: t.score,
      topsisRank: t.rank,
      vikorScore: vikorMap.get(t.alternative_id)!.score,
      vikorRank: vikorMap.get(t.alternative_id)!.rank,
      monteCarloRank: monteCarloMap.get(t.alternative_id)?.rank || 0,
      ensembleScore: ensembleMap.get(t.alternative_id)!.score,
    }));

  const topsisScores = alignedData.map((d) => d.topsisScore);
  const vikorScores = alignedData.map((d) => d.vikorScore);
  const ensembleScores = alignedData.map((d) => d.ensembleScore);
  const topsisRanks = alignedData.map((d) => d.topsisRank);
  const vikorRanks = alignedData.map((d) => d.vikorRank);
  const monteCarloRanks = alignedData.map((d) => d.monteCarloRank);

  return {
    topsisVikor: pearsonCorrelation(topsisScores, vikorScores),
    topsisEnsemble: pearsonCorrelation(topsisScores, ensembleScores),
    vikorEnsemble: pearsonCorrelation(vikorScores, ensembleScores),
    topsisMonteCarloRank: pearsonCorrelation(topsisRanks, monteCarloRanks),
    vikorMonteCarloRank: pearsonCorrelation(vikorRanks, monteCarloRanks),
  };
}

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Analytics Dashboard - Multi-Method Analysis" },
    {
      name: "description",
      content: "Comprehensive analysis dashboard with correlations",
    },
  ];
}

export default function Analytics() {
  const data = useLoaderData<LoaderData>();

  // Combine all results for comparison
  const combinedData = data.topsisResults
    .filter((t) => t.alternative_id !== "" && t.alternative_id != null) // Filter out empty alternative_id
    .map((t) => {
      const vikor = data.vikorResults.find(
        (v) => v.alternative_id === t.alternative_id
      );
      const monteCarlo = data.monteCarloResults.find(
        (m) => m.alternative_id === t.alternative_id
      );
      const ensemble = data.ensembleResults.find(
        (e) => e.alternative_id === t.alternative_id
      );

      return {
        sector: t.alternative_id,
        topsisScore: t.score,
        topsisRank: t.rank,
        vikorScore: vikor?.score || 0,
        vikorRank: vikor?.rank || 0,
        monteCarloScore: monteCarlo?.score || 0,
        monteCarloRank: monteCarlo?.rank || 0,
        ensembleScore: ensemble?.score || 0,
        ensembleRank: ensemble?.rank || 0,
      };
    });

  // Sort for rankings
  const rankingData = [...combinedData].sort((a, b) => a.topsisRank - b.topsisRank);

  return (
    <div className="min-h-screen bg-gray-50" style={{ backgroundColor: PKO_COLORS.grayLight }}>
      <Navigation />
      {/* Header */}
      <header className="shadow" style={{ backgroundColor: PKO_COLORS.navy }}>
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-white">
            Multi-Method Analysis Dashboard
          </h1>
          <p className="mt-1 text-sm" style={{ color: PKO_COLORS.grayLight }}>
            TOPSIS • VIKOR • Monte Carlo • Ensemble Methods
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        {/* Correlation Matrix Card */}
        <div className="px-4 py-6 sm:px-0">
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Method Correlation Analysis
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              <CorrelationCard
                title="TOPSIS vs VIKOR"
                correlation={data.correlations.topsisVikor}
                description="Score correlation between TOPSIS and VIKOR methods"
              />
              <CorrelationCard
                title="TOPSIS vs Ensemble"
                correlation={data.correlations.topsisEnsemble}
                description="Score correlation between TOPSIS and Ensemble"
              />
              <CorrelationCard
                title="VIKOR vs Ensemble"
                correlation={data.correlations.vikorEnsemble}
                description="Score correlation between VIKOR and Ensemble"
              />
              <CorrelationCard
                title="TOPSIS vs Monte Carlo (Rank)"
                correlation={data.correlations.topsisMonteCarloRank}
                description="Rank correlation with Monte Carlo simulation"
              />
              <CorrelationCard
                title="VIKOR vs Monte Carlo (Rank)"
                correlation={data.correlations.vikorMonteCarloRank}
                description="Rank correlation with Monte Carlo simulation"
              />
            </div>
          </div>

          {/* Score Comparison Chart */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Score Comparison Across Methods
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={rankingData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="sector" angle={-45} textAnchor="end" height={100} />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="topsisScore" fill={PKO_COLORS.navy} name="TOPSIS" />
                <Bar dataKey="vikorScore" fill={PKO_COLORS.red} name="VIKOR" />
                <Bar dataKey="monteCarloScore" fill={PKO_COLORS.gold} name="Monte Carlo" />
                <Bar dataKey="ensembleScore" fill="#4a90e2" name="Ensemble" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Ranking Comparison Chart */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Ranking Comparison Across Methods
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart data={rankingData}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="sector" angle={-45} textAnchor="end" height={100} />
                <YAxis reversed label={{ value: 'Rank', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="topsisRank"
                  stroke={PKO_COLORS.navy}
                  name="TOPSIS"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="vikorRank"
                  stroke={PKO_COLORS.red}
                  name="VIKOR"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="monteCarloRank"
                  stroke={PKO_COLORS.gold}
                  name="Monte Carlo"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="ensembleRank"
                  stroke="#4a90e2"
                  name="Ensemble"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* TOPSIS vs VIKOR Scatter Plot */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              TOPSIS vs VIKOR Score Scatter Plot
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="topsisScore"
                  name="TOPSIS Score"
                  label={{ value: 'TOPSIS Score', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  type="number"
                  dataKey="vikorScore"
                  name="VIKOR Score"
                  label={{ value: 'VIKOR Score', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter name="Sectors" data={combinedData} fill={PKO_COLORS.navy}>
                  {combinedData.map((entry, index) => (
                    <Cell key={`cell-${index}`} fill={CHART_COLORS[index % CHART_COLORS.length]} />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Detailed Rankings Table */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Detailed Rankings by Method
            </h2>
            <div className="overflow-x-auto">
              <table className="min-w-full divide-y divide-gray-200">
                <thead style={{ backgroundColor: PKO_COLORS.navy }}>
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">
                      Sector
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">
                      TOPSIS
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">
                      VIKOR
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">
                      Monte Carlo
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-white uppercase tracking-wider">
                      Ensemble
                    </th>
                  </tr>
                </thead>
                <tbody className="bg-white divide-y divide-gray-200">
                  {rankingData.map((row, idx) => (
                    <tr key={row.sector} className={idx % 2 === 0 ? "bg-white" : "bg-gray-50"}>
                      <td className="px-6 py-4 whitespace-nowrap text-sm font-medium text-gray-900">
                        {row.sector}
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        #{row.topsisRank} ({row.topsisScore.toFixed(3)})
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        #{row.vikorRank} ({row.vikorScore.toFixed(3)})
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        #{row.monteCarloRank} ({row.monteCarloScore.toFixed(3)})
                      </td>
                      <td className="px-6 py-4 whitespace-nowrap text-sm text-gray-500">
                        #{row.ensembleRank} ({row.ensembleScore.toFixed(3)})
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function CorrelationCard({
  title,
  correlation,
  description,
}: {
  title: string;
  correlation: number;
  description: string;
}) {
  const getCorrelationColor = (corr: number) => {
    if (corr > 0.7) return PKO_COLORS.navy;
    if (corr > 0.4) return PKO_COLORS.gold;
    return PKO_COLORS.red;
  };

  const getCorrelationText = (corr: number) => {
    if (corr > 0.7) return "Strong Positive";
    if (corr > 0.4) return "Moderate";
    if (corr > 0) return "Weak Positive";
    if (corr > -0.4) return "Weak Negative";
    return "Strong Negative";
  };

  return (
    <div className="bg-white border rounded-lg p-4 shadow-sm">
      <h3 className="text-lg font-semibold mb-2" style={{ color: PKO_COLORS.navy }}>
        {title}
      </h3>
      <div className="flex items-baseline mb-2">
        <span
          className="text-3xl font-bold"
          style={{ color: getCorrelationColor(correlation) }}
        >
          {correlation.toFixed(3)}
        </span>
        <span className="ml-2 text-sm text-gray-500">
          {getCorrelationText(correlation)}
        </span>
      </div>
      <p className="text-xs text-gray-600">{description}</p>
      <div className="mt-3 w-full bg-gray-200 rounded-full h-2">
        <div
          className="h-2 rounded-full"
          style={{
            width: `${Math.abs(correlation) * 100}%`,
            backgroundColor: getCorrelationColor(correlation),
          }}
        ></div>
      </div>
    </div>
  );
}
