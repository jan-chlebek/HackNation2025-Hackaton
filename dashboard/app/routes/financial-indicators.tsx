import type { Route } from "./+types/financial-indicators";
import { useLoaderData } from "react-router";
import Papa from "papaparse";
import { Navigation } from "../components/Navigation";
import {
  LineChart,
  Line,
  BarChart,
  Bar,
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
];

// Indicator codes for data extraction
const INDICATOR_CODES = {
  TOTAL_REVENUE: "GS Przychody ogółem",
  NET_PROFIT: "NP Wynik finansowy netto",
  OPERATING_RESULT: "OP Wynik na działalności operacyjnej",
  CASH_FLOW: "CF Nadwyżka finansowa",
  TOTAL_COSTS: "TC Koszty ogółem",
  SALES_RESULT: "POS Wynik na sprzedaży",
  NET_REVENUE: "PNPM Przychody netto",
  WORKING_CAPITAL: "NWC Kapitał obrotowy",
  CASH: "C Środki pieniężne",
  SHORT_TERM_LIABILITIES: "STL Zobowiązania krótkoterminowe",
  INVENTORY: "INV Zapasy",
} as const;

interface FinancialIndicator {
  PKD: string;
  NAZWA_PKD: string;
  WSKAZNIK: string;
  [year: string]: string;
}

interface TimeSeriesData {
  year: string;
  value: number;
}

interface CorrelationData {
  indicator1: string;
  indicator2: string;
  correlation: number;
}

interface LoaderData {
  overallTrends: {
    revenue: TimeSeriesData[];
    netProfit: TimeSeriesData[];
    operatingResult: TimeSeriesData[];
    cashFlow: TimeSeriesData[];
  };
  correlations: CorrelationData[];
  profitabilityMetrics: {
    year: string;
    netProfitMargin: number;
    operatingMargin: number;
    returnOnSales: number;
  }[];
  liquidityMetrics: {
    year: string;
    currentRatio: number;
    quickRatio: number;
  }[];
}

export async function loader({ request }: Route.LoaderArgs) {
  // Load the full financial indicators CSV
  const response = await fetch(new URL("/data/wsk_fin.csv", request.url));
  const csvText = await response.text();

  // Parse CSV data
  const parsed = Papa.parse<FinancialIndicator>(csvText, {
    header: true,
    delimiter: ";",
    skipEmptyLines: true,
  });

  // Filter for overall (OGÓŁEM) data only
  const overallData = parsed.data.filter(
    (row) => row.PKD === "OG" && row.NAZWA_PKD === "OGÓŁEM"
  );

  // Extract years from columns (2005-2024)
  const years = Object.keys(overallData[0] || {}).filter((key) =>
    /^\d{4}$/.test(key)
  );

  // Helper function to parse financial values
  const parseFinancialValue = (value: string | undefined): number => {
    if (!value) return 0;
    const parsed = parseFloat(value.replace(/\s/g, "").replace(",", "."));
    return isNaN(parsed) ? 0 : parsed;
  };

  // Helper function to extract time series data
  const extractTimeSeries = (indicatorCode: string): TimeSeriesData[] => {
    const row = overallData.find((r) => r.WSKAZNIK.startsWith(indicatorCode));
    if (!row) return [];

    return years
      .map((year) => ({
        year,
        value: parseFinancialValue(row[year]),
      }))
      .filter((d) => d.value !== 0);
  };

  // Get key indicators
  const revenue = extractTimeSeries(INDICATOR_CODES.TOTAL_REVENUE);
  const netProfit = extractTimeSeries(INDICATOR_CODES.NET_PROFIT);
  const operatingResult = extractTimeSeries(INDICATOR_CODES.OPERATING_RESULT);
  const cashFlow = extractTimeSeries(INDICATOR_CODES.CASH_FLOW);
  const totalCosts = extractTimeSeries(INDICATOR_CODES.TOTAL_COSTS);
  const salesResult = extractTimeSeries(INDICATOR_CODES.SALES_RESULT);
  const netRevenue = extractTimeSeries(INDICATOR_CODES.NET_REVENUE);
  
  // Additional metrics for ratios
  const workingCapital = extractTimeSeries(INDICATOR_CODES.WORKING_CAPITAL);
  const cash = extractTimeSeries(INDICATOR_CODES.CASH);
  const shortTermLiabilities = extractTimeSeries(INDICATOR_CODES.SHORT_TERM_LIABILITIES);
  const inventory = extractTimeSeries(INDICATOR_CODES.INVENTORY);

  // Calculate correlations
  const calculatePearson = (x: number[], y: number[]): number => {
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

  const correlations: CorrelationData[] = [
    {
      indicator1: "Net Profit",
      indicator2: "Revenue",
      correlation: calculatePearson(
        netProfit.map((d) => d.value),
        revenue.map((d) => d.value)
      ),
    },
    {
      indicator1: "Operating Result",
      indicator2: "Revenue",
      correlation: calculatePearson(
        operatingResult.map((d) => d.value),
        revenue.map((d) => d.value)
      ),
    },
    {
      indicator1: "Cash Flow",
      indicator2: "Net Profit",
      correlation: calculatePearson(
        cashFlow.map((d) => d.value),
        netProfit.map((d) => d.value)
      ),
    },
    {
      indicator1: "Operating Result",
      indicator2: "Sales Result",
      correlation: calculatePearson(
        operatingResult.map((d) => d.value),
        salesResult.map((d) => d.value)
      ),
    },
    {
      indicator1: "Net Profit",
      indicator2: "Operating Result",
      correlation: calculatePearson(
        netProfit.map((d) => d.value),
        operatingResult.map((d) => d.value)
      ),
    },
    {
      indicator1: "Total Costs",
      indicator2: "Revenue",
      correlation: calculatePearson(
        totalCosts.map((d) => d.value),
        revenue.map((d) => d.value)
      ),
    },
  ];

  // Calculate profitability metrics
  const profitabilityMetrics = years.map((year) => {
    const rev = revenue.find((d) => d.year === year)?.value || 1;
    const profit = netProfit.find((d) => d.year === year)?.value || 0;
    const opResult = operatingResult.find((d) => d.year === year)?.value || 0;
    const salesRes = salesResult.find((d) => d.year === year)?.value || 0;

    return {
      year,
      netProfitMargin: (profit / rev) * 100,
      operatingMargin: (opResult / rev) * 100,
      returnOnSales: (salesRes / rev) * 100,
    };
  }).filter((d) => d.netProfitMargin && !isNaN(d.netProfitMargin));

  // Calculate liquidity metrics
  const liquidityMetrics = years.map((year) => {
    const wc = workingCapital.find((d) => d.year === year)?.value || 0;
    const stl = shortTermLiabilities.find((d) => d.year === year)?.value || 1;
    const inv = inventory.find((d) => d.year === year)?.value || 0;

    // Working Capital = Current Assets - Current Liabilities
    // Therefore: Current Assets = Working Capital + Current Liabilities
    const currentAssets = wc + stl;
    const quickAssets = currentAssets - inv;

    return {
      year,
      currentRatio: currentAssets / stl,
      quickRatio: quickAssets / stl,
    };
  }).filter((d) => d.currentRatio && !isNaN(d.currentRatio) && isFinite(d.currentRatio));

  return {
    overallTrends: {
      revenue,
      netProfit,
      operatingResult,
      cashFlow,
    },
    correlations,
    profitabilityMetrics,
    liquidityMetrics,
  };
}

export function meta({}: Route.MetaArgs) {
  return [
    { title: "Financial Indicators - Time Series Analysis" },
    { name: "description", content: "Financial indicators analysis with correlations" },
  ];
}

export default function FinancialIndicators() {
  const data = useLoaderData<LoaderData>();

  return (
    <div className="min-h-screen bg-gray-50" style={{ backgroundColor: PKO_COLORS.grayLight }}>
      <Navigation />
      {/* Header */}
      <header className="shadow" style={{ backgroundColor: PKO_COLORS.navy }}>
        <div className="max-w-7xl mx-auto py-6 px-4 sm:px-6 lg:px-8">
          <h1 className="text-3xl font-bold text-white">
            Financial Indicators Analysis
          </h1>
          <p className="mt-1 text-sm" style={{ color: PKO_COLORS.grayLight }}>
            Time Series Trends (2005-2024) • Correlation Analysis • Financial Ratios
          </p>
        </div>
      </header>

      <main className="max-w-7xl mx-auto py-6 sm:px-6 lg:px-8">
        <div className="px-4 py-6 sm:px-0">
          
          {/* Correlation Analysis Section */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Financial Indicators Correlation Matrix
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {data.correlations.map((corr, idx) => (
                <CorrelationCard
                  key={idx}
                  indicator1={corr.indicator1}
                  indicator2={corr.indicator2}
                  correlation={corr.correlation}
                />
              ))}
            </div>
          </div>

          {/* Revenue & Profit Trends */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Revenue and Profitability Trends (2005-2024)
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <LineChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  dataKey="year"
                  type="category"
                  allowDuplicatedCategory={false}
                />
                <YAxis
                  yAxisId="left"
                  label={{ value: 'Million PLN', angle: -90, position: 'insideLeft' }}
                />
                <YAxis
                  yAxisId="right"
                  orientation="right"
                  label={{ value: 'Million PLN', angle: 90, position: 'insideRight' }}
                />
                <Tooltip />
                <Legend />
                <Line
                  yAxisId="left"
                  data={data.overallTrends.revenue}
                  type="monotone"
                  dataKey="value"
                  stroke={PKO_COLORS.navy}
                  name="Total Revenue"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                />
                <Line
                  yAxisId="right"
                  data={data.overallTrends.netProfit}
                  type="monotone"
                  dataKey="value"
                  stroke={PKO_COLORS.red}
                  name="Net Profit"
                  strokeWidth={3}
                  dot={{ r: 4 }}
                />
                <Line
                  yAxisId="right"
                  data={data.overallTrends.operatingResult}
                  type="monotone"
                  dataKey="value"
                  stroke={PKO_COLORS.gold}
                  name="Operating Result"
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Cash Flow Trends */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Cash Flow Analysis (2005-2024)
            </h2>
            <ResponsiveContainer width="100%" height={350}>
              <BarChart data={data.overallTrends.cashFlow}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis label={{ value: 'Million PLN', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Bar dataKey="value" name="Cash Flow (Surplus)" fill={PKO_COLORS.navy}>
                  {data.overallTrends.cashFlow.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={entry.value > 0 ? PKO_COLORS.navy : PKO_COLORS.red}
                    />
                  ))}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Profitability Metrics */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Profitability Ratios Over Time
            </h2>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={data.profitabilityMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis label={{ value: 'Percentage (%)', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="netProfitMargin"
                  stroke={PKO_COLORS.navy}
                  name="Net Profit Margin"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="operatingMargin"
                  stroke={PKO_COLORS.red}
                  name="Operating Margin"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="returnOnSales"
                  stroke={PKO_COLORS.gold}
                  name="Return on Sales"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
          </div>

          {/* Liquidity Metrics */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Liquidity Ratios Analysis
            </h2>
            <ResponsiveContainer width="100%" height={350}>
              <LineChart data={data.liquidityMetrics}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="year" />
                <YAxis label={{ value: 'Ratio', angle: -90, position: 'insideLeft' }} />
                <Tooltip />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="currentRatio"
                  stroke={PKO_COLORS.navy}
                  name="Current Ratio"
                  strokeWidth={2}
                />
                <Line
                  type="monotone"
                  dataKey="quickRatio"
                  stroke={PKO_COLORS.red}
                  name="Quick Ratio"
                  strokeWidth={2}
                />
              </LineChart>
            </ResponsiveContainer>
            <div className="mt-4 text-sm text-gray-600">
              <p><strong>Current Ratio:</strong> Current Assets / Short-term Liabilities (optimal: 1.5-2.0)</p>
              <p><strong>Quick Ratio:</strong> (Current Assets - Inventory) / Short-term Liabilities (optimal: 1.0-1.5)</p>
            </div>
          </div>

          {/* Revenue vs Profit Scatter */}
          <div className="bg-white rounded-lg shadow-md p-6 mb-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Revenue vs Net Profit Correlation
            </h2>
            <ResponsiveContainer width="100%" height={400}>
              <ScatterChart>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis
                  type="number"
                  dataKey="revenue"
                  name="Revenue"
                  label={{ value: 'Revenue (Million PLN)', position: 'insideBottom', offset: -5 }}
                />
                <YAxis
                  type="number"
                  dataKey="netProfit"
                  name="Net Profit"
                  label={{ value: 'Net Profit (Million PLN)', angle: -90, position: 'insideLeft' }}
                />
                <Tooltip cursor={{ strokeDasharray: '3 3' }} />
                <Legend />
                <Scatter
                  name="Years 2005-2024"
                  data={data.overallTrends.revenue.map((rev, idx) => ({
                    revenue: rev.value,
                    netProfit: data.overallTrends.netProfit[idx]?.value || 0,
                    year: rev.year,
                  }))}
                  fill={PKO_COLORS.navy}
                >
                  {data.overallTrends.revenue.map((entry, index) => (
                    <Cell
                      key={`cell-${index}`}
                      fill={CHART_COLORS[index % CHART_COLORS.length]}
                    />
                  ))}
                </Scatter>
              </ScatterChart>
            </ResponsiveContainer>
          </div>

          {/* Key Statistics */}
          <div className="bg-white rounded-lg shadow-md p-6">
            <h2 className="text-2xl font-semibold mb-4" style={{ color: PKO_COLORS.navy }}>
              Key Financial Statistics (2005-2024)
            </h2>
            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
              <StatCard
                title="Average Revenue Growth"
                value={calculateGrowthRate(data.overallTrends.revenue)}
                unit="%/year"
                color={PKO_COLORS.navy}
              />
              <StatCard
                title="Average Profit Margin"
                value={calculateAverageMargin(data.profitabilityMetrics, 'netProfitMargin')}
                unit="%"
                color={PKO_COLORS.red}
              />
              <StatCard
                title="Latest Current Ratio"
                value={data.liquidityMetrics[data.liquidityMetrics.length - 1]?.currentRatio || 0}
                unit=""
                color={PKO_COLORS.gold}
              />
              <StatCard
                title="Cash Flow Volatility"
                value={calculateVolatility(data.overallTrends.cashFlow)}
                unit="%"
                color="#4a90e2"
              />
            </div>
          </div>
        </div>
      </main>
    </div>
  );
}

function CorrelationCard({
  indicator1,
  indicator2,
  correlation,
}: {
  indicator1: string;
  indicator2: string;
  correlation: number;
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
      <h3 className="text-sm font-semibold mb-2" style={{ color: PKO_COLORS.navy }}>
        {indicator1} ↔ {indicator2}
      </h3>
      <div className="flex items-baseline mb-2">
        <span
          className="text-2xl font-bold"
          style={{ color: getCorrelationColor(correlation) }}
        >
          {correlation.toFixed(3)}
        </span>
        <span className="ml-2 text-xs text-gray-500">
          {getCorrelationText(correlation)}
        </span>
      </div>
      <div className="w-full bg-gray-200 rounded-full h-2">
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

function StatCard({
  title,
  value,
  unit,
  color,
}: {
  title: string;
  value: number;
  unit: string;
  color: string;
}) {
  return (
    <div className="bg-white border rounded-lg p-4 shadow-sm">
      <h3 className="text-sm font-medium text-gray-600 mb-2">{title}</h3>
      <div className="flex items-baseline">
        <span className="text-3xl font-bold" style={{ color }}>
          {value.toFixed(2)}
        </span>
        <span className="ml-2 text-sm text-gray-500">{unit}</span>
      </div>
    </div>
  );
}

function calculateGrowthRate(data: TimeSeriesData[]): number {
  if (data.length < 2) return 0;
  const first = data[0].value;
  const last = data[data.length - 1].value;
  const years = data.length - 1;
  return (Math.pow(last / first, 1 / years) - 1) * 100;
}

function calculateAverageMargin(data: any[], field: string): number {
  if (data.length === 0) return 0;
  const sum = data.reduce((acc, d) => acc + (d[field] || 0), 0);
  return sum / data.length;
}

function calculateVolatility(data: TimeSeriesData[]): number {
  if (data.length < 2) return 0;
  const mean = data.reduce((acc, d) => acc + d.value, 0) / data.length;
  const variance = data.reduce((acc, d) => acc + Math.pow(d.value - mean, 2), 0) / data.length;
  const stdDev = Math.sqrt(variance);
  return (stdDev / mean) * 100;
}
