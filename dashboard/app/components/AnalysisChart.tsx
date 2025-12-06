import {
  BarChart,
  Bar,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
  LineChart,
  Line,
} from "recharts";

interface AnalysisChartProps {
  data: any[];
  type?: "bar" | "line";
  xKey: string;
  dataKeys: { key: string; color: string; name?: string }[];
  height?: number;
}

export function AnalysisChart({
  data,
  type = "bar",
  xKey,
  dataKeys,
  height = 400,
}: AnalysisChartProps) {
  const ChartComponent = type === "line" ? LineChart : BarChart;

  if (!data || data.length === 0) {
    return (
      <div style={{ width: "100%", height, display: "flex", alignItems: "center", justifyContent: "center" }} className="text-gray-400 text-sm">
        Brak danych do wyświetlenia wykresu
      </div>
    );
  }

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer>
        <ChartComponent
          data={data}
          margin={{
            top: 20,
            right: 30,
            left: 20,
            bottom: 5,
          }}
        >
          <CartesianGrid strokeDasharray="3 3" stroke="#e5e7eb" vertical={false} />
          <XAxis 
            dataKey={xKey} 
            tick={{ fill: "#1a2f3a", fontSize: 12 }}
            axisLine={{ stroke: "#d0d0d0" }}
            tickLine={false}
          />
          <YAxis 
            tick={{ fill: "#1a2f3a", fontSize: 12 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            contentStyle={{
              backgroundColor: "#fff",
              border: "1px solid #d0d0d0",
              borderRadius: "4px",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
            }}
          />
          <Legend wrapperStyle={{ paddingTop: "20px" }} />
          {dataKeys.map((dk) => (
            type === "line" ? (
              <Line
                key={dk.key}
                type="monotone"
                dataKey={dk.key}
                stroke={dk.color}
                name={dk.name || dk.key}
                strokeWidth={2}
                dot={{ r: 4, fill: dk.color }}
                activeDot={{ r: 6 }}
              />
            ) : (
              <Bar
                key={dk.key}
                dataKey={dk.key}
                fill={dk.color}
                name={dk.name || dk.key}
                radius={[4, 4, 0, 0]}
              />
            )
          ))}
        </ChartComponent>
      </ResponsiveContainer>
    </div>
  );
}
