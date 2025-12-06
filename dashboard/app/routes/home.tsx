import type { Route } from "./+types/home";
import { Welcome } from "../welcome/welcome";
import { Link } from "react-router";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "HackNation 2025 - Financial Analytics" },
    { name: "description", content: "Multi-method analysis dashboard for financial data" },
  ];
}

export default function Home() {
  return (
    <div className="min-h-screen bg-gray-50">
      <div className="max-w-4xl mx-auto py-16 px-4">
        <div className="text-center mb-12">
          <h1 className="text-4xl font-bold text-gray-900 mb-4">
            Financial Analytics Dashboard
          </h1>
          <p className="text-xl text-gray-600">
            Multi-method analysis and correlation visualization
          </p>
        </div>
        
        <div className="bg-white rounded-lg shadow-md p-8">
          <h2 className="text-2xl font-semibold mb-6 text-gray-800">
            Available Dashboards
          </h2>
          
          <div className="space-y-4">
            <Link
              to="/analytics"
              className="block p-6 bg-blue-50 hover:bg-blue-100 rounded-lg transition-colors border border-blue-200"
            >
              <h3 className="text-xl font-semibold text-blue-900 mb-2">
                📊 Multi-Method Analysis
              </h3>
              <p className="text-blue-700">
                Comprehensive analysis dashboard featuring TOPSIS, VIKOR, Monte Carlo, 
                and Ensemble methods with correlation analysis and interactive visualizations.
              </p>
              <div className="mt-3 text-blue-600 font-medium">
                View Analytics →
              </div>
            </Link>
            
            <Link
              to="/financial-indicators"
              className="block p-6 bg-green-50 hover:bg-green-100 rounded-lg transition-colors border border-green-200"
            >
              <h3 className="text-xl font-semibold text-green-900 mb-2">
                📈 Financial Indicators Analysis
              </h3>
              <p className="text-green-700">
                Time series analysis (2005-2024), correlation matrix, profitability ratios, 
                liquidity metrics, and comprehensive financial trend visualizations.
              </p>
              <div className="mt-3 text-green-600 font-medium">
                View Financial Analysis →
              </div>
            </Link>
          </div>
        </div>
        
        <div className="mt-8">
          <Welcome />
        </div>
      </div>
    </div>
  );
}
