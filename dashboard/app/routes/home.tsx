import type { Route } from "./+types/home";
import { DashboardCard } from "../components/DashboardCard";
import { Link } from "react-router";
import { ArrowRight, BarChart3, TrendingUp, Activity } from "lucide-react";
import { useCsvData } from "../hooks/useCsvData";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "PKO BP - Analizy Sektorowe" },
    { name: "description", content: "Dashboard analiz sektorowych" },
  ];
}

interface ResultData {
  alternative_id: string;
  ensemble_rank: number;
  ensemble_score: number;
}

export default function Home() {
  const { data: ensembleData } = useCsvData<ResultData>("/data/results_ensemble.csv");
  
  const topSector = ensembleData.find(d => Number(d.ensemble_rank) === 1);

  return (
    <div className="space-y-8">
      <div className="pko-gradient rounded-2xl p-8 text-white shadow-lg">
        <h1 className="text-3xl font-bold mb-2">Witaj w Panelu Analiz Sektorowych</h1>
        <p className="text-white/80 max-w-2xl">
          Kompleksowe narzędzie do analizy kondycji finansowej sektorów gospodarki z wykorzystaniem metod wielokryterialnych (MCDM) oraz sztucznej inteligencji.
        </p>
      </div>

      <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
        <DashboardCard title="Najlepszy Sektor (Ensemble)" className="border-l-4 border-l-pko-gold">
          <div className="flex flex-col h-full justify-between">
            <div>
              <div className="text-4xl font-bold text-pko-navy mb-1">
                {topSector ? topSector.alternative_id : "..."}
              </div>
              <div className="text-sm text-gray-500">
                Score: {topSector ? Number(topSector.ensemble_score).toFixed(4) : "..."}
              </div>
            </div>
            <Link to="/analysis" className="text-pko-red font-semibold text-sm flex items-center mt-4 hover:underline">
              Zobacz szczegóły <ArrowRight size={16} className="ml-1" />
            </Link>
          </div>
        </DashboardCard>

        <DashboardCard title="Metody Analizy">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-pko-navy/10 flex items-center justify-center text-pko-navy">
                <BarChart3 size={20} />
              </div>
              <div>
                <div className="font-semibold text-pko-navy">4 Metody</div>
                <div className="text-xs text-gray-500">Ensemble, Monte Carlo, TOPSIS, VIKOR</div>
              </div>
            </div>
            <Link to="/analysis" className="block w-full text-center py-2 rounded-md border border-pko-navy text-pko-navy font-medium hover:bg-pko-navy hover:text-white transition-colors">
              Przejdź do analizy
            </Link>
          </div>
        </DashboardCard>

        <DashboardCard title="Trendy i Korelacje">
          <div className="space-y-4">
            <div className="flex items-center gap-3">
              <div className="w-10 h-10 rounded-full bg-pko-red/10 flex items-center justify-center text-pko-red">
                <TrendingUp size={20} />
              </div>
              <div>
                <div className="font-semibold text-pko-navy">Analiza Czasowa</div>
                <div className="text-xs text-gray-500">Badanie korelacji wskaźników</div>
              </div>
            </div>
            <Link to="/correlations" className="block w-full text-center py-2 rounded-md bg-pko-red text-white font-medium hover:bg-red-700 transition-colors">
              Sprawdź trendy
            </Link>
          </div>
        </DashboardCard>
      </div>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        <div className="pko-card p-6 bg-white">
          <h3 className="text-lg font-bold text-pko-navy mb-4 flex items-center gap-2">
            <Activity size={20} />
            Ostatnie Aktualizacje
          </h3>
          <ul className="space-y-3">
            <li className="flex items-start gap-3 pb-3 border-b border-gray-100">
              <div className="w-2 h-2 mt-2 rounded-full bg-green-500"></div>
              <div>
                <div className="font-medium text-pko-navy">Zaktualizowano dane finansowe</div>
                <div className="text-xs text-gray-500">Dane za Q3 2025 zostały dodane do systemu.</div>
              </div>
            </li>
            <li className="flex items-start gap-3 pb-3 border-b border-gray-100">
              <div className="w-2 h-2 mt-2 rounded-full bg-blue-500"></div>
              <div>
                <div className="font-medium text-pko-navy">Nowy ranking VIKOR</div>
                <div className="text-xs text-gray-500">Przeliczono rankingi stabilności sektorów.</div>
              </div>
            </li>
          </ul>
        </div>
        
        <div className="pko-card p-6 bg-pko-navy text-white">
          <h3 className="text-lg font-bold mb-4">Informacje o Systemie</h3>
          <div className="space-y-4 text-sm text-white/80">
            <p>
              System wykorzystuje zaawansowane algorytmy do oceny kondycji finansowej przedsiębiorstw na podstawie danych GUS.
            </p>
            <div className="grid grid-cols-2 gap-4 mt-4">
              <div className="bg-white/10 p-3 rounded">
                <div className="text-2xl font-bold text-pko-gold">24</div>
                <div className="text-xs">Wskaźniki</div>
              </div>
              <div className="bg-white/10 p-3 rounded">
                <div className="text-2xl font-bold text-pko-gold">10+</div>
                <div className="text-xs">Sektorów</div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}
