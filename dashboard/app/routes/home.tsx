import type { Route } from "./+types/home";
import { DashboardCard } from "../components/DashboardCard";
import { Link } from "react-router";
import { BarChart3, TrendingUp, Activity, PieChart, GitCompare, Wallet, ArrowRight } from "lucide-react";

export function meta({}: Route.MetaArgs) {
  return [
    { title: "PKO BP - Analizy Sektorowe" },
    { name: "description", content: "Dashboard analiz sektorowych" },
  ];
}

export default function Home() {
  return (
    <div className="space-y-10">
      <div className="pko-gradient rounded-2xl p-8 text-white shadow-lg">
        <h1 className="text-3xl font-bold mb-2">Witaj w Panelu Analiz Sektorowych</h1>
        <p className="text-white/80 max-w-2xl">
          Kompleksowe narzędzie do analizy kondycji finansowej sektorów gospodarki z wykorzystaniem metod wielokryterialnych (MCDM) oraz sztucznej inteligencji.
        </p>
      </div>

      {/* Definicja Metodologii */}
      <section>
        <h2 className="text-xl font-bold text-pko-navy mb-4 flex items-center gap-2">
          <PieChart className="text-pko-red" />
          Definicja Metodologii
        </h2>
        <DashboardCard title="Metodologia Oceny">
          <div className="space-y-4">
            <p className="text-pko-navy/70">
              Szczegółowy opis procesu analitycznego, źródeł danych, inżynierii cech oraz definicje wskaźników podstawowych i zaawansowanych.
            </p>
            <Link to="/methodology" className="inline-flex items-center text-pko-red font-semibold hover:underline">
              Zobacz metodologię <ArrowRight size={16} className="ml-1" />
            </Link>
          </div>
        </DashboardCard>
      </section>

      {/* Metody zapewnienia jakości raportowania i analityki szczegółowej */}
      <section>
        <h2 className="text-xl font-bold text-pko-navy mb-4 flex items-center gap-2">
          <Activity className="text-pko-red" />
          Metody zapewnienia jakości raportowania i analityki szczegółowej
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-6">
          <DashboardCard title="Analiza Metod">
            <div className="flex flex-col h-full justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-pko-navy/10 flex items-center justify-center text-pko-navy">
                  <BarChart3 size={20} />
                </div>
                <div className="text-sm text-gray-600">
                  Porównanie wyników różnych metod MCDM (Ensemble, Monte Carlo).
                </div>
              </div>
              <Link to="/analysis" className="block w-full text-center py-2 rounded-md border border-pko-navy text-pko-navy font-medium hover:bg-pko-navy hover:text-white transition-colors">
                Przejdź do analizy
              </Link>
            </div>
          </DashboardCard>

          <DashboardCard title="TOPSIS vs VIKOR">
            <div className="flex flex-col h-full justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-pko-navy/10 flex items-center justify-center text-pko-navy">
                  <GitCompare size={20} />
                </div>
                <div className="text-sm text-gray-600">
                  Bezpośrednie porównanie rankingów generowanych przez metody TOPSIS i VIKOR.
                </div>
              </div>
              <Link to="/comparison" className="block w-full text-center py-2 rounded-md border border-pko-navy text-pko-navy font-medium hover:bg-pko-navy hover:text-white transition-colors">
                Porównaj metody
              </Link>
            </div>
          </DashboardCard>

          <DashboardCard title="Korelacje">
            <div className="flex flex-col h-full justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-pko-navy/10 flex items-center justify-center text-pko-navy">
                  <Activity size={20} />
                </div>
                <div className="text-sm text-gray-600">
                  Badanie korelacji między wskaźnikami finansowymi a wynikami rankingów.
                </div>
              </div>
              <Link to="/correlations" className="block w-full text-center py-2 rounded-md border border-pko-navy text-pko-navy font-medium hover:bg-pko-navy hover:text-white transition-colors">
                Analizuj korelacje
              </Link>
            </div>
          </DashboardCard>
        </div>
      </section>

      {/* Raportowanie */}
      <section>
        <h2 className="text-xl font-bold text-pko-navy mb-4 flex items-center gap-2">
          <TrendingUp className="text-pko-red" />
          Raportowanie
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-6">
          <DashboardCard title="Trendy">
            <div className="flex flex-col h-full justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-pko-red/10 flex items-center justify-center text-pko-red">
                  <TrendingUp size={20} />
                </div>
                <div className="text-sm text-gray-600">
                  Analiza trendów czasowych dla poszczególnych sektorów i wskaźników.
                </div>
              </div>
              <Link to="/trends" className="block w-full text-center py-2 rounded-md bg-pko-red text-white font-medium hover:bg-red-700 transition-colors">
                Zobacz trendy
              </Link>
            </div>
          </DashboardCard>

          <DashboardCard title="Potrzeby Pożyczkowe">
            <div className="flex flex-col h-full justify-between gap-4">
              <div className="flex items-center gap-3">
                <div className="w-10 h-10 rounded-full bg-pko-red/10 flex items-center justify-center text-pko-red">
                  <Wallet size={20} />
                </div>
                <div className="text-sm text-gray-600">
                  Analiza potrzeb pożyczkowych w relacji do oceny kredytowej i potencjału rozwoju.
                </div>
              </div>
              <Link to="/loan-needs" className="block w-full text-center py-2 rounded-md bg-pko-red text-white font-medium hover:bg-red-700 transition-colors">
                Analiza potrzeb
              </Link>
            </div>
          </DashboardCard>
        </div>
      </section>
    </div>
  );
}
