import { useState } from "react";
import { indicators } from "../data/indicators";
import { DashboardCard } from "../components/DashboardCard";
import { ArrowRight, Database, Server, BrainCircuit, ChevronDown, ChevronUp } from "lucide-react";

export default function Methodology() {
  const basicIndicators = indicators.filter(i => !i.formula);
  const advancedIndicators = indicators.filter(i => i.formula);
  
  const [isBasicExpanded, setIsBasicExpanded] = useState(false);
  const [isAdvancedExpanded, setIsAdvancedExpanded] = useState(true);

  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-pko-navy mb-2">Metodologia Oceny</h2>
        <p className="text-pko-navy/70">
          Kierunek oceny zmiennych w analizie metodami wielokryterialnymi z perspektywy przyznawania kredytów.
        </p>
      </div>

      <div className="grid grid-cols-1 gap-6">
        <DashboardCard title="Proces Analityczny">
          <div className="space-y-8">
            
            {/* Identyfikacja źródeł */}
            <div>
              <h3 className="text-lg font-semibold text-pko-navy mb-4 flex items-center gap-2">
                <Database className="w-5 h-5 text-pko-red" />
                Identyfikacja źródeł
              </h3>
              <ul className="list-disc list-inside space-y-2 text-pko-navy/80 ml-2">
                <li>
                  <a href="https://stat.gov.pl/obszary-tematyczne/podmioty-gospodarcze-wyniki-finansowe/zmiany-strukturalne-grup-podmiotow/kwartalna-informacja-o-podmiotach-gospodarki-narodowej-w-rejestrze-regon-rok-2025,7,15.html" target="_blank" rel="noopener noreferrer" className="hover:text-pko-red hover:underline">
                    Kwartalna informacja o podmiotach gospodarki narodowej w rejestrze REGON rok 2025 (GUS)
                  </a>
                </li>
                <li>
                  <a href="https://stat.gov.pl/obszary-tematyczne/podmioty-gospodarcze-wyniki-finansowe/zmiany-strukturalne-grup-podmiotow/miesieczna-informacja-o-podmiotach-gospodarki-narodowej-w-rejestrze-regon-pazdziernik-2025,4,103.html" target="_blank" rel="noopener noreferrer" className="hover:text-pko-red hover:underline">
                    Miesięczna informacja o podmiotach gospodarki narodowej w rejestrze REGON - październik 2025 (GUS)
                  </a>
                </li>
                <li>Krajowy Rejestr Zadłużonych (dane dostarczone)</li>
                <li>Wskaźniki finansowe (dane dostarczone)</li>
              </ul>
            </div>

            {/* Data Engineering */}
            <div>
              <h3 className="text-lg font-semibold text-pko-navy mb-4 flex items-center gap-2">
                <Server className="w-5 h-5 text-pko-red" />
                Data Engineering
              </h3>
              <div className="flex flex-col md:flex-row gap-4 items-start md:items-center text-sm">
                <div className="bg-gray-100 p-3 rounded-lg border border-gray-200 text-center min-w-[140px] flex-1">
                  Wczytanie plików źródłowych
                </div>
                <ArrowRight className="hidden md:block text-gray-400 shrink-0" />
                <div className="bg-gray-100 p-3 rounded-lg border border-gray-200 text-center min-w-[140px] flex-1">
                  Mapping do PKD 2025
                </div>
                <ArrowRight className="hidden md:block text-gray-400 shrink-0" />
                <div className="bg-gray-100 p-3 rounded-lg border border-gray-200 text-center min-w-[140px] flex-1">
                  Odrzucenie incydentów <span className="text-xs text-gray-500 block mt-1">(błędne kody PKD: 2520Z, 4961z, 1312A, 4749Z, 6839Z, 6819Z, Zielon)</span>
                </div>
                <ArrowRight className="hidden md:block text-gray-400 shrink-0" />
                <div className="bg-gray-100 p-3 rounded-lg border border-gray-200 text-center min-w-[140px] flex-1">
                  Transpozycje do postaci tabeli faktowej
                </div>
                <ArrowRight className="hidden md:block text-gray-400 shrink-0" />
                <div className="bg-gray-100 p-3 rounded-lg border border-gray-200 text-center min-w-[140px] flex-1">
                  Wyznaczenie wskaźników i manualne dopasowanie kryteriów oceny
                </div>
              </div>
            </div>

            {/* Data Science */}
            <div>
              <h3 className="text-lg font-semibold text-pko-navy mb-4 flex items-center gap-2">
                <BrainCircuit className="w-5 h-5 text-pko-red" />
                Data Science
              </h3>
              <div className="flex flex-col md:flex-row gap-4 items-start md:items-center text-sm">
                <div className="bg-gray-100 p-3 rounded-lg border border-gray-200 text-center min-w-[140px] flex-1">
                  Tabela faktowa
                </div>
                <ArrowRight className="hidden md:block text-gray-400 shrink-0" />
                <div className="bg-gray-100 p-3 rounded-lg border border-gray-200 text-center min-w-[140px] flex-1">
                  Wykonanie eksperymentów metodami wielokryterialnymi
                </div>
                <ArrowRight className="hidden md:block text-gray-400 shrink-0" />
                <div className="bg-gray-100 p-3 rounded-lg border border-gray-200 text-center min-w-[140px] flex-1">
                  Poszukiwanie korelacji z danymi zewnętrznymi nieraportowanymi na poziomie PKD
                </div>
              </div>
            </div>

          </div>
        </DashboardCard>
      </div>

      <DashboardCard title="Słownik Wskaźników i Kryteria Oceny">
        <div className="space-y-8">
          {/* Basic Indicators */}
          <div className="border-b border-gray-100 pb-4 last:border-0">
            <button 
              onClick={() => setIsBasicExpanded(!isBasicExpanded)}
              className="w-full flex items-center justify-between px-6 py-2 hover:bg-gray-50 transition-colors rounded-lg group"
            >
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-semibold text-pko-navy">Podstawowe Wskaźniki Finansowe</h3>
                <span className="bg-pko-gray-light text-pko-navy text-xs font-medium px-2.5 py-0.5 rounded-full border border-pko-gray-medium/20">
                  {basicIndicators.length}
                </span>
              </div>
              {isBasicExpanded ? <ChevronUp className="text-pko-navy/50 group-hover:text-pko-navy" /> : <ChevronDown className="text-pko-navy/50 group-hover:text-pko-navy" />}
            </button>
            
            {isBasicExpanded && (
              <div className="overflow-x-auto mt-4">
                <table className="w-full text-sm text-left text-pko-navy">
                  <thead className="text-xs text-pko-navy uppercase bg-gray-50 border-b border-gray-200">
                    <tr>
                      <th scope="col" className="px-6 py-3 font-bold">Kod</th>
                      <th scope="col" className="px-6 py-3 font-bold">Nazwa Wskaźnika</th>
                      <th scope="col" className="px-6 py-3 font-bold">Preferencja (Ocena)</th>
                      <th scope="col" className="px-6 py-3 font-bold">Uzasadnienie Kredytowe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {basicIndicators.map((indicator) => (
                      <tr key={indicator.id} className="bg-white border-b border-gray-100 hover:bg-gray-50">
                        <td className="px-6 py-4 font-medium text-pko-navy/80 whitespace-nowrap">
                          {indicator.code}
                        </td>
                        <td className="px-6 py-4 font-medium">
                          {indicator.name}
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                            indicator.preference.toLowerCase().includes("wyższe") || indicator.preference.toLowerCase().includes("wyższy") || indicator.preference.toLowerCase().includes("więcej")
                              ? "bg-green-100 text-green-800"
                              : indicator.preference.toLowerCase().includes("niższe") || indicator.preference.toLowerCase().includes("niższy") || indicator.preference.toLowerCase().includes("minimalizować")
                              ? "bg-blue-100 text-blue-800"
                              : "bg-gray-100 text-gray-800"
                          }`}>
                            {indicator.preference}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-pko-navy/70">
                          {indicator.justification}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>

          {/* Advanced Indicators */}
          <div className="border-b border-gray-100 pb-4 last:border-0">
            <button 
              onClick={() => setIsAdvancedExpanded(!isAdvancedExpanded)}
              className="w-full flex items-center justify-between px-6 py-2 hover:bg-gray-50 transition-colors rounded-lg group"
            >
              <div className="flex items-center gap-3">
                <h3 className="text-lg font-semibold text-pko-navy">Zaawansowane Wskaźniki i Relacje</h3>
                <span className="bg-pko-gray-light text-pko-navy text-xs font-medium px-2.5 py-0.5 rounded-full border border-pko-gray-medium/20">
                  {advancedIndicators.length}
                </span>
              </div>
              {isAdvancedExpanded ? <ChevronUp className="text-pko-navy/50 group-hover:text-pko-navy" /> : <ChevronDown className="text-pko-navy/50 group-hover:text-pko-navy" />}
            </button>

            {isAdvancedExpanded && (
              <div className="overflow-x-auto mt-4">
                <table className="w-full text-sm text-left text-pko-navy">
                  <thead className="text-xs text-pko-navy uppercase bg-gray-50 border-b border-gray-200">
                    <tr>
                      <th scope="col" className="px-6 py-3 font-bold">Kod</th>
                      <th scope="col" className="px-6 py-3 font-bold">Nazwa Wskaźnika</th>
                      <th scope="col" className="px-6 py-3 font-bold">Zastosowanie</th>
                      <th scope="col" className="px-6 py-3 font-bold">Preferencja (Ocena)</th>
                      <th scope="col" className="px-6 py-3 font-bold">Uzasadnienie Kredytowe</th>
                    </tr>
                  </thead>
                  <tbody>
                    {advancedIndicators.map((indicator) => (
                      <tr key={indicator.id} className="bg-white border-b border-gray-100 hover:bg-gray-50">
                        <td className="px-6 py-4 font-medium text-pko-navy/80 whitespace-nowrap">
                          {indicator.code}
                        </td>
                        <td className="px-6 py-4 font-medium">
                          <div>{indicator.name}</div>
                          {indicator.formula && (
                            <div className="group relative inline-block cursor-help mt-1">
                              <div className="text-xs text-pko-navy/60 font-mono bg-gray-100 px-2 py-0.5 rounded border border-gray-200 border-b border-dotted border-b-pko-navy/40">
                                {indicator.formula}
                              </div>
                              <div className="invisible group-hover:visible absolute left-0 bottom-full mb-2 z-50 bg-pko-navy text-white text-xs rounded py-2 px-3 shadow-xl font-sans whitespace-nowrap">
                                {indicator.formula.split(/([a-zA-Z0-9_]+)/).map((part, i) => {
                                  const found = indicators.find(ind => ind.code === part);
                                  return found ? found.name : part;
                                })}
                                <div className="absolute left-4 top-full w-0 h-0 border-l-[6px] border-l-transparent border-r-[6px] border-r-transparent border-t-[6px] border-t-pko-navy"></div>
                              </div>
                            </div>
                          )}
                        </td>
                        <td className="px-6 py-4">
                          <div className="flex flex-wrap gap-1">
                            {indicator.datasets?.map(ds => (
                              <span key={ds} className={`px-2 py-0.5 rounded text-[10px] font-bold uppercase tracking-wider border ${
                                ds === 'credit' ? 'bg-red-50 text-red-700 border-red-200' :
                                ds === 'development' ? 'bg-amber-50 text-amber-700 border-amber-200' :
                                ds === 'effectivity' ? 'bg-emerald-50 text-emerald-700 border-emerald-200' :
                                'bg-gray-50 text-gray-600 border-gray-200'
                              }`}>
                                {ds === 'credit' ? 'Ryzyko' : ds === 'development' ? 'Rozwój' : ds === 'effectivity' ? 'Efektywność' : ds}
                              </span>
                            ))}
                          </div>
                        </td>
                        <td className="px-6 py-4 whitespace-nowrap">
                          <span className={`px-2 py-1 rounded-full text-xs font-semibold ${
                            indicator.preference.toLowerCase().includes("wyższe") || indicator.preference.toLowerCase().includes("wyższy") || indicator.preference.toLowerCase().includes("więcej")
                              ? "bg-green-100 text-green-800"
                              : indicator.preference.toLowerCase().includes("niższe") || indicator.preference.toLowerCase().includes("niższy") || indicator.preference.toLowerCase().includes("minimalizować")
                              ? "bg-blue-100 text-blue-800"
                              : "bg-gray-100 text-gray-800"
                          }`}>
                            {indicator.preference}
                          </span>
                        </td>
                        <td className="px-6 py-4 text-pko-navy/70">
                          {indicator.justification}
                        </td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            )}
          </div>
        </div>
      </DashboardCard>
    </div>
  );
}
