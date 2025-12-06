import { indicators } from "../data/indicators";
import { DashboardCard } from "../components/DashboardCard";
import { ArrowRight, Database, Server, BrainCircuit } from "lucide-react";

export default function Methodology() {
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
        <div className="overflow-x-auto">
          <table className="w-full text-sm text-left text-pko-navy">
            <thead className="text-xs text-pko-navy uppercase bg-gray-50 border-b border-gray-200">
              <tr>
                <th scope="col" className="px-6 py-3 font-bold">Kod</th>
                <th scope="col" className="px-6 py-3 font-bold">Nazwa Wskaźnika</th>
                <th scope="col" className="px-6 py-3 font-bold">Preferencja (Ocena)</th>
                <th scope="col" className="px-6 py-3 font-bold">Uzasadnienie</th>
              </tr>
            </thead>
            <tbody>
              {indicators.map((indicator) => (
                <tr key={indicator.id} className="bg-white border-b border-gray-100 hover:bg-gray-50">
                  <td className="px-6 py-4 font-medium text-pko-navy/80 whitespace-nowrap">
                    {indicator.code}
                  </td>
                  <td className="px-6 py-4 font-medium">
                    {indicator.name}
                  </td>
                  <td className="px-6 py-4">
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
      </DashboardCard>
    </div>
  );
}
