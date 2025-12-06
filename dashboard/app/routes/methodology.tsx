import { indicators } from "../data/indicators";
import { DashboardCard } from "../components/DashboardCard";

export default function Methodology() {
  return (
    <div className="space-y-8">
      <div>
        <h2 className="text-2xl font-bold text-pko-navy mb-2">Metodologia Oceny</h2>
        <p className="text-pko-navy/70">
          Kierunek oceny zmiennych w analizie metodami wielokryterialnymi z perspektywy przyznawania kredytów.
        </p>
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
