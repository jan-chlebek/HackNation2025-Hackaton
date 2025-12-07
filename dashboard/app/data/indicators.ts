export interface IndicatorDefinition {
  id: number;
  code: string;
  name: string;
  preference: string;
  justification: string;
  formula?: string;
  datasets?: string[];
}

export const indicators: IndicatorDefinition[] = [
  {
    id: 0,
    code: "C",
    name: "Środki pieniężne i pap. wart.",
    preference: "Im wyższe, tym lepiej",
    justification: "Większe zasoby płynnych aktywów → mniejsze ryzyko braku płynności."
  },
  {
    id: 1,
    code: "CF",
    name: "Nadwyżka finansowa",
    preference: "Im wyższe, tym lepiej",
    justification: "Mocna generacja gotówki to kluczowy wskaźnik zdolności spłaty długu."
  },
  {
    id: 2,
    code: "DEPR",
    name: "Amortyzacja",
    preference: "Neutralne / informacyjne",
    justification: "Nie wpływa bezpośrednio na ryzyko niewypłacalności."
  },
  {
    id: 3,
    code: "EN",
    name: "Liczba jednostek gospodarczych",
    preference: "Im wyższe, tym lepiej",
    justification: "Duża liczba jednostek może zwiększać skalę działalności."
  },
  {
    id: 4,
    code: "GS (I)",
    name: "Przychody netto ze sprzedaży i zrównane z nimi",
    preference: "Im wyższe, tym lepiej",
    justification: "Wyższe przychody stabilizują spłatę kredytu."
  },
  {
    id: 5,
    code: "GS",
    name: "Przychody ogółem",
    preference: "Im wyższe, tym lepiej",
    justification: "Zwiększenie skali działalności poprawia zdolność kredytową."
  },
  {
    id: 6,
    code: "INV",
    name: "Zapasy",
    preference: "Raczej niższe / optymalne",
    justification: "Nadmierne zapasy wiążą kapitał."
  },
  {
    id: 7,
    code: "IO",
    name: "Wartość nakładów inwestycyjnych",
    preference: "Raczej niższe",
    justification: "Zbyt duże inwestycje podnoszą ryzyko finansowe w okresie spłaty."
  },
  {
    id: 8,
    code: "IP",
    name: "Odsetki do zapłacenia",
    preference: "Im niższe, tym lepiej",
    justification: "Wyższe odsetki obciążają zdolność do spłaty nowego kredytu."
  },
  {
    id: 9,
    code: "LTC",
    name: "Długoterminowe kredyty bankowe",
    preference: "Im niższe, tym lepiej",
    justification: "Niższe zadłużenie długoterminowe = mniejsze ryzyko kredytowe."
  },
  {
    id: 10,
    code: "LTL",
    name: "Zobowiązania długoterminowe",
    preference: "Im niższe, tym lepiej",
    justification: "Mniejsze stałe obciążenia poprawiają zdolność do spłaty nowego kredytu."
  },
  {
    id: 11,
    code: "Liczba firm z zawieszoną działalnością Ogółem",
    name: "Liczba firm z zawieszoną działalnością",
    preference: "Im niższe, tym lepiej",
    justification: "Wskazuje na problemy w sektorze."
  },
  {
    id: 12,
    code: "Liczba firm zamkniętych Ogółem",
    name: "Liczba firm zamkniętych",
    preference: "Im niższe, tym lepiej",
    justification: "Wysoka liczba zamknięć to sygnał ostrzegawczy."
  },
  {
    id: 13,
    code: "Liczba firm zarejestrowanych Ogółem",
    name: "Liczba firm zarejestrowanych",
    preference: "Im wyższe, tym lepiej",
    justification: "Wzrost liczby firm świadczy o rozwoju sektora."
  },
  {
    id: 14,
    code: "Liczba nowych firm Ogółem",
    name: "Liczba nowych firm",
    preference: "Im wyższe, tym lepiej",
    justification: "Nowe firmy to potencjał wzrostu."
  },
  {
    id: 15,
    code: "NP",
    name: "Wynik finansowy netto (zysk netto)",
    preference: "Im wyższy, tym lepiej",
    justification: "Rentowność to fundament wiarygodności kredytowej."
  },
  {
    id: 16,
    code: "NWC",
    name: "Kapitał obrotowy",
    preference: "Dodatni i stabilny",
    justification: "Stabilny poziom zapewnia płynność."
  },
  {
    id: 17,
    code: "OFE",
    name: "Pozostałe koszty finansowe",
    preference: "Im niższe, tym lepiej",
    justification: "Duże koszty finansowe oznaczają wyższe ryzyko kredytowe."
  },
  {
    id: 18,
    code: "OP",
    name: "Wynik na działalności operacyjnej",
    preference: "Im wyższy, tym lepiej",
    justification: "Kluczowy wskaźnik efektywności działalności podstawowej."
  },
  {
    id: 19,
    code: "Osoby fizyczne",
    name: "Osoby fizyczne prowadzące działalność",
    preference: "Im wyższe, tym lepiej",
    justification: "Informacja o strukturze rynku."
  },
  {
    id: 20,
    code: "Osoby prawne",
    name: "Osoby prawne/jednostki org.",
    preference: "Im wyższe, tym lepiej",
    justification: "Informacja o strukturze rynku."
  },
  {
    id: 21,
    code: "PEN",
    name: "Liczba rentownych jednostek",
    preference: "Im więcej, tym lepiej",
    justification: "Większa liczba zdrowych jednostek zmniejsza ryzyko kredytowe."
  },
  {
    id: 22,
    code: "PNPM",
    name: "Przychody netto",
    preference: "Im wyższe, tym lepiej",
    justification: "Silne przychody stabilizują ocenę kredytową."
  },
  {
    id: 23,
    code: "POS",
    name: "Wynik na sprzedaży",
    preference: "Im wyższy, tym lepiej",
    justification: "Świadczy o zdrowych marżach."
  },
  {
    id: 24,
    code: "PPO",
    name: "Pozostałe przychody operacyjne",
    preference: "Im wyższe, tym lepiej",
    justification: "Dodatkowe źródła przychodów obniżają ryzyko."
  },
  {
    id: 25,
    code: "Pracujący 0-9",
    name: "Przewidywana liczba pracujących 0-9",
    preference: "Im wyższe, tym lepiej",
    justification: "Wielkość zatrudnienia w mikroprzedsiębiorstwach."
  },
  {
    id: 26,
    code: "Pracujący 10-49",
    name: "Przewidywana liczba pracujących 10-49",
    preference: "Im wyższe, tym lepiej",
    justification: "Wielkość zatrudnienia w małych przedsiębiorstwach."
  },
  {
    id: 27,
    code: "Pracujący 250+",
    name: "Przewidywana liczba pracujących 250=>",
    preference: "Im wyższe, tym lepiej",
    justification: "Wielkość zatrudnienia w dużych przedsiębiorstwach."
  },
  {
    id: 28,
    code: "Pracujący 50-249",
    name: "Przewidywana liczba pracujących 50-249",
    preference: "Im wyższe, tym lepiej",
    justification: "Wielkość zatrudnienia w średnich przedsiębiorstwach."
  },
  {
    id: 29,
    code: "Pracujący Ogółem",
    name: "Przewidywana liczba pracujących Ogółem",
    preference: "Im wyższe, tym lepiej",
    justification: "Ogólny poziom zatrudnienia w sektorze."
  },
  {
    id: 30,
    code: "Przych. fin.",
    name: "Przychody finansowe",
    preference: "Im wyższe, tym lepiej",
    justification: "Zwiększają wynik i płynność."
  },
  {
    id: 31,
    code: "REC",
    name: "Należności krótkoterminowe",
    preference: "Im niższe (po optymalizacji), tym lepiej",
    justification: "Niskie należności = dobra ściągalność → mniejsze ryzyko."
  },
  {
    id: 32,
    code: "STC",
    name: "Krótkoterminowe kredyty bankowe",
    preference: "Im niższe, tym lepiej",
    justification: "Mniejsze ryzyko utraty płynności."
  },
  {
    id: 33,
    code: "STL",
    name: "Zobowiązania krótkoterminowe",
    preference: "Im niższe, tym lepiej",
    justification: "Zbyt duże krótkoterminowe zobowiązania zwiększają ryzyko płynności."
  },
  {
    id: 34,
    code: "TC",
    name: "Koszty ogółem",
    preference: "Im niższe, tym lepiej",
    justification: "Niższe koszty poprawiają rentowność i ocenę kredytową."
  },
  {
    id: 35,
    code: "Upadłość",
    name: "Upadłość",
    preference: "Minimalizować",
    justification: "Podstawowy negatywny czynnik ryzyka kredytowego."
  },
  {
    id: 36,
    code: "Firmy Ogółem",
    name: "Liczba firm i działalności gospodarczych Ogółem",
    preference: "Im wyższe, tym lepiej",
    justification: "Całkowita liczba podmiotów na rynku."
  },
  {
    id: 1000,
    code: "Net Profit Margin",
    name: "Marża netto",
    preference: "Im wyższa, tym lepiej",
    justification: "Im większa marża, tym większy bufor na wahania kosztów.",
    formula: "NP / PNPM",
    datasets: ["credit"]
  },
  {
    id: 1001,
    code: "Operating Margin",
    name: "Marża operacyjna",
    preference: "Im wyższa, tym lepiej",
    justification: "Wysoka rentowność operacyjna istotna dla oceny ryzyka.",
    formula: "OP / PNPM",
    datasets: ["credit"]
  },
  {
    id: 1002,
    code: "Current Ratio",
    name: "Wskaźnik bieżącej płynności",
    preference: "Optymalnie 1,5–2,5",
    justification: "Za niski = ryzyko; zbyt wysoki = nieefektywność.",
    formula: "(C + REC + INV) / STL",
    datasets: ["credit"]
  },
  {
    id: 1003,
    code: "Quick Ratio",
    name: "Wskaźnik szybki",
    preference: "Im wyższy, tym lepiej (≥ 1)",
    justification: "Bardziej konserwatywny miernik płynności.",
    formula: "(C + REC) / STL",
    datasets: ["credit"]
  },
  {
    id: 1004,
    code: "Cash Ratio",
    name: "Wskaźnik gotówkowy",
    preference: "Im wyższy, tym lepiej",
    justification: "Najbardziej konserwatywny miernik płynności.",
    formula: "C / STL",
    datasets: ["credit"]
  },
  {
    id: 1005,
    code: "Short Debt Share",
    name: "Udział długu krótkoterminowego",
    preference: "Im niższy, tym lepiej",
    justification: "Mniejsze uzależnienie od finansowania krótkoterminowego.",
    formula: "STL / (STL + LTL)",
    datasets: ["credit"]
  },
  {
    id: 1006,
    code: "Long-term Debt Share",
    name: "Udział długu długoterminowego",
    preference: "Zależne od strategii",
    justification: "Stabilne finansowanie, ale kosztowne.",
    formula: "LTL / (STL + LTL)",
    datasets: ["credit"]
  },
  {
    id: 1007,
    code: "Interest Coverage",
    name: "Pokrycie odsetek",
    preference: "Im wyższe, tym lepiej (≥ 3)",
    justification: "Wysoka zdolność do obsługi zadłużenia.",
    formula: "OP / IP",
    datasets: ["credit"]
  },
  {
    id: 1008,
    code: "Financial Risk Ratio",
    name: "Wskaźnik ryzyka finansowego",
    preference: "Im niższy, tym lepiej",
    justification: "Mniejsze obciążenie kosztami finansowymi.",
    formula: "OFE / OP",
    datasets: ["credit"]
  },
  {
    id: 1009,
    code: "Cash Flow Margin",
    name: "Marża przepływów pieniężnych",
    preference: "Im wyższa, tym lepiej",
    justification: "Efektywność generowania gotówki ze sprzedaży.",
    formula: "CF / PNPM",
    datasets: ["credit"]
  },
  {
    id: 1010,
    code: "Operating Cash Coverage",
    name: "Pokrycie gotówkowe operacyjne",
    preference: "Im wyższe, tym lepiej",
    justification: "Zdolność do pokrycia zobowiązań z działalności operacyjnej.",
    formula: "(OP + DEPR) / (STL + LTL)",
    datasets: ["credit"]
  },
  {
    id: 1011,
    code: "Bankruptcy Rate",
    name: "Wskaźnik upadłości",
    preference: "Im niższy, tym lepiej",
    justification: "Ryzyko systemowe w sektorze.",
    formula: "Upadłość / EN",
    datasets: ["credit"]
  },
  {
    id: 1012,
    code: "Closure Rate",
    name: "Wskaźnik zamknięć",
    preference: "Im niższy, tym lepiej",
    justification: "Stabilność sektora.",
    formula: "Zamknięte / EN",
    datasets: ["credit"]
  },
  {
    id: 1013,
    code: "Profit Firms Share",
    name: "Udział firm rentownych",
    preference: "Im wyższy, tym lepiej",
    justification: "Ogólna kondycja sektora.",
    formula: "PEN / EN",
    datasets: ["credit"]
  },
  {
    id: 1020,
    code: "Sales Profitability",
    name: "Rentowność sprzedaży",
    preference: "Im wyższa, tym lepiej",
    justification: "Efektywność sprzedaży.",
    formula: "POS / PNPM",
    datasets: ["effectivity"]
  },
  {
    id: 1022,
    code: "Cost Share Ratio",
    name: "Wskaźnik udziału kosztów",
    preference: "Im niższy, tym lepiej",
    justification: "Kontrola kosztów.",
    formula: "TC / PNPM",
    datasets: ["effectivity"]
  },
  {
    id: 1023,
    code: "Receivables Turnover",
    name: "Rotacja należności",
    preference: "Im wyższa, tym lepiej",
    justification: "Szybkość odzyskiwania należności.",
    formula: "PNPM / REC",
    datasets: ["effectivity"]
  },
  {
    id: 1024,
    code: "Inventory Turnover",
    name: "Rotacja zapasów",
    preference: "Im wyższa, tym lepiej",
    justification: "Efektywność zarządzania zapasami.",
    formula: "TC / INV",
    datasets: ["effectivity"]
  },
  {
    id: 1025,
    code: "Current Asset Turnover",
    name: "Rotacja aktywów obrotowych",
    preference: "Im wyższa, tym lepiej",
    justification: "Efektywność wykorzystania aktywów.",
    formula: "PNPM / (C + REC + INV)",
    datasets: ["effectivity"]
  },
  {
    id: 1026,
    code: "Investment Ratio",
    name: "Wskaźnik inwestycji",
    preference: "Im wyższy, tym lepiej (dla rozwoju)",
    justification: "Skłonność do inwestowania.",
    formula: "IO / PNPM",
    datasets: ["effectivity"]
  },
  {
    id: 1027,
    code: "Financial Revenue Share",
    name: "Udział przychodów finansowych",
    preference: "Zależne od specyfiki",
    justification: "Dywersyfikacja przychodów.",
    formula: "Przych.fin. / GS",
    datasets: ["effectivity"]
  },
  {
    id: 1028,
    code: "Net Firm Growth Rate",
    name: "Wskaźnik wzrostu netto firm",
    preference: "Im wyższy, tym lepiej",
    justification: "Dynamika rozwoju sektora.",
    formula: "(Zarejestrowane - Zamknięte) / EN",
    datasets: ["effectivity"]
  },
  {
    id: 1029,
    code: "Average Firm Size",
    name: "Średnia wielkość firmy",
    preference: "Im wyższa, tym lepiej",
    justification: "Skala działalności.",
    formula: "Pracujący / EN",
    datasets: ["effectivity"]
  },
  {
    id: 1040,
    code: "Investment Ratio",
    name: "Wskaźnik inwestycji (Rozwój)",
    preference: "Im wyższy, tym lepiej",
    justification: "Potencjał rozwojowy.",
    formula: "IO / PNPM",
    datasets: ["development"]
  },
  {
    id: 1041,
    code: "Amortization Ratio",
    name: "Wskaźnik amortyzacji",
    preference: "Im niższy, tym lepiej (nowoczesność)",
    justification: "Stopień zużycia majątku.",
    formula: "DEPR / PNPM",
    datasets: ["development"]
  },
  {
    id: 1042,
    code: "Cash Flow Margin",
    name: "Marża CF (Rozwój)",
    preference: "Im wyższa, tym lepiej",
    justification: "Zdolność do samofinansowania rozwoju.",
    formula: "CF / PNPM",
    datasets: ["development"]
  },
  {
    id: 1043,
    code: "Operating Cash Coverage",
    name: "Pokrycie gotówkowe (Rozwój)",
    preference: "Im wyższe, tym lepiej",
    justification: "Bezpieczeństwo finansowe rozwoju.",
    formula: "(OP + DEPR) / (STL + LTL)",
    datasets: ["development"]
  },
  {
    id: 1044,
    code: "Profit Firms Share",
    name: "Udział firm rentownych (Rozwój)",
    preference: "Im wyższy, tym lepiej",
    justification: "Kondycja sektora sprzyjająca rozwojowi.",
    formula: "PEN / EN",
    datasets: ["development"]
  },
  {
    id: 1045,
    code: "Net Firm Growth Rate",
    name: "Wzrost netto firm (Rozwój)",
    preference: "Im wyższy, tym lepiej",
    justification: "Ekspansja sektora.",
    formula: "(Zarejestrowane - Zamknięte) / EN",
    datasets: ["development"]
  },
  {
    id: 1046,
    code: "New Firms Rate",
    name: "Wskaźnik nowych firm",
    preference: "Im wyższy, tym lepiej",
    justification: "Atrakcyjność sektora dla nowych podmiotów.",
    formula: "Nowe / EN",
    datasets: ["development"]
  },
  {
    id: 1047,
    code: "Closure Rate",
    name: "Wskaźnik zamknięć (Rozwój)",
    preference: "Im niższy, tym lepiej",
    justification: "Stabilność.",
    formula: "Zamknięte / EN",
    datasets: ["development"]
  },
  {
    id: 1048,
    code: "Suspension Rate",
    name: "Wskaźnik zawieszeń",
    preference: "Im niższy, tym lepiej",
    justification: "Problemy w prowadzeniu działalności.",
    formula: "Zawieszone / EN",
    datasets: ["development"]
  },
  {
    id: 1049,
    code: "Operating Margin",
    name: "Marża operacyjna (Rozwój)",
    preference: "Im wyższa, tym lepiej",
    justification: "Efektywność operacyjna.",
    formula: "OP / PNPM",
    datasets: ["development"]
  },
  {
    id: 1050,
    code: "POS Margin",
    name: "Marża ze sprzedaży (Rozwój)",
    preference: "Im wyższa, tym lepiej",
    justification: "Rentowność podstawowa.",
    formula: "POS / PNPM",
    datasets: ["development"]
  },
  {
    id: 1051,
    code: "Bank Loans Ratio",
    name: "Wskaźnik kredytowania",
    preference: "Im wyższy, tym lepiej (dostępność)",
    justification: "Dostępność finansowania bankowego.",
    formula: "(STC + LTC) / (STL + LTL)",
    datasets: ["development"]
  }
];
