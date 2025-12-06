export interface IndicatorDefinition {
  id: number;
  code: string;
  name: string;
  preference: string;
  justification: string;
}

export const indicators: IndicatorDefinition[] = [
  {
    id: 0,
    code: "C",
    name: "Środki pieniężne i papiery wartościowe",
    preference: "Im wyższe, tym lepiej",
    justification: "Większe zasoby płynnych aktywów → mniejsze ryzyko braku płynności."
  },
  {
    id: 1,
    code: "CF",
    name: "Nadwyżka finansowa (cash flow)",
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
    preference: "Ostrożnie / neutralne",
    justification: "Duża liczba jednostek może zwiększać skalę działalności, ale też ryzyko."
  },
  {
    id: 4,
    code: "GS (I)",
    name: "Przychody netto ze sprzedaży",
    preference: "Im wyższe, tym lepiej",
    justification: "Wyższe przychody stabilizują spłatę kredytu."
  },
  {
    id: 5,
    code: "GS",
    name: "Przychody ogółem",
    preference: "Im wyższe, tym lepiej",
    justification: "Zwiększenie skali działalności poprawia zdolność kredytową, jeśli koszty są pod kontrolą."
  },
  {
    id: 6,
    code: "INV",
    name: "Zapasy",
    preference: "Raczej niższe / optymalne",
    justification: "Nadmierne zapasy wiążą kapitał i mogą wskazywać na problemy ze sprzedażą."
  },
  {
    id: 7,
    code: "IO",
    name: "Nakłady inwestycyjne",
    preference: "Raczej niższe (o ile nie poprawiają zdolności)",
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
    name: "Długoterminowe kredyty",
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
    code: "NP",
    name: "Zysk netto",
    preference: "Im wyższy, tym lepiej",
    justification: "Rentowność to fundament wiarygodności kredytowej."
  },
  {
    id: 12,
    code: "NWC",
    name: "Kapitał obrotowy",
    preference: "Dodatni i stabilny",
    justification: "Stabilny poziom zapewnia płynność, ale nadmierny nie jest konieczny."
  },
  {
    id: 13,
    code: "OFE",
    name: "Pozostałe koszty finansowe",
    preference: "Im niższe, tym lepiej",
    justification: "Duże koszty finansowe oznaczają wyższe ryzyko kredytowe."
  },
  {
    id: 14,
    code: "OP",
    name: "Wynik operacyjny",
    preference: "Im wyższy, tym lepiej",
    justification: "Kluczowy wskaźnik efektywności działalności podstawowej."
  },
  {
    id: 15,
    code: "PEN",
    name: "Rentowne jednostki",
    preference: "Im więcej, tym lepiej",
    justification: "Większa liczba zdrowych jednostek zmniejsza ryzyko kredytowe."
  },
  {
    id: 16,
    code: "PNPM",
    name: "Przychody netto",
    preference: "Im wyższe, tym lepiej",
    justification: "Silne przychody stabilizują ocenę kredytową."
  },
  {
    id: 17,
    code: "POS",
    name: "Wynik na sprzedaży",
    preference: "Im wyższy, tym lepiej",
    justification: "Świadczy o zdrowych marżach."
  },
  {
    id: 18,
    code: "PPO",
    name: "Pozostałe przychody operacyjne",
    preference: "Im wyższe, tym lepiej",
    justification: "Dodatkowe źródła przychodów obniżają ryzyko."
  },
  {
    id: 19,
    code: "Przych. fin.",
    name: "Przychody finansowe",
    preference: "Im wyższe, tym lepiej",
    justification: "Zwiększają wynik i płynność."
  },
  {
    id: 20,
    code: "REC",
    name: "Należności krótkoterminowe",
    preference: "Im niższe (po optymalizacji), tym lepiej",
    justification: "Niskie należności = dobra ściągalność → mniejsze ryzyko."
  },
  {
    id: 21,
    code: "STC",
    name: "Krótkoterminowe kredyty",
    preference: "Im niższe, tym lepiej",
    justification: "Mniejsze ryzyko utraty płynności."
  },
  {
    id: 22,
    code: "STL",
    name: "Zobowiązania krótkoterminowe",
    preference: "Im niższe, tym lepiej",
    justification: "Zbyt duże krótkoterminowe zobowiązania zwiększają ryzyko płynności."
  },
  {
    id: 23,
    code: "TC",
    name: "Koszty ogółem",
    preference: "Im niższe, tym lepiej",
    justification: "Niższe koszty poprawiają rentowność i ocenę kredytową."
  },
  {
    id: 24,
    code: "Upadłość",
    name: "Ryzyko upadłości",
    preference: "Minimalizować",
    justification: "Podstawowy negatywny czynnik ryzyka kredytowego."
  },
  {
    id: 1000,
    code: "Marża netto",
    name: "Marża netto",
    preference: "Im wyższa, tym lepiej",
    justification: "Im większa marża, tym większy bufor na wahania kosztów."
  },
  {
    id: 1001,
    code: "Marża operacyjna",
    name: "Marża operacyjna",
    preference: "Im wyższa, tym lepiej",
    justification: "Wysoka rentowność operacyjna istotna dla oceny ryzyka."
  },
  {
    id: 1002,
    code: "Wskaźnik bieżącej płynności",
    name: "Wskaźnik bieżącej płynności",
    preference: "Optymalnie 1,5–2,5",
    justification: "Za niski = ryzyko; zbyt wysoki = nieefektywność."
  },
  {
    id: 1003,
    code: "Wskaźnik szybki",
    name: "Wskaźnik szybki",
    preference: "Im wyższy, tym lepiej (≥ 1)",
    justification: "Bardziej konserwatywny miernik płynności."
  },
  {
    id: 1004,
    code: "Wskaźnik zadłużenia",
    name: "Wskaźnik zadłużenia",
    preference: "Im niższy, tym lepiej",
    justification: "Niższy lewar finansowy = mniejsze ryzyko."
  },
  {
    id: 1005,
    code: "Pokrycie odsetek",
    name: "Pokrycie odsetek",
    preference: "Im wyższe, tym lepiej (≥ 3)",
    justification: "Wysoka zdolność do obsługi zadłużenia."
  },
  {
    id: 1006,
    code: "Rotacja należności",
    name: "Rotacja należności",
    preference: "Im wyższa, tym lepiej",
    justification: "Szybkie odzyskiwanie należności zwiększa płynność."
  },
  {
    id: 1007,
    code: "Cash flow margin",
    name: "Cash flow margin",
    preference: "Im wyższy, tym lepiej",
    justification: "Wysoka konwersja przychodów w gotówkę obniża ryzyko kredytowe."
  }
];
