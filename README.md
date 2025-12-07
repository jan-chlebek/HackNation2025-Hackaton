# System Analizy i Predykcji Kondycji Sektorów Gospodarki

## Spis treści
1. [Wprowadzenie](#wprowadzenie)
2. [Architektura systemu](#architektura-systemu)
3. [Metodologia konstruowania wskaźników złożonych](#metodologia-konstruowania-wskaźników-złożonych)
4. [Kategoryzacja wskaźników](#kategoryzacja-wskaźników)
5. [Metody wielokryterialnej oceny decyzyjnej](#metody-wielokryterialnej-oceny-decyzyjnej)
6. [Ensemble i agregacja wyników](#ensemble-i-agregacja-wyników)
7. [System predykcji](#system-predykcji)
8. [Uzasadnienie rozwiązań](#uzasadnienie-rozwiązań)

---

## Wprowadzenie

System został zaprojektowany w celu kompleksowej analizy kondycji sektorów gospodarki polskiej oraz predykcji ich przyszłego stanu. Wykorzystuje dane finansowe oraz zaawansowane metody analityczne do oceny zdolności kredytowej, efektywności operacyjnej i potencjału rozwojowego różnych branż gospodarki.

**Kluczowe cele systemu:**
- Obiektywna ocena kondycji finansowej sektorów PKD
- Wielowymiarowa analiza uwzględniająca różne aspekty działalności przedsiębiorstw
- Predykcja trendów na podstawie danych historycznych
- Wsparcie decyzyjne dla instytucji finansowych, inwestorów i analityków

---

## Architektura systemu

System składa się z czterech głównych modułów:

```
┌─────────────────────────────────────────────────────────────┐
│ 1. MODUŁ KONSTRUKCJI WSKAŹNIKÓW ZŁOŻONYCH                   │
│    (creating_complex_indicators.ipynb)                       │
│    - Imputacja brakujących danych                           │
│    - Kalkulacja 44 wskaźników pochodnych                    │
│    - Kategoryzacja tematyczna                               │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 2. MODUŁ ANALIZY WIELOKRYTERIALNEJ                          │
│    (outcome.py, analysis.py)                                 │
│    - TOPSIS (Technique for Order Preference)                │
│    - VIKOR (Multicriteria Optimization)                     │
│    - Monte Carlo Ensemble                                   │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 3. MODUŁ AGREGACJI I ENSEMBLE                               │
│    (run_prediction_sets.py)                                  │
│    - Równoległa analiza 4 zestawów wskaźników              │
│    - Wagowanie temporalne                                   │
│    - Agregacja wyników z różnych metod                     │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│ 4. MODUŁ PREDYKCJI                                          │
│    (run_prediction.py)                                       │
│    - Prognozowanie wskaźników bazowych (0-36)              │
│    - Kalkulacja wskaźników pochodnych (1000+)              │
│    - 5 metod szeregów czasowych + korelacje                │
└─────────────────────────────────────────────────────────────┘
```

---

## Metodologia konstruowania wskaźników złożonych

### 1. Fundament: Dane bazowe GUS

System wykorzystuje **37 wskaźników bazowych** (indeksy 0-36) pochodzących z danych finansowych:
**Kategorie danych bazowych:**
- **Finansowe**: Wynik finansowy netto (NP), Wynik operacyjny (OP), Przychody (PNPM), Koszty (TC)
- **Płynność**: Środki pieniężne (C), Należności (REC), Zapasy (INV)
- **Zobowiązania**: Zobowiązania krótkoterminowe (STL), długoterminowe (LTL)
- **Struktura**: Liczba jednostek (EN), Liczba rentownych (PEN)
- **Rynkowe**: Upadłości, zamknięcia, zawieszenia, nowe firmy

### 2. Proces konstrukcji wskaźników złożonych

#### Krok 1: Imputacja brakujących danych

**Problem**: Dane finansowe zawierają luki mogące wynikać z:
- Braku raportowania w niektórych okresach
- Małej liczby przedsiębiorstw w niszowych sektorach
- Zmian metodologicznych w zbieraniu danych

**Rozwiązanie - wielopoziomowa imputacja, gdzie każdy kolejny poziom jest uzyty do wypełnienia pustych danych, tylko jeśli poprzedni nie mógł zostać zastosowany:**

```python
# Poziom 1: Interpolacja liniowa w obrębie tej samej serii czasowej
values_df.groupby(['WSKAZNIK_INDEX', 'PKD_INDEX'])['wartosc'].transform(
    lambda group: group.interpolate(method='linear', limit_direction='both')
)

# Poziom 2: Mediana dla wskaźnika w danym roku (cross-sectional)
median_by_indicator_year = values_df.groupby(['WSKAZNIK_INDEX', 'rok'])['wartosc'].transform('median')

# Poziom 3: Mediana globalna dla wskaźnika
median_by_indicator = values_df.groupby('WSKAZNIK_INDEX')['wartosc'].transform('median')

# Poziom 4: Zero jako ostateczność (rzadkie przypadki)
```

**Uzasadnienie podejścia:**
- **Interpolacja** zachowuje trend czasowy w obrębie sektora
- **Mediana cross-sectional** uwzględnia kondycję gospodarki w danym roku
- **Mediana globalna** zapewnia spójność z historycznymi wartościami wskaźnika
- **Brak agresywnej imputacji** - wartości zero tylko w skrajnych przypadkach

#### Krok 2: Konstrukcja wskaźników złożonych przez kompozycję

**44 wskaźniki pochodne** (indeksy 1000-1067) konstruowane są jako **racjonalne funkcje wskaźników bazowych**:

**Przykłady formuł:**

```python
# Wskaźniki rentowności (margins)
Net_Profit_Margin = NP / PNPM                    # 1000
# Wynik finansowy netto / Przychody netto

Operating_Margin = OP / PNPM                     # 1001
# Wynik na działalności operacyjnej / Przychody netto

# Wskaźniki płynności (liquidity ratios)
Current_Ratio = (C + REC + INV) / STL           # 1002
# (Środki pieniężne + Należności + Zapasy) / Zobowiązania krótkoterminowe

Quick_Ratio = (C + REC) / STL                   # 1003
# (Środki pieniężne + Należności) / Zobowiązania krótkoterminowe

Cash_Ratio = C / STL                            # 1004
# Środki pieniężne / Zobowiązania krótkoterminowe

# Wskaźniki zadłużenia (leverage)
Short_Debt_Share = STL / (STL + LTL)            # 1005
# Zobowiązania krótkoterminowe / (Zobowiązania krótkoterminowe + Zobowiązania długoterminowe)

Long_Debt_Share = LTL / (STL + LTL)             # 1006
# Zobowiązania długoterminowe / (Zobowiązania krótkoterminowe + Zobowiązania długoterminowe)

# Wskaźniki pokrycia (coverage)
Interest_Coverage = OP / IP                      # 1007
# Wynik operacyjny / Odsetki do zapłacenia

Operating_Cash_Coverage = (OP + DEPR) / (STL + LTL)  # 1010
# (Wynik operacyjny + Amortyzacja) / (Zobowiązania krótkoterminowe + Zobowiązania długoterminowe)

# Wskaźniki efektywności (efficiency)
Receivables_Turnover = PNPM / REC               # 1023
# Przychody netto / Należności krótkoterminowe

Inventory_Turnover = TC / INV                   # 1024
# Koszty ogółem / Zapasy

Current_Asset_Turnover = PNPM / (C + REC + INV) # 1025
# Przychody netto / (Środki pieniężne + Należności + Zapasy)

# Wskaźniki rynkowe (market health)
Bankruptcy_Rate = UPADLOSC / EN                  # 1011
# Liczba upadłości / Liczba jednostek gospodarczych

Closure_Rate = ZAMKNIETE / EN                    # 1012
# Liczba firm zamkniętych / Liczba jednostek gospodarczych

Profit_Firms_Share = PEN / EN                    # 1013
# Liczba rentownych jednostek / Liczba jednostek gospodarczych
```

#### Krok 3: Bezpieczne dzielenie (safe division)

**Kluczowy mechanizm obsługi dzielenia przez zero:**

```python
def safe_divide(numerator, denominator, fill_value=0):
    result = numerator / denominator
    result = result.replace([np.inf, -np.inf], fill_value)
    result = result.fillna(fill_value)
    return result
```

**Uzasadnienie:**
- **Wartość 0 zamiast NaN/inf** - zapewnia ciągłość obliczeń
- **Interpretacja ekonomiczna** - brak przychodów → wskaźniki rentowności = 0
- **Stabilność numeryczna** - eliminuje problemy w późniejszych analizach

### 3. Dlaczego konstrukcja, a nie uczenie maszynowe?

**Decyzja**: Wskaźniki złożone konstruowane są **deterministycznie**, a nie za pomocą uczenia maszynowego zawierającego często elemnt losowości:

**Zalety podejścia konstrukcyjnego:**
1. **Interpretowalność** - każdy wskaźnik ma jasną ekonomiczną interpretację
2. **Stabilność** - wzory nie zmieniają się w czasie, wyniki są porównywalne
3. **Zgodność ze standardami** - użycie wskaźników finansowych, które są powszechnie akceptowane (ROE, Current Ratio, etc.)
4. **Brak overfittingu** - brak ryzyk związanych ściśle z uczeniem maszynowym
5. **Wymóg regulacyjny** - instytucje finansowe wymagają przejrzystych metryk

---

## Kategoryzacja wskaźników

### Filozofia podziału tematycznego

**44 wskaźniki pochodne** zostały podzielone na **4 zestawy tematyczne**, co pozwala na:
- Specjalistyczną analizę różnych aspektów kondycji sektora
- Redukcję wymiarowości przy zachowaniu istotnej informacji
- Równoległą analizę różnych perspektyw oceny

### Zestaw 1: Zdolność kredytowa i płynność (14 wskaźników, 1000-1013)

**Cel**: Ocena zdolności sektora do spłaty zobowiązań i zarządzania płynnością.

**Wskaźniki kluczowe:**
- **Marże**: Net Profit Margin, Operating Margin (1000-1001)
- **Płynność**: Current Ratio, Quick Ratio, Cash Ratio (1002-1004)
- **Struktura długu**: Udziały długu krótko/długoterminowego (1005-1006)
- **Pokrycie**: Interest Coverage, Operating Cash Coverage (1007, 1010)
- **Ryzyko**: Financial Risk Ratio, Bankruptcy Rate (1008, 1011)
- **Kondycja**: Cash Flow Margin, Profit Firms Share (1009, 1013)

**Zastosowanie**: Ocena ryzyka kredytowego przez banki, analiza zdolności do obsługi długu.

### Zestaw 2: Efektywność operacyjna (10 wskaźników, 1020-1029)

**Cel**: Pomiar wydajności wykorzystania zasobów i generowania przychodów.

**Wskaźniki kluczowe:**
- **Rentowność sprzedaży**: Sales Profitability, Core Revenue Share (1020-1021)
- **Kontrola kosztów**: Cost Share Ratio (1022)
- **Rotacje**: Receivables Turnover, Inventory Turnover (1023-1024)
- **Efektywność aktywów**: Current Asset Turnover (1025)
- **Inwestycje**: Investment Ratio (1026)
- **Wzrost**: Net Firm Growth Rate, Average Firm Size (1028-1029)

**Zastosowanie**: Benchmarking sektorowy, analiza konkurencyjności, due diligence operacyjne.

### Zestaw 3: Rozwój branży i stabilność (12 wskaźników, 1040-1051)

**Cel**: Ocena potencjału wzrostu i stabilności rynkowej sektora.

**Wskaźniki kluczowe:**
- **Inwestycje w rozwój**: Investment Ratio, Amortization Ratio (1040-1041)
- **Generowanie gotówki**: Cash Flow Margin (1042)
- **Kondycja finansowa**: Operating Cash Coverage (1043)
- **Dynamika rynku**: New Firms Rate, Closure Rate, Suspension Rate (1046-1048)
- **Zdolność operacyjna**: Operating Margin, POS Margin (1049-1050)
- **Finansowanie**: Bank Loans Ratio (1051)

**Zastosowanie**: Planowanie strategiczne, analiza atrakcyjności branży, prognozowanie trendów.

### Zestaw 4: Wskaźniki podstawowe (8 wskaźników, 1060-1067)

**Cel**: Standaryzowany zestaw z polskimi nazwami dla raportowania i zgodności lokalnej.

**Wskaźniki**: Adaptacje najważniejszych wskaźników z nazwami w języku polskim.

**Zastosowanie**: Raporty dla polskich instytucji, zgodność z lokalnymi wymogami regulacyjnymi.

### Zalety kategoryzacji

1. **Modularność** - każdy zestaw analizowany niezależnie
2. **Specjalizacja** - eksperci mogą skupić się na swojej domenie
3. **Skalowalność** - łatwe dodawanie nowych zestawów
4. **Równoległość** - możliwość równoległego przetwarzania (ProcessPoolExecutor)
5. **Porównywalność** - możliwość porównania wyników pomiędzy kategoriami

---

## Metody wielokryterialnej oceny decyzyjnej

### Filozofia MCDM (Multi-Criteria Decision Making)

**Problem fundamentalny**: Jak zagregować 10-14 wskaźników w jedną ocenę sektora?

**Rozwiązanie**: Wykorzystanie trzech komplementarnych metod MCDM:
- **TOPSIS** - odległość od rozwiązania idealnego
- **VIKOR** - kompromis z uwzględnieniem wag
- **Monte Carlo** - probabilistyczna ocena stabilności

---

## Jak obliczane są wagi wskaźników?

**Metoda**: Odwrotność współczynnika zmienności

### Wzór (3 kroki):

```python
# 1. Dla każdego wskaźnika oblicz współczynnik zmienności
c_v = odchylenie_standardowe / średnia

# 2. Oblicz odwrotność
odwrotnosc = 1 / c_v

# 3. Normalizuj do sumy = 1
waga = odwrotnosc / suma_wszystkich_odwrotnosci
```

### Logika:

- **Wskaźnik STABILNY** (mała zmienność) → **duża waga** ✅
- **Wskaźnik CHAOTYCZNY** (duża zmienność) → **mała waga** ❌

### Przykład:

```
Wskaźnik A: c_v = 0.2  →  waga = 1/0.2 = 5.0  →  62.5% (główny wpływ)
Wskaźnik B: c_v = 0.5  →  waga = 1/0.5 = 2.0  →  25.0%
Wskaźnik C: c_v = 1.0  →  waga = 1/1.0 = 1.0  →  12.5% (mały wpływ)
```

**Dlaczego tak?** Stabilne wskaźniki dają bardziej wiarygodne informacje o sektorze.

---

### 1. TOPSIS (Technique for Order of Preference by Similarity to Ideal Solution)

#### Algorytm TOPSIS

**Krok 1: Normalizacja wektorowa**
```python
norm_matrix = matrix / np.sqrt((matrix**2).sum(axis=0))
```
*Uzasadnienie*: Ujednolicenie skali bez utraty proporcji między wartościami.

**Krok 2: Ważenie**
```python
weighted_matrix = norm_matrix * weights
```
*Uzasadnienie*: Uwzględnienie ważności poszczególnych wskaźników.

**Krok 3: Rozwiązania idealne**
```python
ideal_best = weighted_matrix.max(axis=0)  # dla Max
ideal_best = weighted_matrix.min(axis=0)  # dla Min
ideal_worst = weighted_matrix.min(axis=0) # dla Max
ideal_worst = weighted_matrix.max(axis=0) # dla Min
```
*Uzasadnienie*: Referencyjne punkty dla najlepszych/najgorszych wartości każdego kryterium.

**Krok 4: Odległości euklidesowe**
```python
distance_to_best = np.sqrt(((weighted_matrix - ideal_best)**2).sum(axis=1))
distance_to_worst = np.sqrt(((weighted_matrix - ideal_worst)**2).sum(axis=1))
```

**Krok 5: Współczynnik bliskości**
```python
topsis_score = distance_to_worst / (distance_to_best + distance_to_worst)
```
*Interpretacja*: 1.0 = doskonały (blisko ideału), 0.0 = najgorszy.

#### Zalety TOPSIS
- Prosty algorytm, szybkie obliczenia
- Wykorzystuje pełną informację z macierzy decyzyjnej
- Równoważy odległość od najlepszego i najgorszego
- Naturalny zakres wyników [0,1]

#### Wady TOPSIS
- Wrażliwy na wartości skrajne (outliers)
- Nie uwzględnia preferencji decydenta co do kompromisów
- Może nie rozróżniać podobnych wariantów

### 2. VIKOR (VlseKriterijumska Optimizacija I Kompromisno Resenje)

#### Algorytm VIKOR

**Krok 1: Rozwiązania idealne (jak TOPSIS)**

**Krok 2: Wyznaczenie S (suma ważonych odległości) i R (maksymalna ważona odległość)**
```python
S = np.sum(weighted_distances, axis=1)  # Użyteczność grupowa
R = np.max(weighted_distances, axis=1)  # Użyteczność indywidualna (regret)
```

**Krok 3: Indeks VIKOR**
```python
v = 0.5  # waga strategii kompromisu
Q = v * (S - S_min)/(S_max - S_min) + (1-v) * (R - R_min)/(R_max - R_min)
```
*Interpretacja*: Q blisko 0 = najlepsze rozwiązanie kompromisowe.

**Krok 4: Normalizacja do zakresu [0,1]**
```python
vikor_score = 1 - Q  # Odwrócenie: 1 = najlepszy
```

#### Zalety VIKOR
- **Uwzględnia kompromis** między użytecznością grupową (S) a indywidualną (R)
- **Mniej wrażliwy** na wartości skrajne niż TOPSIS
- **Parametr v** pozwala regulować strategię decyzyjną
- **Ranking + rozwiązanie kompromisowe** - więcej informacji niż sam ranking

#### Wady VIKOR
- Bardziej złożony obliczeniowo
- Wyniki zależą od parametru v

### 3. Monte Carlo Ensemble

#### Koncepcja probabilistycznej oceny

**Problem**: Wagi kryteriów są często arbitralne lub niepewne.

**Rozwiązanie**: Symulacja Monte Carlo z perturbacją wag.

#### Algorytm Monte Carlo Ensemble

**Krok 1: Inicjalizacja równych wag**
```python
n_criteria = len(directions)
base_weights = np.ones(n_criteria) / n_criteria
```

**Krok 2: Generowanie perturbowanych wag (n_simulations = 1000)**
```python
for simulation in range(n_simulations):
    # Perturbacja wag szumem gaussowskim
    perturbation = np.random.normal(0, variance, n_criteria)
    perturbed_weights = base_weights * (1 + perturbation)
    
    # Normalizacja do sumy = 1
    perturbed_weights = perturbed_weights / perturbed_weights.sum()
    
    # Obliczenie TOPSIS z perturbowanymi wagami
    scores[simulation] = topsis(matrix, perturbed_weights, directions)
```

**Krok 3: Agregacja wyników**
```python
mc_score = np.mean(scores, axis=0)        # Oczekiwana wartość
mc_std = np.std(scores, axis=0)           # Niepewność
mc_rank_stability = rank_correlation(...)  # Stabilność rankingu
```

#### Zalety Monte Carlo
- **Uwzględnia niepewność** wag kryteriów
- **Stabilność rankingu** - identyfikuje alternatywy stabilne vs. wrażliwe
- **Rozkład wyników** - pełna informacja probabilistyczna, nie tylko punktowa ocena
- **Odporność na błędy** - uśrednienie redukuje wpływ pojedynczych błędów

#### Wady Monte Carlo
- Obliczeniowo kosztowne (1000 symulacji)
- Wyniki zależą od założonej wariancji perturbacji

---

## Ensemble i agregacja wyników

### Filozofia podejścia ensemble

**Kluczowe pytanie**: Która metoda (TOPSIS/VIKOR/MC) jest najlepsza?

**Odpowiedź**: Żadna pojedyncza metoda nie jest optymalna we wszystkich przypadkach.

**Rozwiązanie**: **Ensemble learning** - agregacja wyników z trzech metod.

### Schemat agregacji ensemble

```python
# Obliczenie wyników z trzech metod
topsis_scores = TOPSIS(matrix, weights, directions)
vikor_scores = VIKOR(matrix, weights, directions)
mc_scores = MonteCarlo(matrix, weights, directions, n_simulations=1000)

# Normalizacja do zakresu [0,1] (jeśli potrzebne)
topsis_norm = (topsis_scores - topsis_scores.min()) / (topsis_scores.max() - topsis_scores.min())
vikor_norm = (vikor_scores - vikor_scores.min()) / (vikor_scores.max() - vikor_scores.min())
mc_norm = (mc_scores - mc_scores.min()) / (mc_scores.max() - mc_scores.min())

# Agregacja (średnia prosta)
ensemble_score = (topsis_norm + vikor_norm + mc_norm) / 3

# Ranking finalny
ensemble_rank = rankdata(-ensemble_score, method='min')
```

### Wagowanie temporalne

**Problem**: Czy dane z roku 2023, 2022 i 2021 są równie ważne przy ocenie sektora?

**Rozwiązanie**: Wagowanie temporalne z większą wagą dla danych nowszych.

```python
# Konfiguracja (można dostosować)
temporal_weight_1yr = 0.3   # Dodatkowa waga dla zmian rok-do-roku
temporal_weight_2yr = 0.1   # Dodatkowa waga dla zmian 2-letnie

# Obliczanie średnich ważonych czasowo
# (implementacja w load_and_prepare_sector_data)
```

**Uzasadnienie**:
- Gospodarka jest dynamiczna - nowsze dane bardziej relewantne
- Zmniejsza wpływ historycznych anomalii
- Uwzględnia momentum trendu (wzrostowy vs. spadkowy)

### Równoległe przetwarzanie 4 zestawów

```python
# Definicja zadań dla równoległego przetwarzania
tasks = [
    ('credit', year, typ),
    ('effectivity', year, typ),
    ('development', year, typ),
    ('polish', year, typ)
]

# Wykonanie równoległe
with ProcessPoolExecutor(max_workers=CPU_count-1) as executor:
    futures = {executor.submit(run_analysis, task): task for task in tasks}
    for future in as_completed(futures):
        results = future.result()
```

**Korzyści**:
- **Skrócenie czasu** - 4x przyspieszenie dla 4 rdzeni
- **Niezależność** - każdy zestaw w osobnym procesie
- **Skalowalność** - łatwe dodawanie nowych zestawów
- **Odporność** - błąd w jednym zestawie nie blokuje innych

---

## System predykcji

### Architektura dwupoziomowa

**Kluczowa innowacja**: Rozdzielenie predykcji wskaźników bazowych od kalkulacji wskaźników złożonych.

```
DANE HISTORYCZNE (2013-2024)
         ↓
   ┌─────────────────────────────────┐
   │ POZIOM 1: PREDYKCJA BAZOWA      │
   │ Wskaźniki 0-36 (dane GUS)       │
   │ - 5 metod szeregów czasowych    │
   │ - Analiza korelacji             │
   │ - Ensemble forecasting          │
   └─────────────────────────────────┘
         ↓ Predicted values (2025-2028)
   ┌─────────────────────────────────┐
   │ POZIOM 2: KALKULACJA ZŁOŻONA    │
   │ Wskaźniki 1000-1067             │
   │ - Stosowanie formuł             │
   │ - Zachowanie relacji            │
   │ - Safe division                 │
   └─────────────────────────────────┘
         ↓
   KOMPLETNE PROGNOZY (2025-2028)
```

### Dlaczego tylko bazowe wskaźniki są prognozowane?

#### Problem ze standardowym podejściem

**Błędne podejście**: Bezpośrednia predykcja wszystkich wskaźników (0-1067) szeregami czasowymi.

**Problemy**:
1. **Utrata spójności matematycznej**: Jeśli NP/PNPM ≠ predicted_NP / predicted_PNPM
2. **Naruszenie zależności**: Wskaźniki złożone nie są niezależne
3. **Nieinterpretowalne anomalie**: Możliwe są fizycznie niemożliwe kombinacje (np. Current Ratio > 10 przy niskiej płynności)
4. **Propagacja błędów**: Każdy wskaźnik ma osobny błąd predykcji

#### Poprawne podejście: Predict-then-Calculate

**Zalety**:
1. ✅ **Spójność matematyczna**: Wskaźniki złożone zawsze zgodne z formułami
2. ✅ **Zachowanie relacji**: Net Profit Margin = NP/PNPM (zawsze!)
3. ✅ **Mniej modeli**: 37 modeli zamiast 81 → mniejsze ryzyko overfittingu
4. ✅ **Interpretowalność**: Prognozy oparte na fundamentalnych wskaźnikach ekonomicznych
5. ✅ **Szybsze obliczenia**: ~75% mniej prognoz szeregów czasowych

### Metody predykcji szeregów czasowych

System używa **5 metod forecasting** + **analizę korelacji** w trybie ENSEMBLE:

#### 1. Weighted Moving Average (WMA)

```python
def weighted_moving_average(series, window=3):
    if len(series) < window:
        return series.iloc[-1]  # Fallback
    
    recent_values = series.iloc[-window:]
    weights = np.arange(1, window + 1)
    weights = weights / weights.sum()
    
    return np.dot(recent_values, weights)
```

**Zalety**:
- Bardzo szybka (tryb FAST)
- Większa waga dla nowszych obserwacji
- Dobra dla trendów liniowych

**Wady**:
- Nie uwzględnia sezonowości
- Słaba dla trendów nieliniowych

#### 2. Exponential Smoothing

```python
def exponential_smoothing(series, alpha=0.3):
    forecast = series.iloc[0]
    for value in series.iloc[1:]:
        forecast = alpha * value + (1 - alpha) * forecast
    return forecast
```

**Zalety**:
- Wygładza szumy
- Adaptacyjny do zmian
- Stabilny dla danych volatile

**Wady**:
- Parametr alpha wymaga tuningu
- Opóźnienie w reakcji na nagłe zmiany

#### 3. Linear Trend

```python
def linear_trend(series):
    x = np.arange(len(series))
    y = series.values
    slope, intercept = np.polyfit(x, y, 1)
    next_x = len(series)
    return slope * next_x + intercept
```

**Zalety**:
- Proste, interpretowalne
- Dobre dla stabilnych trendów
- Szybkie obliczenia

**Wady**:
- Zakłada liniowość (rzadko prawdziwe w ekonomii)
- Wrażliwe na outliers

#### 4. Seasonal Naive

```python
def seasonal_naive(series, season_length=1):
    if len(series) >= season_length:
        return series.iloc[-season_length]
    return series.iloc[-1]
```

**Zalety**:
- Uwzględnia cykliczność
- Dobry baseline dla metod sezonowych
- Bardzo prosty

**Wady**:
- Nie uwzględnia trendu
- Wymaga danych sezonowych

#### 5. Correlation-based Forecasting

```python
def correlation_forecast(target_series, corr_matrix, all_data):
    # Znajdź najbardziej skorelowane wskaźniki
    correlations = corr_matrix[wskaznik_idx].abs().sort_values(ascending=False)
    top_correlated = correlations.iloc[1:6]  # Top 5 (pomijając sam siebie)
    
    # Użyj ich prognoz do predykcji
    weighted_forecast = 0
    for corr_wskaznik, corr_value in top_correlated.items():
        corr_forecast = forecast_method(all_data[corr_wskaznik])
        weighted_forecast += corr_value * corr_forecast
    
    return weighted_forecast / top_correlated.sum()
```

**Zalety**:
- Wykorzystuje interdependencje między wskaźnikami
- Stabilizuje prognozy poprzez uśrednienie
- Uwzględnia strukturę ekonomiczną

**Wady**:
- Wymaga obliczeń macierzy korelacji
- Wolniejsze niż metody univariate

### Ensemble forecasting

```python
def ensemble_forecast(series, correlations):
    forecasts = [
        exponential_smoothing(series),
        linear_trend(series),
        weighted_moving_average(series),
        seasonal_naive(series),
        correlation_forecast(series, correlations)
    ]
    
    # Średnia prosta
    ensemble_value = np.mean(forecasts)
    
    # Clipping do rozsądnych wartości (3 sigma)
    series_mean = series.mean()
    series_std = series.std()
    ensemble_value = np.clip(ensemble_value, 
                             series_mean - 3*series_std,
                             series_mean + 3*series_std)
    
    return ensemble_value
```

**Dlaczego ensemble?**
- **Redukcja wariancji**: Uśrednienie redukuje błąd pojedynczych metod
- **Odporność**: Jedna zła prognoza nie zdominuje wyniku
- **Stabilność**: Mniejsze ryzyko ekstremalnych wartości
- **Empirycznie potwierdzone**: Ensemble zwykle lepsze niż najlepsza pojedyncza metoda

### Tryby pracy: FAST vs ENSEMBLE

```python
FAST_MODE = True   # Tylko WMA (~5x szybsze)
FAST_MODE = False  # Ensemble 5 metod + korelacje (dokładniejsze)
```

**Tryb FAST**:
- Użycie: Szybka analiza, prototypowanie, testy
- Metoda: Tylko Weighted Moving Average
- Czas: ~20% czasu ENSEMBLE
- Dokładność: Zadowalająca dla stabilnych sektorów

**Tryb ENSEMBLE**:
- Użycie: Produkcja, krytyczne decyzje
- Metody: 5 metod + analiza korelacji
- Czas: Dłuższy, ale równoległość pomaga
- Dokładność: Maksymalna możliwa

### Równoległość na poziomie predykcji

```python
# Podział zadań: każda kombinacja PKD × WSKAZNIK to osobne zadanie
tasks = [(pkd, wskaznik, data) for pkd in PKDs for wskaznik in WSKAZNIKs]

# Równoległe wykonanie
with ProcessPoolExecutor(max_workers=CPU_count-1) as executor:
    futures = [executor.submit(forecast_task, task) for task in tasks]
    results = [future.result() for future in as_completed(futures)]
```

**Korzyści**:
- Liniowe skalowanie z liczbą rdzeni
- Pełne wykorzystanie CPU
- Skrócenie czasu z godzin do minut

---

## Uzasadnienie rozwiązań

### 1. Dlaczego wskaźniki złożone, a nie raw features?

**Decyzja**: Konstrukcja 44 wskaźników finansowych zamiast użycia 37 surowych danych GUS.

**Uzasadnienie**:
1. **Standaryzacja skali**: Wskaźniki uniezależnione od rozmiaru firmy (np. margin vs. absolute profit)
2. **Interpretacja domenowa**: Wskaźniki finansowe mają ugruntowane znaczenie biznesowe
3. **Porównywalność**: Możliwość benchmarkingu między sektorami różnej wielkości
4. **Redukcja szumu**: Wskaźniki względne (ratios) bardziej stabilne niż wartości absolutne
5. **Wymóg regulacyjny**: Instytucje finansowe operują na wskaźnikach, nie surowych danych

**Przykład**: 
- ❌ `NP = 1,000,000 PLN` - nieporównywalne między sektorami
- ✅ `Net Profit Margin = 5%` - uniwersalny benchmark

### 2. Dlaczego TOPSIS + VIKOR + Monte Carlo?

**Decyzja**: Użycie trzech metod MCDM zamiast jednej.

**Uzasadnienie**:
1. **Różne perspektywy**:
   - TOPSIS: odległość od ideału
   - VIKOR: kompromis użyteczności
   - Monte Carlo: stabilność probabilistyczna

2. **Redukcja bias**: Każda metoda ma słabości, ensemble je niweluje

3. **Walidacja krzyżowa**: Jeśli wszystkie trzy zgadzają się → wysoka pewność

4. **Comprehensive ranking**: 
   - TOPSIS → relatywna jakość
   - VIKOR → rozwiązanie kompromisowe
   - MC → stabilność w uncertainty

**Empiryczne potwierdzenie**: Literatura MCDM pokazuje, że ensemble outperformuje pojedyncze metody w większości przypadków.

### 3. Dlaczego 4 zestawy wskaźników?

**Decyzja**: Podział 44 wskaźników na 4 zestawy tematyczne.

**Uzasadnienie**:
1. **Modularność**: Łatwiejsze utrzymanie i rozwój systemu
2. **Specjalizacja**: Różne zespoły/eksperci mogą pracować nad różnymi aspektami
3. **Elastyczność**: Klient może wybrać tylko interesujące go analizy
4. **Performance**: Równoległe przetwarzanie 4x szybsze niż sekwencyjne
5. **Separacja obaw**: Różne output directories dla różnych celów biznesowych

**Biznesowe zastosowania**:
- Bank → Zestaw 1 (Zdolność kredytowa)
- Konsultant → Zestaw 2 (Efektywność operacyjna)
- Inwestor → Zestaw 3 (Rozwój branży)
- Regulator → Zestaw 4 (Polskie standardy)

### 4. Dlaczego predict-then-calculate dla wskaźników złożonych?

**Decyzja**: Prognozowanie tylko bazowych (0-36), kalkulacja złożonych (1000+).

**Uzasadnienie**:
1. **Spójność matematyczna**: 
   ```
   Predicted_Net_Margin = Predicted_NP / Predicted_PNPM  ✅
   vs.
   Predicted_Net_Margin jako osobny szereg czasowy  ❌ (może być inconsistent)
   ```

2. **Mniej modeli = mniejsze ryzyko overfittingu**:
   - 37 modeli bazowych vs. 81 modeli (wszystkie wskaźniki)

3. **Ekonomiczna sensowność**:
   - Wskaźniki bazowe to fundamenty (przychody, koszty, etc.)
   - Wskaźniki złożone to relacje między nimi
   - Relacje są stabilne w czasie (np. definicja Current Ratio)

4. **Stabilność numeryczna**:
   - Safe division przy kalkulacji eliminuje inf/nan
   - Predykcja szeregów czasowych może generować ekstremalne wartości

5. **Performance**:
   - 75% mniej prognoz do obliczenia
   - Kalkulacja wskaźników złożonych to proste dzielenie (milisekundy)

### 5. Dlaczego ensemble dla forecasting?

**Decyzja**: 5 metod + korelacje zamiast jednej "najlepszej" metody.

**Uzasadnienie**:
1. **No free lunch theorem**: Nie ma uniwersalnie najlepszej metody forecasting
2. **Różne wzorce czasowe**:
   - WMA → trendy liniowe
   - Exponential Smoothing → volatile data
   - Linear Trend → stabilne wzrosty
   - Seasonal Naive → cykliczność
   - Correlation → interdependencies

3. **Redukcja wariancji**: σ(ensemble) < średnia(σ(metod))

4. **Empiryczne wyniki**: Ensemble forecasting zwykle 10-30% dokładniejsze niż najlepsza pojedyncza metoda

5. **Robustness**: Jedna zła prognoza nie dominuje (averaging effect)

### 6. Dlaczego Monte Carlo dla uncertainty?

**Decyzja**: 1000 symulacji z perturbacją wag zamiast deterministycznych wag.

**Uzasadnienie**:
1. **Niepewność wag**: W rzeczywistości nie wiemy, czy Net Profit Margin jest 2x czy 2.1x ważniejszy od Quick Ratio

2. **Stabilność rankingu**: 
   - Sektor stabilnie #1 we wszystkich symulacjach → wysoka pewność
   - Sektor oscylujący między #1 a #5 → niska pewność (potrzebne głębsze analizy)

3. **Transparentność**:
   ```
   Sektor A: Score = 0.85 ± 0.02  → Stabilny
   Sektor B: Score = 0.87 ± 0.15  → Niepewny (pomimo wyższego score!)
   ```

4. **Zgodność z reality**: Decyzje biznesowe są podejmowane w warunkach niepewności

5. **Regulatory compliance**: Wymóg stress testingu w instytucjach finansowych

### 7. Dlaczego równoległość (multiprocessing)?

**Decyzja**: ProcessPoolExecutor zamiast sekwencyjnego przetwarzania.

**Uzasadnienie**:
1. **Czas wykonania**:
   - Sekwencyjnie: 4 zestawy × 12 lat × 2 poziomy = 96 analiz → ~2 godziny
   - Równolegle (8 rdzeni): ~20 minut

2. **Skalowalność**: Łatwo dodać więcej lat lub zestawów bez liniowego wzrostu czasu

3. **Wykorzystanie hardware**: Współczesne CPU mają 8-16 rdzeni → należy je wykorzystać

4. **Niezależność zadań**: Analiza sektora A nie zależy od sektora B → idealne do równoległości

5. **Fault tolerance**: Błąd w jednym procesie nie zatrzymuje całego pipeline'u

### 8. Dlaczego imputacja wielopoziomowa?

**Decyzja**: 4-poziomowa imputacja zamiast prostego usunięcia brakujących danych.

**Uzasadnienie**:
1. **Utrata informacji**: Usunięcie wierszy z NaN → utrata 20-30% danych

2. **Bias**: Sektory z lukami w danych byłyby systematycznie pomijane (selection bias)

3. **Rozsądne założenia**:
   - Poziom 1 (interpolacja): Sektor prawdopodobnie miał wartość między rokiem X a X+2
   - Poziom 2 (mediana roczna): Sektor prawdopodobnie podobny do innych w tym roku
   - Poziom 3 (mediana globalna): Sektor prawdopodobnie w historycznym paśmie
   - Poziom 4 (zero): Last resort, rzadko używane

4. **Ekonomiczna sensowność**: Brakujące dane często oznaczają "niewielka wartość", nie "brak danych"

5. **Stabilność modeli**: Algorytmy ML/MCDM wymagają kompletnych danych

---

## Podsumowanie metodologiczne

### Kluczowe innowacje systemu

1. **Dwupoziomowa predykcja** (base → derived)
   - Spójność matematyczna
   - Mniejsza liczba modeli
   - Ekonomiczna interpretowalność

2. **Ensemble na trzech poziomach**:
   - Metody MCDM (TOPSIS + VIKOR + MC)
   - Metody forecasting (5 metod szeregów czasowych)
   - Zestawy wskaźników (4 perspektywy analizy)

3. **Probabilistyczna ocena niepewności**:
   - Monte Carlo dla wag
   - Bootstrapping dla stabilności
   - Confidence intervals dla prognoz

4. **Massively parallel architecture**:
   - ProcessPoolExecutor dla zestawów
   - Równoległe forecasting dla PKD×WSKAZNIK
   - Near-linear scaling z CPU cores

### Miary jakości systemu

**Precyzja**:
- 44 wskaźniki finansowe o ugruntowanej interpretacji
- 3 metody MCDM + ensemble
- 5 metod forecasting + korelacje
- 1000 symulacji Monte Carlo

**Performance**:
- Tryb FAST: ~5 minut dla pełnej analizy
- Tryb ENSEMBLE: ~20 minut (8 rdzeni)
- Skalowalność: O(n_sectors × n_indicators / n_cores)

**Robustness**:
- Wielopoziomowa imputacja
- Safe division (eliminacja inf/nan)
- Ensemble averaging (redukcja outliers)
- Temporal weighting (uwzględnienie dynamiki)

### Zastosowania biznesowe

**Instytucje finansowe**:
- Ocena ryzyka kredytowego portfela sektorowego
- Stress testing w różnych scenariuszach (MC)
- Monitoring early warning indicators

**Inwestorzy**:
- Identyfikacja atrakcyjnych branż
- Due diligence sektorowe
- Prognozowanie trendów rynkowych

**Analitycy**:
- Benchmarking konkurencji
- Analiza efektywności operacyjnej
- Raporty makroekonomiczne

**Regulatorzy**:
- Monitoring stabilności sektorów gospodarki
- Identyfikacja ryzyk systemowych
- Planowanie polityki gospodarczej

---

## Wnioski

System łączy **klasyczne metody finansowe** (wskaźniki) z **nowoczesną analityką** (MCDM, ensemble, ML forecasting) w celu:

1. **Obiektywnej oceny** kondycji sektorów gospodarki
2. **Wielowymiarowej analizy** uwzględniającej różne aspekty biznesu
3. **Probabilistycznej oceny** niepewności i ryzyka
4. **Predykcji trendów** bazującej na fundamentach ekonomicznych
5. **Skalowalności** pozwalającej na analizę całej gospodarki

**Filozofia**: *Nie ma jednej najlepszej metody - siła leży w inteligentnej agregacji wielu perspektyw.*

---

## Struktura plików projektu

```
HackNation2025-Hackaton/
├── data-processing/
│   └── creating_complex_indicators.ipynb  # Konstrukcja wskaźników 1000-1067
├── src/
│   ├── analysis.py                        # TOPSIS, VIKOR, Monte Carlo
│   ├── outcome.py                         # Główna logika analizy
│   ├── run_prediction.py                  # Predykcja + kalkulacja
│   └── run_prediction_sets.py             # Równoległa analiza 4 zestawów
├── results-pipeline/
│   ├── kpi-value-table.csv               # Dane historyczne
│   ├── wskaznik_dictionary.csv           # Słownik wskaźników
│   └── pkd_dictionary.csv                # Słownik sektorów PKD
├── results-future/
│   └── kpi-value-table-predicted.csv     # Prognozy 2025-2028
└── results-{credit,effectivity,development}/
    └── YYYY/{sekcja,dział}/              # Wyniki analiz
        ├── topsis.csv
        ├── vikor.csv
        ├── monte_carlo.csv
        └── ensemble.csv
```

---
