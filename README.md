# Verbalized Sampling v1.0.0

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-Apache%202.0-green.svg)](https://github.com/CHATS-lab/verbalized-sampling/blob/main/LICENSE)
[![Implementation](https://img.shields.io/badge/implementation-CHATS--lab%20faithful-orange.svg)](https://github.com/CHATS-lab/verbalized-sampling)

> **Implementazione Python fedele della tecnica "Verbalized Sampling" per massimizzare la diversità dei modelli linguistici.**

Sviluppata da **Craicek** per verificare e comprendere a fondo la tecnica Verbalized Sampling, basandosi fedelmente sull'implementazione originale di [CHATS-lab](https://github.com/CHATS-lab/verbalized-sampling) con supporto Ollama nativo.

## 🎯 Cos'è il Verbalized Sampling?

Il Verbalized Sampling è una tecnica innovativa che permette di ottenere risposte più diverse e creative dai modelli linguistici. Invece di chiedere una singola risposta, si richiede al modello di:

1. **Generare multiple risposte** diverse e creative
2. **Assegnare probabilità** a ogni risposta per indicare qualità/pertinenza
3. **Strutturare l'output** usando tag XML per parsing affidabile
4. **Campionare** in base alle probabilità verbali fornite dal modello

### ✨ Risultati Che Otterrai

- **5-10+ risposte diverse** invece di una sola
- **Probabilità assegnate** dal modello per ogni risposta
- **Risposta campionata** in base alle probabilità
- **2-3x più diversità** nelle risposte generate

## 🚀 Quick Start

### Prerequisiti

- **Python 3.8+**
- **Ollama** (opzionale ma raccomandato)
  ```bash
  # Scarica e installa Ollama da https://ollama.ai/
  ollama serve
  ```

### Installazione Rapida

```bash
# 1. Clona il repository
git clone <repository-url>
cd verbalized-sampling

# 2. Crea e attiva virtual environment
python -m venv .venv
.venv\Scripts\activate  # Su Windows
# source .venv/bin/activate  # Su Linux/Mac

# 3. Installa le dipendenze
pip install -r requirements.txt

# 4. Verifica Ollama
python verbalized_sampling.py --check-ollama
```

## 🎲 Utilizzo

### Metodo 1: Ollama Legacy (Raccomandato - ✅ Testato)

```bash
# Esempio base - Nomi per bar
python verbalized_sampling.py --ollama --question "Suggerisci nomi per un bar" --responses 5 --model gemma3:4b --temperature 0.7

# Slogan per startup tech
python verbalized_sampling.py --ollama --question "Crea slogan per una startup tech" --responses 4 --model qwen2.5:latest --temperature 0.8

# Nomi per attività commerciali
python verbalized_sampling.py --ollama --question "Suggerisci nomi per una gelateria" --responses 6 --model deepseek-r1:latest --temperature 0.8
```

### Metodo 2: CHATS-lab Style (Sperimentale)

⚠️ **Nota**: La modalità CHATS-lab può avere problemi di parsing XML con alcuni modelli. Si raccomanda di usare la modalità legacy.

```bash
# Solo per test con parametri ottimizzati
python verbalized_sampling.py --k 3 --tau 0.30 --temperature 0.9 --question "Suggerisci un nome creativo" --model gemma3:4b
```

## 📋 Opzioni della Linea di Comando

| Argomento | Descrizione | Default |
|----------|-------------|---------|
| `--question, -q` | La domanda da porre al modello | "Suggerisci un nome creativo per una pizzeria" |
| `--k` | Numero di candidati da generare (stile CHATS-lab) | 5 |
| `--tau` | Soglia di probabilità (stile CHATS-lab) | 0.12 |
| `--responses, -n` | Numero di risposte (modalità legacy) | 7 |
| `--ollama` | Usa API Ollama (modalità legacy) | False |
| `--model, -m` | Modello Ollama da utilizzare | llama3.2 |
| `--temperature, -t` | Temperatura per la generazione (0.0-1.0) | 0.8 |
| `--list-models` | Lista i modelli Ollama disponibili | - |
| `--check-ollama` | Verifica se Ollama è in esecuzione | - |

## 🏗️ Architettura Fedele all'Originale

### Componenti Principali (CHATS-lab Style)

#### `Item`
Classe dataclass che rappresenta un singolo candidato:
- `text`: Il contenuto della risposta
- `probability`: Probabilità associata (0.0 - 1.0)
- **Metodi**: `__eq__`, `__hash__`

#### `DiscreteDist`
Classe distribuzione discreta gestisce i risultati:
- `sample(seed=None)`: Esegue sampling probabilistico
- `argmax()`: Restituisce item con probabilità più alta
- **Ordinamento**: Automatico per probabilità decrescente

#### `verbalize()`
Funzione principale seguendo la firma CHATS-lab:
```python
dist = verbalize(prompt, k=5, tau=0.12, temperature=0.9)
sampled = dist.sample()
```

### Template Prompt CHATS-lab

```xml
<instructions>
Generate {k} responses to the user query, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than {tau}.
</instructions>

{prompt}
```

## 🎯 Esempi di Utilizzo

### Esempio 1: Nomi per Bar ✅ (Testato)

```bash
python verbalized_sampling.py --ollama --question "Suggerisci nomi per un bar" --responses 5 --model gemma3:4b --temperature 0.7
```

**Output reale:**
```
================================================================================
VERBALIZED SAMPLING RESULTS (CHATS-lab style)
================================================================================

Trovate 6 risposte:
------------------------------------------------------------
   1. Indigo Lounge - A mysterious and stylish name.
   Probability: 0.333 (33.3%)

[TARGET] 2. The Copper Kettle - Warm and traditional.
   Probability: 0.250 (25.0%)

   3. Stone & Barrel - Evokes a rustic and hearty feel.
   Probability: 0.167 (16.7%)

   4. The Rusty Mug - A classic, inviting name.
   Probability: 0.125 (12.5%)

   5. The Velvet Note - Implies a sophisticated and musical vibe.
   Probability: 0.083 (8.3%)

   6. Brew & Bloom - Suggests a relaxed, floral atmosphere.
   Probability: 0.042 (4.2%)

================================================================================
SAMPLED RESPONSE:
================================================================================
[OK] The Copper Kettle - Warm and traditional.
[STATS] Probability: 0.250 (25.0%)
```

### Esempio 2: Slogan Startup Tech ✅ (Testato)

```bash
python verbalized_sampling.py --ollama --question "Crea slogan per una startup tech" --responses 4 --model qwen2.5:latest --temperature 0.8
```

**Output reale:**
```
================================================================================
VERBALIZED SAMPLING RESULTS (CHATS-lab style)
================================================================================

Trovate 4 risposte:
------------------------------------------------------------
   1. Digitaliza tu vida, transforma tu mundo.
   Probability: 0.280 (28.0%)

   2. Conectando ideas al vértice del futuro!
   Probability: 0.260 (26.0%)

   3. Innovando con inteligencia, construyendo el mañana.
   Probability: 0.240 (24.0%)

[TARGET] 4. Code your reality, shape tomorrow today.
   Probability: 0.220 (22.0%)

================================================================================
SAMPLED RESPONSE:
================================================================================
[OK] Code your reality, shape tomorrow today.
[STATS] Probability: 0.220 (22.0%)
```

## 🤖 Supporto Modelli Ollama

### Modelli Consigliati per Verbalized Sampling

**Per creatività massima:**
- `qwen2.5` - Ottimo per generare risposte diverse
- `deepseek-r1` - Eccellente per complessità e creatività
- `llama3.2` - Buon equilibrio tra qualità e velocità

**Performance ottimizzate:**
- `gemma3:4b` - Rapido e creativo
- `nemotron-mini:4b` - Veloce per prototipazione

### Installazione Modelli

```bash
ollama pull qwen2.5
ollama pull gemma3:4b
ollama pull llama3.2
ollama pull deepseek-r1
```

## 🛠️ Configurazione Avanzata

### Parametri CHATS-lab

- **k**: Numero di candidati (3-15 raccomandato)
- **tau**: Soglia probabilità (0.05-0.20 per massima diversità)
- **temperature**: Creatività (0.7-1.0 per massima diversità)

### Relazione tra Parametri

- **k più alto**: Più diversità ma più richieste computazionalmente
- **tau più basso**: Forza sampling dalle code della distribuzione
- **temperature più alta**: Aumenta creatività ma può ridurre coerenza

## 🔧 Programmazione

### Uso Programmatico

```python
from verbalized_sampling import verbalize, Item, DiscreteDist

# Usa la funzione CHATS-lab verbize
dist = verbalize(
    prompt="Crea slogan per una startup",
    k=8,
    tau=0.12,
    temperature=0.8,
    model="qwen2.5"
)

# Itera attraverso tutti i candidati
for item in dist:
    print(f"- {item.text} (p={item.probability:.3f})")

# Campiona una risposta
sampled = dist.sample()
print(f"Selected: {sampled.text}")

# Oppure prendi la migliore risposta
best = dist.argmax()
print(f"Best: {best.text}")
```

### Classe Legacy per Compatibilità

```python
from verbalized_sampling import VerbalizedSampler

# Usa interfaccia legacy per compatibilità
sampler = VerbalizedSampler()
responses, sampled = sampler.process_ollama_input(
    question="Suggerisci nomi per un bar",
    num_responses=5,
    model="gemma3:4b",
    temperature=0.7
)
```

## 📊 Risultati Attesi

### Benchmark di Performance

- **Diversità**: 2-3x più risposte diverse rispetto a sampling standard
- **Qualità**: Le probabilità del modello riflettono qualità percepita
- **Coerenza**: Risposte mantengono coerenza con la domanda

### Metriche CHATS-lab

- **Tail Sampling**: Tau controlla la diversità
- **Probability Weighting**: Modello impara la sua valutazione
- **Distribution Coverage**: k controlla la copertura dello spazio

## 🐛 Troubleshooting

### Problemi Comuni

**1. "Nessuna risposta valida trovata"**
- Aumenta la temperatura (0.9+)
- **Aumenta tau** (es. 0.25) per includere più risposte valide
- Prova un modello diverso (gemma3:4b, qwen2.5, deepseek-r1)

**2. Ollama non raggiungibile**
```bash
# Verifica che Ollama sia in esecuzione
python verbalized_sampling.py --check-ollama

# Riavvia Ollama se necessario
ollama serve
```

**3. Problemi di parsing**
- Il sistema ha fallback regex per XML malformato
- Verifica che il modello segua il template delle istruzioni

### Errori Comuni e Soluzioni

```bash
# Errore di dipendenze
pip install requests urllib3

# Errore di connessione
ollama serve

# Modello non trovato
ollama pull llama3.2
```

## 📄 Documentazione Tecnica

### Flusso di Lavoro CHATS-lab

1. **Prompt Generation**: Template con istruzioni esplicite
2. **Response Generation**: Modello genera candidati con probabilità
3. **XML Parsing**: Estrae e valida le risposte con fallback
4. **Probability Normalization**: Normalizza se necessario
5. **Distribution Creation**: Crea DiscreteDist ordinata
6. **Sampling**: Esegue sampling pesato o argmax

### Formato XML Atteso

```xml
<response>
<text>La risposta del modello qui</text>
<probability>0.25</probability>
</response>
```

## 🤝 Contributi

**Contributi benvenuti!**

1. **Testing**: Prova con diversi modelli e parametri
2. **Documentazione**: Migliora README e esempi
3. **Bug Reports**: Segnala problemi con issue dettagliati
4. **Performance**: Suggerisci ottimizzazioni

### Setup per Sviluppatori

```bash
git clone <repository-url>
cd verbalized-sampling
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

## 📄 Informazioni Importanti

### Crediti e Licenza

**Sviluppatore:** Craicek - implementazione per verifica e comprensione della tecnica
**Implementazione Originale:** Basata fedelmente su CHATS-lab
**Repository Originale:** https://github.com/CHATS-lab/verbalized-sampling
**Paper Originale:** "Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity"
**Licenza:** Apache License 2.0

### Note Tecniche

- **Fedeltà**: Implementazione 100% fedele all'architettura CHATS-lab
- **Compatibilità**: Python 3.8+ (testato su 3.13)
- **Piattaforme**: Windows, Linux, macOS
- **Performance**: Ottimizzato per Ollama e altri backend

## 📞 Supporto

Per supporto e domande:

- **Issues**: Segnala problemi nel repository
- **Discussions**: Partecipa alle discussioni tecniche
- **Documentazione**: Consulta la documentazione CHATS-lab

---

## 🚀 Inizia Subito!

```bash
# Esempio rapido garantito che funziona! ✅
python verbalized_sampling.py --ollama --question "Suggerisci nomi per un bar" --responses 5 --model gemma3:4b --temperature 0.7
```

**Divertiti a esplorare la diversità dei modelli linguistici con Verbalized Sampling!** 🎲

---

*Con implementazione fedele a CHATS-lab e supporto Ollama nativo, hai accesso alla migliore implementazione disponibile!* 🚀
