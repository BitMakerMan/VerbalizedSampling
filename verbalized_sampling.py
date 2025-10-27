#!/usr/bin/env python3
"""
Verbalized Sampling Implementation
=================================

Implementazione Python fedele della tecnica "Verbalized Sampling" come descritta nel paper:
"Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity"

Questa implementazione segue fedelmente l'approccio del repository CHATS-lab:
https://github.com/CHATS-lab/verbalized-sampling

Basato su CHATS-lab implementation - Apache License 2.0
"""

import random
import re
import xml.etree.ElementTree as ET
from typing import List, Optional, Tuple
from dataclasses import dataclass
import argparse

# Import opzionali per Ollama
try:
    import requests
    import urllib3
    URLLIB3_AVAILABLE = True
except ImportError:
    URLLIB3_AVAILABLE = False
    requests = None
    urllib3 = None

# Import libreria ufficiale Ollama (se disponibile)
try:
    import ollama
    OLLAMA_LIB_AVAILABLE = True
    # Test se la libreria funziona davvero
    try:
        test = ollama.list()
        OLLAMA_LIB_AVAILABLE = True
    except Exception:
        OLLAMA_LIB_AVAILABLE = False
        ollama = None
except ImportError:
    OLLAMA_LIB_AVAILABLE = False
    ollama = None


@dataclass
class Item:
    """
    Item class representing a single candidate with probability.
    Fedele all'implementazione CHATS-lab.
    """
    text: str
    probability: float

    def __eq__(self, other):
        if not isinstance(other, Item):
            return False
        return self.text == other.text

    def __hash__(self):
        return hash(self.text)


class DiscreteDist:
    """
    Classe di distribuzione discreta per gestire i risultati del verbalized sampling.
    Fedele all'implementazione DiscreteDist di CHATS-lab.
    """

    def __init__(self, items: List[Item]):
        self.items = items
        # Ordina per probabilità (decrescente) come nell'originale
        self.items.sort(key=lambda x: x.probability, reverse=True)

    def sample(self, seed: Optional[int] = None) -> Item:
        """Esegue il sampling dalla distribuzione."""
        if seed is not None:
            random.seed(seed)

        if not self.items:
            raise ValueError("Impossibile eseguire sampling da distribuzione vuota")

        # Estrae testi e probabilità
        texts = [item.text for item in self.items]
        probs = [item.probability for item in self.items]

        # Esegue sampling usando scelta pesata casuale
        selected_text = random.choices(texts, weights=probs, k=1)[0]

        # Trova e restituisce l'item corrispondente
        for item in self.items:
            if item.text == selected_text:
                return item

        return self.items[0]

    def argmax(self) -> Item:
        """Restituisce l'item con probabilità più alta."""
        if not self.items:
            raise ValueError("Impossibile ottenere argmax da distribuzione vuota")
        return self.items[0]

    def __len__(self):
        return len(self.items)

    def __iter__(self):
        return iter(self.items)


def verbalize(prompt: str, k: int = 5, tau: float = 0.12, temperature: float = 0.9,
              model: str = "gpt-4o-mini") -> DiscreteDist:
    """
    Funzione verbalize principale - fedele all'implementazione CHATS-lab.

    Args:
        prompt: Prompt di input per il modello
        k: Numero di candidati da generare (default 5)
        tau: Soglia di probabilità (default 0.12)
        temperature: Temperatura di sampling (default 0.9)
        model: Modello da usare (default gpt-4o-mini)

    Returns:
        DiscreteDist: Distribuzione di candidati con probabilità
    """
    # Costruisce il prompt verbalized sampling seguendo il template CHATS-lab
    instructions = f"""<instructions>
Generate {k} responses to the user query, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than {tau}.
</instructions>

{prompt}"""

    # Per implementazione Ollama
    if OLLAMA_LIB_AVAILABLE or URLLIB3_AVAILABLE:
        try:
            response = _call_ollama_api(instructions, model, temperature)
            items = _parse_verbalized_response(response)
            return DiscreteDist(items)
        except Exception as e:
            print(f"[ERROR] Impossibile chiamare il modello: {e}")
            raise

    # Fallback: restituisce distribuzione vuota
    return DiscreteDist([])


def _call_ollama_api(prompt: str, model: str, temperature: float) -> str:
    """Chiama API Ollama per inferenza del modello."""
    # Prova prima la libreria ufficiale
    if OLLAMA_LIB_AVAILABLE:
        try:
            response = ollama.generate(
                model=model,
                prompt=prompt,
                options={
                    'temperature': temperature,
                    'num_predict': 2000
                }
            )
            return response['response']
        except Exception:
            pass

    # Fallback a HTTP
    if URLLIB3_AVAILABLE:
        try:
            payload = {
                "model": model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": temperature,
                    "num_predict": 2000
                }
            }

            response = requests.post(
                "http://localhost:11434/api/generate",
                json=payload,
                timeout=120,
                verify=False
            )

            if response.status_code == 200:
                result = response.json()
                return result["response"]
            else:
                raise Exception(f"HTTP {response.status_code}: {response.text}")
        except Exception as e:
            raise Exception(f"Chiamata HTTP Ollama fallita: {e}")

    raise Exception("Nessun backend Ollama disponibile")


def _parse_verbalized_response(response: str) -> List[Item]:
    """
    Parso la risposta verbalized sampling in items.
    Segue l'approccio di parsing CHATS-lab.
    """
    items = []

    # Aggiunge elemento radice per XML valido
    xml_content = f"<root>{response}</root>"

    try:
        root = ET.fromstring(xml_content)

        for resp_elem in root.findall('response'):
            # Estrae testo
            text_elem = resp_elem.find('text')
            if text_elem is None or text_elem.text is None:
                continue

            text = text_elem.text.strip()
            if not text:
                continue

            # Estrae probabilità
            prob_elem = resp_elem.find('probability')
            if prob_elem is None or prob_elem.text is None:
                continue

            try:
                prob_text = prob_elem.text.strip()
                if prob_text.endswith('%'):
                    probability = float(prob_text.rstrip('%')) / 100.0
                else:
                    probability = float(prob_text)

                if 0.0 <= probability <= 1.0:
                    items.append(Item(text=text, probability=probability))

            except (ValueError, TypeError):
                continue

    except ET.ParseError:
        # Fallback a parsing regex
        items = _fallback_parsing(response)

    # Normalizza le probabilità
    if items:
        total_prob = sum(item.probability for item in items)
        if total_prob > 0 and abs(total_prob - 1.0) > 0.01:
            for item in items:
                item.probability = item.probability / total_prob

    return items


def _fallback_parsing(response: str) -> List[Item]:
    """Parsing regex fallback per XML malformato."""
    items = []

    # Pattern per <response>...</response>
    pattern = r'<response>(.*?)</response>'
    matches = re.findall(pattern, response, re.DOTALL)

    for match in matches:
        # Estrae testo
        text_match = re.search(r'<text>(.*?)</text>', match, re.DOTALL)
        if not text_match:
            continue

        text = text_match.group(1).strip()
        if not text:
            continue

        # Estrae probabilità
        prob_match = re.search(r'<probability>(.*?)</probability>', match)
        if not prob_match:
            continue

        try:
            prob_text = prob_match.group(1).strip()
            if prob_text.endswith('%'):
                probability = float(prob_text.rstrip('%')) / 100.0
            else:
                probability = float(prob_text)

            if 0.0 <= probability <= 1.0:
                items.append(Item(text=text, probability=probability))

        except (ValueError, TypeError):
            continue

    return items


class VerbalizedSampler:
    """
    Wrapper legacy per compatibilità backward.
    Mantiene l'interfaccia originale usando internamente l'approccio CHATS-lab.
    """

    def __init__(self):
        self.base_url = "http://localhost:11434"
        if URLLIB3_AVAILABLE:
            urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)

    def create_prompt(self, question: str, num_responses: int = 7) -> str:
        """Crea prompt verbalized sampling seguendo il template CHATS-lab."""
        return f"""<instructions>
Generate {num_responses} responses to the user query, each within a separate <response> tag. Each <response> must include a <text> and a numeric <probability>.
Please sample at random from the tails of the distribution, such that the probability of each response is less than 0.15.
</instructions>

{question}"""

    def parse_xml_responses(self, raw_output: str) -> List[Item]:
        """Parso le risposte usando l'approccio CHATS-lab."""
        return _parse_verbalized_response(raw_output)

    def sample_response(self, responses: List[Item]) -> Item:
        """Esegue sampling dalle risposte."""
        dist = DiscreteDist(responses)
        return dist.sample()

    def process_ollama_input(self, question: str, num_responses: int = 7,
                           model: str = "llama3.2", temperature: float = 0.8) -> Tuple[List[Item], Item]:
        """Processa input usando Ollama con approccio CHATS-lab."""
        prompt = self.create_prompt(question, num_responses)

        try:
            response = _call_ollama_api(prompt, model, temperature)
            items = self.parse_xml_responses(response)

            if not items:
                raise ValueError("Nessuna risposta valida trovata")

            sampled = self.sample_response(items)
            return items, sampled

        except Exception as e:
            raise Exception(f"Errore nel processare input Ollama: {e}")

    def display_results(self, responses: List[Item], sampled: Item):
        """Visualizza i risultati in stile CHATS-lab."""
        print("\n" + "="*80)
        print("VERBALIZED SAMPLING RESULTS (CHATS-lab style)")
        print("="*80)

        print(f"\nTrovate {len(responses)} risposte:")
        print("-"*60)

        for i, response in enumerate(responses, 1):
            marker = "[TARGET]" if response == sampled else "  "
            print(f"{marker} {i}. {response.text}")
            print(f"   Probability: {response.probability:.3f} ({response.probability*100:.1f}%)")
            print()

        print("="*80)
        print("SAMPLED RESPONSE:")
        print("="*80)
        print(f"[OK] {sampled.text}")
        print(f"[STATS] Probability: {sampled.probability:.3f} ({sampled.probability*100:.1f}%)")
        print("="*80)


# Funzioni di supporto Ollama legacy
def check_ollama_status() -> bool:
    """Verifica se Ollama è disponibile."""
    if OLLAMA_LIB_AVAILABLE:
        try:
            ollama.list()
            return True
        except:
            pass

    if URLLIB3_AVAILABLE:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=5, verify=False)
            return response.status_code == 200
        except:
            pass

    return False


def get_available_ollama_models() -> List[str]:
    """Ottiene i modelli Ollama disponibili."""
    if OLLAMA_LIB_AVAILABLE:
        try:
            models = ollama.list()
            return [model['name'] for model in models.get('models', [])]
        except:
            pass

    if URLLIB3_AVAILABLE:
        try:
            response = requests.get("http://localhost:11434/api/tags", timeout=10, verify=False)
            if response.status_code == 200:
                data = response.json()
                return [model["name"] for model in data.get("models", [])]
        except:
            pass

    return []


def main():
    """Funzione principale con interfaccia CLI."""
    parser = argparse.ArgumentParser(description="Verbalized Sampling - Implementazione fedele CHATS-lab")
    parser.add_argument("--question", "-q", type=str,
                       default="Suggerisci un nome creativo per una pizzeria",
                       help="La domanda da porre al modello")
    parser.add_argument("--responses", "-n", type=int, default=7,
                       help="Numero di risposte da generare")
    parser.add_argument("--ollama", action="store_true",
                       help="Usa API Ollama invece della funzione verbalize")
    parser.add_argument("--model", "-m", type=str, default="llama3.2",
                       help="Modello Ollama da usare")
    parser.add_argument("--temperature", "-t", type=float, default=0.8,
                       help="Temperatura per generazione (0.0-1.0)")
    parser.add_argument("--k", type=int, default=5,
                       help="Numero di candidati per funzione verbalize")
    parser.add_argument("--tau", type=float, default=0.12,
                       help="Soglia di probabilità per funzione verbalize")
    parser.add_argument("--list-models", action="store_true",
                       help="Lista i modelli Ollama disponibili")
    parser.add_argument("--check-ollama", action="store_true",
                       help="Verifica se Ollama è in esecuzione")

    args = parser.parse_args()

    # Funzioni utility Ollama
    if args.list_models:
        print("Controllo modelli Ollama...")
        if check_ollama_status():
            models = get_available_ollama_models()
            if models:
                print(f"\n[LIST] Modelli disponibili ({len(models)}):")
                for i, model in enumerate(models, 1):
                    print(f"  {i}. {model}")
            else:
                print("[X] Nessun modello trovato. Installa con: ollama pull <model>")
        else:
            print("[X] Ollama non è in esecuzione. Avvia con: ollama serve")
        return 0

    if args.check_ollama:
        print("Controllo stato Ollama...")
        if check_ollama_status():
            print("[OK] Ollama è in esecuzione e raggiungibile")
            models = get_available_ollama_models()
            if models:
                print(f"[STATS] {len(models)} modelli disponibili")
        else:
            print("[X] Ollama non è in esecuzione o non è raggiungibile")
            print("[INFO] Avvia Ollama con: ollama serve")
        return 0

    print("[CHATS-LAB] VERBALIZED SAMPLING IMPLEMENTATION")
    print("="*80)
    print(f"Domanda: {args.question}")

    try:
        if args.ollama:
            # Usa interfaccia Ollama legacy
            sampler = VerbalizedSampler()
            responses, sampled = sampler.process_ollama_input(
                question=args.question,
                num_responses=args.responses,
                model=args.model,
                temperature=args.temperature
            )
            sampler.display_results(responses, sampled)
        else:
            # Usa funzione verbalize CHATS-lab
            print(f"Parametri: k={args.k}, tau={args.tau}, temperature={args.temperature}")
            print("="*80)

            dist = verbalize(
                prompt=args.question,
                k=args.k,
                tau=args.tau,
                temperature=args.temperature,
                model=args.model
            )

            if len(dist) == 0:
                print("[ERROR] Nessuna risposta generata")
                return 1

            # Mostra tutte le risposte
            print(f"\n[RESULTS] Generati {len(dist)} risposte:")
            print("-"*60)
            for i, item in enumerate(dist, 1):
                print(f"  {i}. {item.text}")
                print(f"     Probability: {item.probability:.3f}")

            # Mostra risposta campionata
            sampled = dist.sample()
            print(f"\n[SAMPLED] {sampled.text}")
            print(f"[PROBABILITY] {sampled.probability:.3f}")

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())