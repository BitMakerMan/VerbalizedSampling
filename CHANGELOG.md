# Changelog

Tutte le modifiche importanti a Verbalized Sampling saranno documentate in questo file.

Il formato è basato su [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2025-10-27

### Added
- **Implementazione completa** della tecnica Verbalized Sampling
- **Supporto Ollama nativo** con libreria ufficiale
- **Fallback HTTP automatico** se la libreria Ollama non è disponibile
- **Modalità manuale** per input da altri LLM (ChatGPT, Claude, etc.)
- **XML parsing robusto** con fallback regex
- **Probability sampling** basato sulle probabilità verbali del modello
- **CLI completa** con tutte le opzioni necessarie
- **Retry mechanism** per chiamate API fallite
- **Normalizzazione automatica** delle probabilità
- **Multi-piattaforma** (Windows, Linux, macOS)

### Features
- Genera 5-10+ risposte diverse invece di una singola
- Campionamento probabilistico basato sulle probabilità del modello
- Supporto per temperature configurabile (0.0-1.0)
- Supporto per modelli Ollama multipli
- Visualizzazione completa dei risultati con statistiche
- Messaggi di debug dettagliati per troubleshooting

### Technical
- Compatibile con Python 3.8+
- Architettura modulare con classi separate
- Gestione errori robusta
- Timeout configurabile (120 secondi default)
- Connection pooling per performance

### Documentation
- README.md completo con esempi e troubleshooting
- Requirements.txt semplificato
- Licenza MIT
- Supporto per virtual environment

### Known Issues
- La libreria Ollama ufficiale potrebbe avere problemi di configurazione su alcuni sistemi Windows (risolto con fallback HTTP)

### Credits
- Implementazione originale: Craicek
- Basato su paper: "Verbalized Sampling: How to Mitigate Mode Collapse and Unlock LLM Diversity"
- Fonte originale: CHATS-lab