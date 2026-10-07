# GEN1 Event-based GNN Detector

Event-basierter Objektdetektor für den Prophesee GEN1 Datensatz.

Die Architektur kombiniert:

* Graph Neural Networks (GNNs)
* Event-basierte Verarbeitung
* Sparse-to-Dense Projektion
* CNN-basierte Detection Heads

Basierend auf Konzepten aus:

* AEGNN
* DAGR
* PyTorch Geometric

---

# Projektstruktur

```text
project/
├── train.py
├── gen1_dataset.py
├── gen1_gnn_detector.py
├── config.yaml
├── requirements.txt
│
├── gen1/
│   ├── train/
│   │    ├── *.dat
│   │    ├── *.npy
│   │
│   ├── val/
│   │    ├── *.dat
│   │    ├── *.npy
│   │
│   ├── processed/
│   │    ├── train/
│   │    │    ├── *.pkl
│   │    ├── val/
│   │         ├── *.pkl
```

---

# Voraussetzungen

Empfohlen:

* Python 3.10
* CUDA 11.8
* NVIDIA GPU

---

# Installation

Virtuelle Umgebung erstellen:

```bash
python -m venv venv
```

Aktivieren:

Windows:

```bash
venv\\Scripts\\activate
```

Pakete installieren:

```bash
pip install -r requirements.txt
```

---

# Training starten

Beispiel:

```bash
python train.py --data path/to/gen1
```

oder:

```bash
python train.py --config config.yaml --data path/to/gen1
```

---

# Wichtige Hinweise

## Dataset

Das Projekt verwendet:

* `.dat` Dateien für rohe Eventdaten
* `.pkl` Dateien für vorverarbeitete Graph Samples
* `.npy` Dateien für zusätzliche GEN1 Daten

Die `.pkl` Dateien enthalten:

* Event-Indizes
* Bounding Boxes
* vorberechnete Graphstrukturen
* Referenzen auf die ursprünglichen `.dat` Dateien

---

# Architektur

```text
Events
→ Graph Konstruktion
→ Node Encoder
→ GATv2 Backbone
→ Sparse-to-Dense Projektion
→ CNN Detection Head
→ Bounding Box Prediction
```

---

# Wichtige Hyperparameter

In `config.yaml`:

```yaml
batch_size
hidden_dim
max_edges
learning_rate
```

---

# Speicherhinweise

Event-Graphen können sehr groß werden.

Deshalb:

* `max_edges` reduziert die Anzahl der Kanten
* große Batchgrößen können den GPU Speicher überschreiten

---

# Quellen

AEGNN:
https://github.com/uzh-rpg/aegnn

DAGR:
https://github.com/uzh-rpg/dagr

Prophesee GEN1:
https://www.prophesee.ai/2020/01/24/gen1-automotive-detection-dataset/
