# GEN1 Event-based GNN Detector

**University Project · Technische Hochschule Ingolstadt**

Research-oriented prototype for **object detection on event-based camera data** using a hybrid **Graph Neural Network (GNN) + CNN architecture**.

The project explores how asynchronous event streams can be represented as graphs, processed with **GATv2**, and converted into dense feature maps for object detection.

**Focus:** Event-based Vision · GNNs · GATv2 · Object Detection · PyTorch Geometric · GPU Memory Constraints

---

## Architecture Overview

![GEN1 GNN Detector Architecture](fig4_architecture.png)

The model follows a hybrid processing pipeline:

```text
Event Camera
     │
     ▼
Graph Construction
     │
     ▼
Node Encoder
     │
     ▼
GATv2 Backbone
     │
     ▼
Sparse-to-Dense
     │
     ├───────────────┐
     ▼               ▼
Stride 8          Stride 16
Detection Head    Detection Head
     │               │
     └───────┬───────┘
             ▼
      Object Detection
```

The GNN processes the irregular event representation, while CNN-based detection heads operate on dense spatial feature maps.

---

## My Contribution

My main contribution was the **design and implementation of the GNN-based detection architecture and training pipeline**.

Key areas:

- GATv2-based graph feature extraction
- Node feature encoding
- Sparse-to-dense feature projection
- Multi-scale CNN detection heads
- Anchor-based object detection
- Detection loss and bounding-box decoding
- GEN1 dataset integration
- GPU memory and graph-size handling
- Training and validation pipeline
- Experiment monitoring and analysis

The project gave me practical experience with the interaction between **ML architecture, graph representations and hardware constraints**.

---

## Architecture

### Event & Graph Representation

The project uses the **Prophesee GEN1 automotive detection dataset**.

Instead of conventional image frames, the input consists of asynchronous events containing:

```text
(x, y, timestamp, polarity)
```

The events are represented as a graph with neighbourhood relationships between events.

The node encoder uses:

```text
polarity
normalized x
normalized y
```

while temporal information is retained in the event/graph representation.

---

### GATv2 Backbone

The graph features are processed by a three-layer **GATv2 backbone**.

The main configuration uses:

- 3 GATv2 layers
- 4 attention heads
- residual connections
- LayerNorm
- hidden dimension of 128

The GNN extracts feature representations directly from the irregular event graph.

---

### Sparse-to-Dense Projection

The resulting graph features are converted into regular 2D feature maps using a scatter-based projection.

```text
GNN Node Features
       │
       ▼
Spatial Grid Projection
       │
       ▼
Scatter Aggregation
       │
       ▼
Dense Feature Maps
```

Two spatial resolutions are generated:

```text
Stride 8
Stride 16
```

This creates the input for the multi-scale detection heads.

---

### Object Detection

Each detection scale uses a lightweight CNN detection head.

The heads predict:

- bounding box coordinates
- objectness
- class probabilities

Predictions from both scales are decoded and combined before applying **Non-Maximum Suppression (NMS)**.

---

## Engineering Challenge: GPU Memory

A major challenge was the size of the event graphs.

Preprocessed samples can contain **hundreds of thousands of edges**, with some graphs reaching approximately:

```text
~800,000 edges
```

Processing such graphs through multiple GATv2 layers creates substantial GPU memory requirements.

To make training feasible, the number of edges is limited:

```text
~800k edges
      │
      ▼
Edge Subsampling
      │
      ▼
max_edges = 50,000
```

The baseline configuration uses:

```yaml
batch_size: 4
hidden_dim: 128
max_edges: 50000
```

This introduced an important engineering trade-off between:

```text
Graph information
      ↕
GPU memory
      ↕
Computational cost
```

This was one of the most interesting aspects of the project from a systems perspective.

---

## Training Pipeline

The project includes a complete training and validation pipeline rather than only the model implementation.

The training setup includes:

- AdamW optimizer
- weight decay
- learning-rate warmup
- cosine learning-rate scheduling
- gradient clipping
- checkpointing and resume
- TensorBoard logging
- Weights & Biases integration
- training/validation monitoring

Simplified pipeline:

```text
GEN1 Dataset
     │
     ▼
Graph DataLoader
     │
     ▼
GNN Detector
     │
     ▼
Detection Loss
     │
     ▼
Backpropagation
     │
     ▼
AdamW + Scheduler
     │
     ▼
Checkpoint / Logging
```

---

## Training Behaviour

The experiments revealed a significant **generalization problem**.

![Training vs Validation Loss](loss_curve.png)

The training loss continuously decreases, while the validation loss reaches its minimum very early and subsequently increases.

In the shown experiment:

```text
Training loss
~1.75 → ~1.0

Validation loss
~2.0 → ~2.8
```

The best validation loss occurs around **epoch 1**, indicating strong overfitting in this configuration.

Rather than presenting this as a successful benchmark result, I use it as an example of how the architecture was **experimentally evaluated and its limitations analysed**.

---

## Key Takeaways

This project gave me practical experience in:

- designing and implementing a non-trivial GNN architecture
- working with irregular event-based data
- combining GNN and CNN processing
- dealing with large graph structures
- analysing GPU memory constraints
- building a complete training pipeline
- interpreting training and validation behaviour
- identifying overfitting and architectural limitations

A key lesson was that **ML architecture and hardware constraints cannot be considered independently**. Graph size, feature dimensions and batch size directly affect memory consumption and computational feasibility.

---

## Technologies

**Machine Learning**
- PyTorch
- PyTorch Geometric
- Torch Scatter
- Torchvision

**Architecture**
- Graph Neural Networks
- GATv2
- CNNs
- Multi-scale object detection
- Anchor-based detection

**Training**
- AdamW
- Learning-rate scheduling
- Gradient clipping
- TensorBoard
- Weights & Biases

**Compute**
- NVIDIA GPU
- CUDA 11.8

---

## Project Structure

```text
GEN1-GNN-Detector/
│
├── README.md
├── fig4_architecture.png
├── loss_curve.png
│
├── gen1_gnn_detector.py
│   └── GNN architecture, detection heads and loss
│
├── gen1_dataset.py
│   └── Dataset interface and graph processing
│
├── train.py
│   └── Training and validation pipeline
│
├── config.yaml
│   └── Experiment configuration
│
└── requirements.txt
    └── Python dependencies
```

---

## Installation

The project was developed with:

```text
Python 3.10
PyTorch 2.1
CUDA 11.8
PyTorch Geometric
Torch Scatter
Torch Sparse
Torchvision
TensorBoard
Weights & Biases
```

Install the required Python packages:

```bash
pip install -r requirements.txt
```

---

## Dataset

The **GEN1 dataset is not included in this repository**.

The dataset must be obtained separately and preprocessed before training.

Expected structure:

```text
data/
├── train/
├── val/
└── processed/
```

---

## Training

After preparing the dataset and configuring `config.yaml`:

```bash
python train.py \
    --data data \
    --config config.yaml \
    --output runs/
```

Training logs can be inspected using TensorBoard:

```bash
tensorboard --logdir runs/
```

---

## References

The implementation builds on concepts from:

- **AEGNN** – Event-based graph neural network processing
- **DAGR** – Event-based object detection
- **Prophesee GEN1** – Event-based automotive detection dataset
- **PyTorch Geometric** – Graph neural network framework
- **PyTorch** – Deep learning framework

---

## Project Context

**Project Type:** University Project  
**Institution:** Technische Hochschule Ingolstadt  
**Topic:** Event-based Object Detection with Graph Neural Networks  
**Primary Contribution:** GNN architecture and training pipeline

This project was developed as an academic/research-oriented prototype for investigating graph-based processing of event-camera data and its integration with conventional object detection.
