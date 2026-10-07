"""
gen1_gnn_detector.py
====================

Event-basierter GNN Objektdetektor für Prophesee GEN1.

Architektur:

Events
  → Graph Konstruktion
  → Node Encoder
  → GATv2 Backbone
  → Sparse-to-Dense Projektion
  → CNN Detection Head
  → Bounding Box Prediction

Kombiniert Konzepte aus:
- AEGNN
- DAGR
- YOLO-style Dense Detection
- PyTorch Geometric

------------------------------------------------------------
Anpassungen für eigenes Training
------------------------------------------------------------

1. Sensorauflösung prüfen:
   H und W müssen zur Eventkamera passen.

2. Anzahl Klassen anpassen:
   NUM_CLASSES ändern.

3. Anchor Größen prüfen:
   DetectionLoss.ANCHORS_S
   DetectionLoss.ANCHORS_L

4. GPU Speicher:
   hidden_dim und max_edges beeinflussen Speicherverbrauch.

5. Event Dichte:
   Bei sehr dichten Event Streams max_edges reduzieren.

------------------------------------------------------------
Wissenschaftlicher Hintergrund
------------------------------------------------------------

Die Architektur verwendet Graph Attention Networks (GATv2)
für die Verarbeitung asynchroner Eventdaten.

Die Sparse-to-Dense Projektion transformiert unregelmäßige
Event-Graphen in ein reguläres 2D Feature Grid, wodurch
klassische CNN Detection Heads verwendet werden können.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GATv2Conv
from torch_scatter import scatter

# ── Sensor Konstanten ──────────────────────────────────────
H           = 240
W           = 304
NUM_CLASSES = 2
STRIDE_S    = 8    # Small Scale:  H/8=30,  W/8=38
STRIDE_L    = 16   # Large Scale:  H/16=15, W/16=19


# =============================================================
# NODE ENCODER
# =============================================================
class NodeEncoder(nn.Module):
    """
    Input: polarity (1) + x_norm (1) + y_norm (1) = 3 Features
    Output: hidden_dim
    """
    def __init__(self, hidden_dim: int):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

    def forward(self, data: Data) -> torch.Tensor:
        p   = data.x                          # (N, 1)
        x_n = data.pos[:, 0:1] / W           # normalisiert
        y_n = data.pos[:, 1:2] / H
        return self.mlp(torch.cat([p, x_n, y_n], dim=1))

# Der NodeEncoder transformiert rohe Eventinformationen
# in einen höherdimensionalen Feature Space.
#
# Verwendete Features:
#   - Polarität
#   - normalisierte x-Koordinate
#   - normalisierte y-Koordinate
#
# Zeit wird separat über die Graphstruktur modelliert.

# =============================================================
# GNN BACKBONE
# =============================================================
class GNNBackbone(nn.Module):
    """
    3x GATv2Conv mit Residual Connections.
    
    Gibt zwei Feature-Tensoren zurück:
      feat_s: (N, hidden_dim)     → für STRIDE_S (kleine Objekte)
      feat_l: (N, hidden_dim*2)   → für STRIDE_L (große Objekte)
    
    Warum GATv2?
      Standard GAT hat statische Attention — GATv2 berechnet
      Attention dynamisch pro Node-Paar. Besser für unregelmäßige
      Event-Graphen wo Nachbarschaften stark variieren.
    """
    def __init__(self, hidden_dim: int, heads: int = 4):
        super().__init__()
        self.gat1 = GATv2Conv(hidden_dim, hidden_dim,
                              heads=heads, concat=False,
                              add_self_loops=True)
        self.norm1 = nn.LayerNorm(hidden_dim)

        self.gat2 = GATv2Conv(hidden_dim, hidden_dim,
                              heads=heads, concat=False,
                              add_self_loops=True)
        self.norm2 = nn.LayerNorm(hidden_dim)

        self.gat3 = GATv2Conv(hidden_dim, hidden_dim * 2,
                              heads=heads, concat=False,
                              add_self_loops=True)
        self.norm3 = nn.LayerNorm(hidden_dim * 2)

    def forward(self, x, edge_index):
        h = F.gelu(self.norm1(self.gat1(x, edge_index))) + x
        h = F.gelu(self.norm2(self.gat2(h, edge_index)))
        feat_s = h                                        # Scale S
        feat_l = F.gelu(self.norm3(
            self.gat3(h, edge_index)))                    # Scale L
        return feat_s, feat_l
# GATv2Conv wird verwendet, da die Attention dynamisch
# pro Knotenpaar berechnet wird.
#
# Dies ist besonders wichtig für Eventdaten, da lokale
# Nachbarschaften stark variieren können.
#
# Die Architektur orientiert sich konzeptionell an
# Event-basierten GNN Arbeiten wie AEGNN und DAGR.

# =============================================================
# SPARSE → DENSE
# =============================================================
def sparse_to_dense(node_feats, pos, batch,
                    stride, batch_size, feat_dim):
    """
    Projiziert Node Features auf 2D Grid via Mean Scatter.
    
    Mehrere Nodes in der gleichen Grid-Zelle
    → ihr Feature-Mittelwert landet in der Zelle.
    Leere Zellen bleiben 0 (kein Event dort = kein Signal).
    """
    gh = H // stride
    gw = W // stride

    gx = (pos[:, 0] / stride).long().clamp(0, gw - 1)
    gy = (pos[:, 1] / stride).long().clamp(0, gh - 1)

    flat_idx = batch * (gh * gw) + gy * gw + gx

    grid_flat = scatter(node_feats, flat_idx,
                        dim=0,
                        dim_size=batch_size * gh * gw,
                        reduce='max')

    return grid_flat.view(batch_size, gh, gw, feat_dim) \
                    .permute(0, 3, 1, 2).contiguous()

# Sparse-to-Dense Projektion:
#
# Eventdaten liegen als unregelmäßiger Graph vor.
# CNN Detection Heads benötigen jedoch reguläre 2D Grids.
#
# Daher werden Node Features über Scatter-Operationen
# auf ein dichtes Grid projiziert.
#
# Mehrere Nodes pro Zelle werden aggregiert.

# =============================================================
# DETECTION HEAD
# =============================================================
class DetectionHead(nn.Module):
    """
    Zwei Conv Schichten auf dem Feature Grid.
    Output: (B, num_anchors * (5 + num_classes), gh, gw)
    """
    NUM_ANCHORS = 3

    def __init__(self, in_channels: int, num_classes: int):
        super().__init__()
        out = self.NUM_ANCHORS * (5 + num_classes)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels * 2,
                      3, padding=1, bias=False),
            nn.BatchNorm2d(in_channels * 2),
            nn.GELU(),
            nn.Conv2d(in_channels * 2, out, 1, bias=True),
        )

    def forward(self, x):
        return self.conv(x)


# =============================================================
# DETECTION LOSS
# =============================================================
class DetectionLoss(nn.Module):
    """
    YOLO-style Loss: Objectness + Box Regression + Classification.
    
    Anchors empirisch für GEN1 304x240:
      ANCHORS_S (stride=8):  kleine Objekte, Fußgänger nah
      ANCHORS_L (stride=16): große Objekte, Autos
    """
    ANCHORS_S = torch.tensor([      # Stride 8 → kleine Objekte
        [25., 30.],   # kleines Auto / Fußgänger klein
        [45., 45.],   # mittleres Objekt quadratisch
        [35., 60.],   # Fußgänger hochkant
    ])
    ANCHORS_L = torch.tensor([      # Stride 16 → große Objekte
        [65., 65.],   # großes Objekt quadratisch
        [90., 70.],   # großes Auto
        [55., 90.],   # großer Fußgänger
    ])

    def __init__(self, num_classes=2,
                 lambda_box=1.0,
                 lambda_obj=0.5,
                 lambda_cls=1.0):
        super().__init__()
        self.nc         = num_classes
        self.lambda_box = lambda_box
        self.lambda_obj = lambda_obj
        self.lambda_cls = lambda_cls

    def forward(self, pred_s, pred_l, targets, batch_size):
        _, ds = self._scale_loss(
            pred_s, targets,
            self.ANCHORS_S.to(pred_s.device), STRIDE_S)
        _, dl = self._scale_loss(
            pred_l, targets,
            self.ANCHORS_L.to(pred_l.device), STRIDE_L)

        total = (self.lambda_box * (ds['box'] + dl['box'])
               + self.lambda_obj * (ds['obj'] + dl['obj'])
               + self.lambda_cls * (ds['cls'] + dl['cls']))

        return {
            "total_loss": total,
            "loss_box":   ds['box']  + dl['box'],
            "loss_obj":   ds['obj']  + dl['obj'],
            "loss_cls":   ds['cls']  + dl['cls'],
        }

    def _scale_loss(self, pred, targets, anchors, stride):
        na = anchors.shape[0]
        B, _, gh, gw = pred.shape

        # (B, na, gh, gw, 5+nc)
        p = pred.view(B, na, 5 + self.nc, gh, gw) \
                .permute(0, 1, 3, 4, 2).contiguous()

        obj_t = torch.zeros(B, na, gh, gw,          device=pred.device)
        box_t = torch.zeros(B, na, gh, gw, 4,       device=pred.device)
        cls_t = torch.zeros(B, na, gh, gw, self.nc, device=pred.device)
        mask  = torch.zeros(B, na, gh, gw,          dtype=torch.bool,
                            device=pred.device)

        for b, tgt in enumerate(targets):
            if tgt is None or tgt['boxes'].shape[0] == 0:
                continue
            boxes  = tgt['boxes'].to(pred.device)
            labels = tgt['labels'].to(pred.device)

            #for i in range(boxes.shape[0]):
                #bx, by, bw, bh = boxes[i]
            for i in range(boxes.shape[0]):
                bx, by, bw, bh = boxes[i]

                # center berechnen (nur debug!)
                cx = bx + bw / 2
                cy = by + bh / 2


                gx = int((cx / stride).clamp(0, gw - 1).item())
                gy = int((cy / stride).clamp(0, gh - 1).item())

                #gx = int((bx / stride).clamp(0, gw - 1).item())
                #gy = int((by / stride).clamp(0, gh - 1).item())
                a  = self._best_anchor(bw, bh, anchors)

                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        gx_i = gx + dx
                        gy_i = gy + dy

                        if 0 <= gx_i < gw and 0 <= gy_i < gh:
                            obj_t[b, a, gy_i, gx_i] = 1.

                            box_t[b, a, gy_i, gx_i, 0] = cx / stride - gx_i
                            box_t[b, a, gy_i, gx_i, 1] = cy / stride - gy_i
                            box_t[b, a, gy_i, gx_i, 2] = torch.log(
                                bw / anchors[a, 0] + 1e-8)
                            box_t[b, a, gy_i, gx_i, 3] = torch.log(
                                bh / anchors[a, 1] + 1e-8)

                            cls_t[b, a, gy_i, gx_i, labels[i].long()] = 1.
                            mask[b, a, gy_i, gx_i] = True


        # ── Losses ────────────────────────────────────────────────
        # Objectness: auf allen Zellen
        loss_obj = F.binary_cross_entropy_with_logits(
            p[..., 4], obj_t,
            pos_weight=torch.tensor([3.0], device=pred.device)) 

        # Box + Class: nur auf positiven Zellen (mask=True)
        # mask ist (B, na, gh, gw) → für p brauchen wir (N_pos, 5+nc)
        # Daher mask explizit auf die 5. Dim expandieren
        if mask.any():
            # p[mask] hat Shape (N_pos, 5+nc) ✓
            pred_box = p[mask][:, :4]           # (N_pos, 4)
            pred_cls = p[mask][:, 5:]           # (N_pos, nc)
            tgt_box  = box_t[mask]              # (N_pos, 4)
            tgt_cls  = cls_t[mask]              # (N_pos, nc)

            loss_box = F.smooth_l1_loss(pred_box, tgt_box)
            smooth = 0.1
            tgt_cls_smooth = tgt_cls * (1 - smooth) + smooth / self.nc
            loss_cls = F.binary_cross_entropy_with_logits(
                pred_cls, tgt_cls_smooth)
        else:
            loss_box = pred.new_zeros(1).squeeze()
            loss_cls = pred.new_zeros(1).squeeze()

        return loss_obj + loss_box + loss_cls, \
            {"obj": loss_obj, "box": loss_box, "cls": loss_cls}

    @staticmethod
    def _best_anchor(bw, bh, anchors):
        iou = (torch.min(anchors[:,0], bw) *
               torch.min(anchors[:,1], bh)) / \
              (anchors[:,0]*anchors[:,1] + bw*bh -
               torch.min(anchors[:,0], bw) *
               torch.min(anchors[:,1], bh) + 1e-8)
        return int(iou.argmax())

# Die Loss Funktion orientiert sich an YOLO-artigen
# Dense Detection Verfahren.
#
# Optimiert werden:
#   - Bounding Box Regression
#   - Objectness
#   - Klassifikation
#
# Die Anchorgrößen wurden empirisch an GEN1 angepasst.

# =============================================================
# HAUPT-MODELL
# =============================================================
class Gen1GNNDetector(nn.Module):
    """
    Vollständiges Modell. Interface:
      Training: model(data) → loss_dict
      Test:     model(data) → (detections, targets)
    """

    def __init__(self, hidden_dim=128, num_classes=NUM_CLASSES):
        super().__init__()
        self.hidden_dim  = hidden_dim
        self.num_classes = num_classes
        self.height      = H      # für DetectionBuffer
        self.width       = W

        self.encoder  = NodeEncoder(hidden_dim)
        self.backbone = GNNBackbone(hidden_dim, heads=4)
        self.head_s   = DetectionHead(hidden_dim,     num_classes)
        self.head_l   = DetectionHead(hidden_dim * 2, num_classes)
        self.loss_fn  = DetectionLoss(num_classes)

    def forward(self, data: Data, compute_loss: bool = True):
        # Batch Size bestimmen
        B = int(data.batch.max().item()) + 1 \
            if data.batch is not None \
            else 1

        batch = data.batch if data.batch is not None \
                else torch.zeros(data.x.shape[0],
                                 dtype=torch.long,
                                 device=data.x.device)

        # ── Forward ───────────────────────────────────────────
        node_h          = self.encoder(data)
        feat_s, feat_l  = self.backbone(node_h, data.edge_index)

        grid_s = sparse_to_dense(feat_s, data.pos, batch,
                                 STRIDE_S, B, self.hidden_dim)
        grid_l = sparse_to_dense(feat_l, data.pos, batch,
                                 STRIDE_L, B, self.hidden_dim * 2)

        pred_s = self.head_s(grid_s)
        pred_l = self.head_l(grid_l)

        targets = self._extract_targets(data, B)

        if compute_loss:
            loss_dict = self.loss_fn(pred_s, pred_l, targets, B)
            if not self.training:
                detections = self._decode(pred_s, pred_l)
                return loss_dict, detections
            return loss_dict
        else:
            return self._decode(pred_s, pred_l), targets

    def _extract_targets(self, data, B):
        # Bei Inference gibt es kein bbox → leere Targets zurückgeben
        if not hasattr(data, 'bbox') or data.bbox is None:
            return [{'boxes':  torch.zeros((0, 4)),
                    'labels': torch.zeros((0,), dtype=torch.long)}
                    for _ in range(B)]

        targets = []
        for b in range(B):
            if hasattr(data, 'bbox_batch'):
                mask = (data.bbox_batch == b)
            else:
                mask = torch.ones(data.bbox.shape[0],
                                dtype=torch.bool,
                                device=data.bbox.device)
            if mask.any() and data.bbox.shape[0] > 0:
                boxes  = data.bbox[mask, :4].float()
                labels = data.bbox[mask,  4].long()
                targets.append({'boxes': boxes, 'labels': labels})
            else:
                targets.append({'boxes':  torch.zeros((0, 4)),
                                'labels': torch.zeros((0,),
                                        dtype=torch.long)})
        return targets

    def _decode(self, pred_s, pred_l):
        dets = []
        B = pred_s.shape[0]
        for b in range(B):
            boxes_s, sc_s, lb_s = self._decode_scale(
                pred_s[b:b+1],
                DetectionLoss.ANCHORS_S.to(pred_s.device),
                STRIDE_S)
            boxes_l, sc_l, lb_l = self._decode_scale(
                pred_l[b:b+1],
                DetectionLoss.ANCHORS_L.to(pred_l.device),
                STRIDE_L)
            boxes  = torch.cat([boxes_s,  boxes_l],  dim=0)
            scores = torch.cat([sc_s,  sc_l],  dim=0)
            labels = torch.cat([lb_s,  lb_l],  dim=0)
            if boxes.shape[0] > 0:
                from torchvision.ops import nms
                x1 = boxes[:,0] - boxes[:,2]/2
                y1 = boxes[:,1] - boxes[:,3]/2
                x2 = boxes[:,0] + boxes[:,2]/2
                y2 = boxes[:,1] + boxes[:,3]/2
                keep = nms(torch.stack([x1,y1,x2,y2],1),
                           scores, 0.45)
                dets.append((boxes[keep], scores[keep],
                             labels[keep]))
            else:
                dets.append((boxes, scores, labels))
        return dets

    def _decode_scale(self, pred, anchors, stride):

        B, _, gh, gw = pred.shape
        na = anchors.shape[0]
        p  = pred.view(B, na, 5+self.num_classes, gh, gw) \
                  .permute(0,1,3,4,2).contiguous()[0]

        gy_grid, gx_grid = torch.meshgrid(
            torch.arange(gh, device=pred.device, dtype=torch.float32),
            torch.arange(gw, device=pred.device, dtype=torch.float32),
            indexing='ij')

        bx = (p[...,0].sigmoid() + gx_grid) * stride
        by = (p[...,1].sigmoid() + gy_grid) * stride
        bw = p[...,2].exp() * anchors[:,0].view(na,1,1)
        bh = p[...,3].exp() * anchors[:,1].view(na,1,1)
        sc = p[...,4].sigmoid()
        cp = p[...,5:].sigmoid()

        scores, labels = (sc.unsqueeze(-1) * cp).max(-1)
        m = scores > 0.1


        return (torch.stack([bx[m], by[m], bw[m], bh[m]], 1),
                scores[m], labels[m])