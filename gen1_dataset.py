"""
gen1_dataset.py
===============
PyTorch Geometric Dataset für GEN1.
Liest AEGNN preprocessed .pkl Dateien und gibt
PyG Data Objekte zurück.

Wichtig:
  bbox Format in pkl: (M, 8) = ts, x, y, w, h, class_id, conf, track_id
  Wir extrahieren: x, y, w, h, class_id → (M, 5)
  
  edge_index hat ~800k Edges → wird auf max_edges reduziert
  (zufälliges Subsampling der Edges für Speed)
"""
import os
import glob
import pickle
import numpy as np
import torch
from torch_geometric.data import Data, Dataset
#from dagr.utils.multiprocessing import TaskManager  # optional, nur für preprocessing


# PSEELoader — direkt aus AEGNN übernommen
EV_TYPE = [('t', 'u4'), ('_', 'i4')]

class PSEELoader:
    """Minimaler Loader für .dat Dateien (aus AEGNN)."""
    def __init__(self, datfile):
        import sys
        self._file = open(datfile, "rb")
        self._start, self._ev_size, self._size = self._parse_header()
        self._decode_dtype = [('t', 'u4'), ('x', 'u2'), ('y', 'u2'), ('p', 'u1')]
        self._file.seek(0, os.SEEK_END)
        self._end = self._file.tell()
        self._ev_count = (self._end - self._start) // self._ev_size
        self._file.seek(self._start)

# PSEELoader basiert auf der offiziellen AEGNN / Prophesee
# Event-Parsing Implementierung.
#
# Aufgabe:
#   Dekodierung der binären .dat Event Streams.
#
# Event Format:
#   t : Timestamp
#   x : Pixel x-Koordinate
#   y : Pixel y-Koordinate
#   p : Polarität
#
# Die Events werden direkt aus dem Prophesee Format gelesen.

    def _parse_header(self):
        import sys
        self._file.seek(0)
        bod = None
        size = [None, None]
        while True:
            bod = self._file.tell()
            line = self._file.readline()
            first = line.decode("latin-1")[:2]
            if first != '% ':
                break
            words = line.split()
            if len(words) > 1:
                if words[1] in [b'Height', b'height']:
                    size[0] = int(words[2])
                if words[1] in [b'Width', b'width']:
                    size[1] = int(words[2])
        self._file.seek(bod)
        ev_type = np.frombuffer(self._file.read(1), dtype=np.uint8)[0]
        ev_size = np.frombuffer(self._file.read(1), dtype=np.uint8)[0]
        return self._file.tell(), ev_size, size

    def seek_event(self, ev_count):
        self._file.seek(self._start + ev_count * self._ev_size)

    def load_n_events(self, ev_count):
        buf = np.empty(ev_count, dtype=[('t','u4'),('_','i4')])
        dat = np.fromfile(self._file, dtype=[('t','u4'),('_','i4')],
                          count=ev_count)
        out = np.empty(len(dat), dtype=self._decode_dtype)
        out['t'] = dat['t']
        out['x'] = np.bitwise_and(dat['_'], 16383)
        out['y'] = np.right_shift(np.bitwise_and(dat['_'], 268419072), 14)
        out['p'] = np.right_shift(np.bitwise_and(dat['_'], 268435456), 28)
        return out

    def __del__(self):
        if hasattr(self, '_file'):
            self._file.close()


class Gen1Dataset(Dataset):
    """
    PyG Dataset für GEN1 preprocessed pkl Dateien.
    
    Gibt Data Objekte zurück mit:
      data.x          (N, 1)   Polarität
      data.pos        (N, 3)   x, y, t_normalized
      data.edge_index (2, E)   Graph Kanten (reduziert)
      data.bbox       (M, 5)   x, y, w, h, class_id (float)
      data.num_nodes  int
    """

    # Sensor Dimensionen
    HEIGHT = 240
    WIDTH  = 304

    def __init__(self, root: str, split: str,
                 max_edges: int = 50000,
                 transform=None):
        """
        root:      Pfad zum gen1 Ordner
                   (enthält train/, val/, processed/)
        split:     "train" oder "val"
        max_edges: maximale Anzahl Edges pro Graph
                   (809k → 50k ist gut für Speed)
        """
        self.root_dir  = root
        self.split     = split
        self.max_edges = max_edges
        self._transform = transform

        # Alle pkl Dateien finden
        processed_dir = os.path.join(root, "processed", split)
        self.files = sorted(glob.glob(
            os.path.join(processed_dir, "*.pkl")))

        if len(self.files) == 0:
            raise RuntimeError(
                f"Keine pkl Dateien in {processed_dir}\n"
                f"Hast du AEGNN preprocessing ausgeführt?")
        
        print(f"[Gen1Dataset] Checking duplicates (loading pkl headers)...")
        seen_starts = set()
        unique = []
        for f in self.files:
            with open(f, 'rb') as fh:
                d = pickle.load(fh)
            start = int(d['raw'][0])
            if start not in seen_starts:
                seen_starts.add(start)
                unique.append(f)
        removed = len(self.files) - len(unique)
        self.files = unique
        print(f"[Gen1Dataset] {split}: {len(self.files)} unique "
              f"({removed} Duplikate entfernt)")


        # Für DetectionBuffer (DAGR kompatibel)
        self.height  = self.HEIGHT
        self.width   = self.WIDTH
        self.classes = ["car", "pedestrian"]

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        with open(self.files[idx], 'rb') as f:
            d = pickle.load(f)

        # Pfad zur .dat Datei dynamisch aus root_dir aufbauen
        # → funktioniert auf jedem PC unabhängig vom Original-Pfad
        filename  = os.path.basename(d['raw_file'])          # z.B. "17-03-30_..._td.dat"
        subfolder = os.path.basename(
                        os.path.dirname(d['raw_file']))       # z.B. "training"
        raw_file  = os.path.join(self.root_dir, subfolder, filename)

        # ── Events laden ──────────────────────────────────────
        loader = PSEELoader(raw_file)
        raw_start, raw_num_events = d['raw']
        loader.seek_event(int(raw_start))
        events = loader.load_n_events(int(raw_num_events))
        events = events[d['sample_idx']]

        # ── Node Features ─────────────────────────────────────
        x = torch.from_numpy(events['x'].astype(np.float32))
        y = torch.from_numpy(events['y'].astype(np.float32))
        t = torch.from_numpy(events['t'].astype(np.float32))
        p = torch.from_numpy(events['p'].astype(np.float32))

        # Zeit normalisieren [0, 1]
        t_min, t_max = t.min(), t.max()
        t_norm = (t - t_min) / (t_max - t_min + 1e-8)

        pos      = torch.stack([x, y, t_norm], dim=1)  # (N, 3)
        polarity = p.unsqueeze(1)                       # (N, 1)

        # ── Edge Index ────────────────────────────────────────
        edge_index = d['edge_index']  # (2, E) — kann 800k sein

        # Große Event-Graphen können mehrere hunderttausend
        # Kanten enthalten.
        #
        # Um Speicherverbrauch und Laufzeit zu reduzieren,
        # wird ein zufälliges Edge Subsampling durchgeführt.
        #
        # Das Sampling bleibt reproduzierbar, da der Sample-Index
        # als Random Seed verwendet wird.
        #
        # Idee inspiriert durch sparse Event-GNN Verarbeitung
        # aus AEGNN und DAGR.
        # Edge Subsampling: 800k → max_edges
        # Zufällig, aber reproduzierbar per Sample

        if edge_index.shape[1] > self.max_edges:
            rng = np.random.RandomState(idx)
            keep = rng.choice(edge_index.shape[1],
                              self.max_edges,
                              replace=False)
            keep = torch.from_numpy(keep)
            edge_index = edge_index[:, keep]

        # Bounding Boxes stammen aus dem GEN1 Dataset.
        #
        # Originalformat:
        #   (timestamp, x, y, w, h, class_id, confidence, track_id)
        #
        # Für das Training werden nur die für Detection relevanten
        # Komponenten extrahiert.

        # ── Bounding Boxes ────────────────────────────────────
        # pkl bbox: (M, 8) = ts, x, y, w, h, class_id, conf, track_id
        bbox_raw = d['bbox']  # torch.Tensor (M, 8)

        if bbox_raw.shape[0] > 0:
            bbox = torch.stack([
                bbox_raw[:, 1],   # x
                bbox_raw[:, 2],   # y
                bbox_raw[:, 3],   # w
                bbox_raw[:, 4],   # h
                bbox_raw[:, 5],   # class_id
            ], dim=1).float()

            # Negative Koordinaten und zu kleine Boxen rausfiltern
            valid = (bbox[:, 0] >= 0) & \
                    (bbox[:, 1] >= 0) & \
                    (bbox[:, 2] > 5) & \
                    (bbox[:, 3] > 5)
            bbox = bbox[valid]

            # Falls nach Filter nichts übrig
            if bbox.shape[0] == 0:
                bbox = torch.zeros((0, 5), dtype=torch.float32)
        else:
            bbox = torch.zeros((0, 5), dtype=torch.float32)

        # ── PyG Data Objekt ───────────────────────────────────
        data = Data(
            x          = polarity,       # (N, 1)
            pos        = pos,            # (N, 3)
            edge_index = edge_index,     # (2, E)
            bbox       = bbox,           # (M, 5)
            num_nodes  = polarity.shape[0],
        )

        if self._transform is not None:
            data = self._transform(data)

        return data

    def get(self, idx):
        return self.__getitem__(idx)

    def len(self):
        return len(self.files)