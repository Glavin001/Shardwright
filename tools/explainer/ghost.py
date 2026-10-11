"""Overlay a graph render on a faded render of the solid (same camera):
pixels where GRAPH differs from EMPTY (the same view with nothing drawn)
come from GRAPH, the rest is SOLID faded towards the background.

Usage: ghost.py SOLID.png GRAPH.png EMPTY.png OUT.png [FADE]"""
import sys
import numpy as np
from PIL import Image

solid, graph, empty = (np.asarray(Image.open(p).convert("RGB")).astype(float) for p in sys.argv[1:4])
fade = float(sys.argv[5]) if len(sys.argv) > 5 else 0.3
mask = (np.abs(graph - empty).sum(-1) > 12)[..., None]
out = np.where(mask, graph, empty + (solid - empty) * fade)
Image.fromarray(out.clip(0, 255).astype(np.uint8)).save(sys.argv[4])
