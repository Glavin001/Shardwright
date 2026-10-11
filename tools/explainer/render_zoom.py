"""Zoomed material sequences: render larger, keep the centre (folder suffix _z)."""
import os, subprocess, math
from concurrent.futures import ThreadPoolExecutor
from PIL import Image
B = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..")); BIN = B + "/target/release/prefracture"
D = B + "/out/why/render"
R, U = B + "/out/why/shardwright", B + "/out/why/uniform"
W, H = 560, 380
jobs = []
for m, az, el, z in (("glass_tempered_pane", 30, 18, 2.4), ("timber_beam", 22, 32, 2.6), ("brick_wall_window", 35, 22, 1.7)):
    for v, root in (("sw", R), ("un", U)):
        d = f"{D}/{m}_{v}_z"; os.makedirs(d, exist_ok=True)
        for f in os.listdir(d): os.remove(f"{d}/{f}")
        for i in range(12):
            t = i / 11; t = 0.5 - 0.5 * math.cos(math.pi * t)
            jobs.append(([BIN, "preview", "--input", f"{root}/{m}.asset.json", "--out", f"{d}/{i:03d}.png", "--level", "3",
                          "--width", str(int(W * z)), "--height", str(int(H * z)), "--explode", f"{0.05 * t:.4f}",
                          "--azimuth", f"{az + 8 * t:.3f}", "--elevation", str(el)], f"{d}/{i:03d}.png"))
def run(j):
    a, p = j
    subprocess.run(a, check=True, capture_output=True)
    im = Image.open(p).convert("RGB"); w, h = im.size
    # panel area excludes the footer (44 px); centre crop of the panel
    ph = h - 44; cx, cy = w // 2, ph // 2
    im.crop((cx - W // 2, cy - H // 2, cx + W // 2, cy + H // 2)).save(p)
with ThreadPoolExecutor(4) as ex: list(ex.map(run, jobs))
print(len(jobs))
