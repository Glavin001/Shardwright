"""Compose the "Why Shardwright?" explainer: scenes with captions, labels and
animated charts over the pre-rendered previews (render.py, render_zoom.py)
and the measured evidence (metrics.py). Writes PNG frames to out/why/frames
(12 fps) for make.sh."""
import json, math, os, shutil
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
R = ROOT + "/out/why/render"
OUT = ROOT + "/out/why/frames"
DATA = ROOT + "/out/why/data"
W, H, FPS = 1280, 720, 12
BG = (247, 247, 250); INK = (24, 24, 32); GREY = (96, 96, 108); LIGHT = (226, 226, 233)
RED = (214, 48, 64); GREEN = (22, 146, 78); BLUE = (37, 99, 235); ORANGE = (230, 120, 20)
PURPLE = (124, 58, 237); TEAL = (13, 148, 136); WHITE = (255, 255, 255); YELLOW = (250, 204, 21)
FD = "/usr/share/fonts/truetype/dejavu/"
_fc = {}
def F(size, bold=False, mono=False):
    k = (size, bold, mono)
    if k not in _fc:
        name = "DejaVuSansMono.ttf" if mono else ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        _fc[k] = ImageFont.truetype(FD + name, size)
    return _fc[k]

def clamp(x, a=0.0, b=1.0): return max(a, min(b, x))
def ease(t): t = clamp(t); return t * t * (3 - 2 * t)
def mix(c0, c1, a): return tuple(int(round(c0[i] + (c1[i] - c0[i]) * a)) for i in range(3))
def fade(c, a): return mix(BG, c, clamp(a))

ev = json.load(open(DATA + "/evidence.json"))
counts = json.load(open(DATA + "/partial_counts.json"))

_img = {}
def frame(name, i, box=None):
    """Pre-rendered frame `i` of sequence `name`, cropped to the panel (no
    renderer title/footer) and fitted into `box` = (w, h)."""
    d = f"{R}/{name}"
    files = sorted(os.listdir(d))
    i = max(0, min(i, len(files) - 1))
    key = (name, i, box)
    if key in _img: return _img[key]
    im = Image.open(f"{d}/{files[i]}").convert("RGB")
    if not name.endswith("_z"):
        w, h = im.size
        im = im.crop((0, 36, w, h - 44))
    if box:
        bw, bh = box
        w, h = im.size
        s = min(bw / w, bh / h)
        im = im.resize((int(w * s), int(h * s)), Image.LANCZOS)
        canvas = Image.new("RGB", box, BG)
        canvas.paste(im, ((bw - im.width) // 2, (bh - im.height) // 2))
        im = canvas
    if len(_img) > 400: _img.clear()
    _img[key] = im
    return im

def nframes(name): return len(os.listdir(f"{R}/{name}"))

# ---------------------------------------------------------------- drawing
def text(d, xy, s, size=24, bold=False, color=INK, a=1.0, anchor="la", mono=False):
    d.text(xy, s, font=F(size, bold, mono), fill=fade(color, a), anchor=anchor)

def wrap(s, size, maxw, bold=False):
    f = F(size, bold); words = s.split(); lines = []; cur = ""
    for w_ in words:
        t = (cur + " " + w_).strip()
        if f.getlength(t) <= maxw: cur = t
        else: lines.append(cur); cur = w_
    if cur: lines.append(cur)
    return lines

def para(d, xy, s, size, maxw, color=INK, a=1.0, bold=False, gap=1.3):
    x, y = xy
    for line in wrap(s, size, maxw, bold):
        text(d, (x, y), line, size, bold, color, a); y += int(size * gap)
    return y

def pill(d, xy, s, color, a=1.0, size=22, anchor="l"):
    f = F(size, True); tw = f.getlength(s); x, y = xy
    if anchor == "c": x -= (tw + 28) / 2
    d.rounded_rectangle([x, y, x + tw + 28, y + size + 16], radius=(size + 16) // 2, fill=fade(color, a))
    d.text((x + 14, y + 7), s, font=f, fill=mix(fade(color, a), WHITE, clamp(a)))

def header(d, chip, title, color, a=1.0, sub=None):
    f = F(20, True); tw = f.getlength(chip)
    d.rounded_rectangle([40, 30, 40 + tw + 24, 62], radius=8, fill=fade(color, a))
    d.text((52, 35), chip, font=f, fill=mix(fade(color, a), WHITE, clamp(a)))
    text(d, (40 + tw + 40, 26), title, 34, True, INK, a)
    if sub: text(d, (40 + tw + 42, 70), sub, 21, False, GREY, a)

def arrow(d, p0, p1, color, width=5, head=16, a=1.0):
    c = fade(color, a)
    d.line([p0, p1], fill=c, width=width)
    ang = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    for s in (-1, 1):
        q = (p1[0] - head * math.cos(ang + s * 0.45), p1[1] - head * math.sin(ang + s * 0.45))
        d.line([p1, q], fill=c, width=width)

def ring(d, c, r, color, width=5, a=1.0):
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], outline=fade(color, a), width=width)

def paste(img, im, xy, a=1.0, border=True):
    if a < 1.0:
        base = img.crop((xy[0], xy[1], xy[0] + im.width, xy[1] + im.height))
        im = Image.blend(base, im, clamp(a))
    img.paste(im, xy)
    if border:
        ImageDraw.Draw(img).rounded_rectangle([xy[0] - 1, xy[1] - 1, xy[0] + im.width, xy[1] + im.height], radius=6,
                                              outline=fade(LIGHT, a), width=2)

def bars(d, x, y, w, rows, t, vmax, row_h=56, label_w=300, fmt=None, size=22, bar_h=30):
    """Horizontal bars growing with t. rows = (label, value, color, value label)."""
    for k, (lab, v, col, vl) in enumerate(rows):
        tk = ease((t - 0.12 * k) / 0.6)
        yy = y + k * row_h
        text(d, (x, yy + bar_h // 2), lab, size, False, INK, min(1, t * 3), anchor="lm")
        bw = (w - label_w - 110) * (v / vmax) * tk
        d.rounded_rectangle([x + label_w, yy, x + label_w + max(bw, 4), yy + bar_h], radius=6, fill=fade(col, min(1, t * 3)))
        if tk > 0.05:
            text(d, (x + label_w + bw + 12, yy + bar_h // 2), vl, size, True, col, tk, anchor="lm")

def tick(d, c, r, color, a=1.0, cross=False):
    d.ellipse([c[0] - r, c[1] - r, c[0] + r, c[1] + r], fill=fade(color, a))
    w_ = max(3, r // 4); col = mix(fade(color, a), WHITE, clamp(a))
    if cross:
        o = r * 0.45
        d.line([(c[0] - o, c[1] - o), (c[0] + o, c[1] + o)], fill=col, width=w_)
        d.line([(c[0] - o, c[1] + o), (c[0] + o, c[1] - o)], fill=col, width=w_)
    else:
        d.line([(c[0] - r * 0.45, c[1] + r * 0.02), (c[0] - r * 0.1, c[1] + r * 0.38), (c[0] + r * 0.5, c[1] - r * 0.38)], fill=col, width=w_)

def new(): img = Image.new("RGB", (W, H), BG); return img, ImageDraw.Draw(img)

# ---------------------------------------------------------------- scenes
def s_title():
    T = 40
    for i in range(T):
        t = i / FPS; img, d = new()
        paste(img, frame("title", i, (540, 430)), (720, 140), a=ease(t / 0.8), border=False)
        a = ease(t / 0.6)
        text(d, (70, 210 + 12 * (1 - a)), "Why Shardwright?", 60, True, INK, a)
        a2 = ease((t - 0.5) / 0.6)
        text(d, (72, 300), "Destruction that breaks", 38, False, INK, a2)
        text(d, (72, 348), "like the real thing.", 38, False, INK, a2)
        a3 = ease((t - 1.1) / 0.6)
        para(d, (72, 430), "A pre-fracture library: fragments, bond graphs and colliders baked from your mesh, validated against physics.", 23, 560, GREY, a3)
        yield img

def s_problem():
    T = 100; n = nframes("problem")
    items = [("Cracks ignore the shape", "seeds land anywhere: not at notches, joints, supports"),
             ("Every material breaks the same", "glass, wood, brick and concrete look alike"),
             ("No idea what holds what", "no bonds: the solver guesses strength and support"),
             ("One flat level of pieces", "thousands of bodies at once, or no detail")]
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "THE PROBLEM", "Most pre-fractured destruction is random", RED, ease(t / 0.4))
        paste(img, frame("problem", min(i, n - 1), (560, 460)), (40, 135))
        pill(d, (60, 112), "Typical: random Voronoi shatter, no physics", RED, ease(t / 0.4), size=19)
        for k, (a_, b_) in enumerate(items):
            ak = ease((t - 0.7 - 1.0 * k) / 0.45)
            y = 150 + k * 108
            tick(d, (662, y + 18), 17, RED, ak, cross=True)
            text(d, (692, y + 2), a_, 27, True, INK, ak)
            text(d, (692, y + 40), b_, 20, False, GREY, ak)
        ab = ease((t - 5.2) / 0.5)
        if ab > 0:
            d.rectangle([0, 622, W, 700], fill=mix(BG, INK, ab))
            text(d, (40, 661), "TL;DR", 28, True, YELLOW, ab, anchor="lm")
            d.text((150, 661), "Pre-fractured destruction looks fake, and the physics can't tell what should break.",
                   font=F(25), fill=mix(mix(BG, INK, ab), WHITE, ab), anchor="lm")
        yield (img, "why_problem") if i == T - 1 else img

def s_pitch():
    T = 96
    cards = [(BLUE, "1", "Where it breaks", "Fracture modes find the cuts that need the least energy. Material recipes break glass, wood, brick and concrete their own way."),
             (GREEN, "2", "What holds it together", "An exact bond graph: contact area, normal, moments, rebar, joint type, Weibull strength, ground anchors."),
             (ORANGE, "3", "Ready for the runtime", "Four nested levels, non-overlapping convex colliders, watertight render meshes, glTF + FlatBuffers.")]
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "THE PITCH", "Bake a mesh into everything destruction needs", GREEN, ease(t / 0.4))
        for k, (col, num, ttl, body) in enumerate(cards):
            a = ease((t - 0.3 - 0.5 * k) / 0.5)
            x = 50 + k * 400; y = 140 + 40 * (1 - a)
            if a <= 0: continue
            d.rounded_rectangle([x, y, x + 370, y + 330], radius=16, fill=mix(BG, WHITE, a), outline=fade(LIGHT, a), width=2)
            d.rounded_rectangle([x, y, x + 370, y + 10], radius=5, fill=fade(col, a))
            d.ellipse([x + 24, y + 34, x + 72, y + 82], fill=fade(col, a))
            text(d, (x + 48, y + 58), num, 26, True, WHITE if a > 0.5 else BG, a, anchor="mm")
            text(d, (x + 86, y + 46), ttl, 23, True, INK, a)
            para(d, (x + 26, y + 112), body, 21, 320, GREY, a)
        a = ease((t - 2.8) / 0.5)
        text(d, (W // 2, 530), "All cut from one exact cell complex, so fragments, bonds and colliders always agree.", 25, True, INK, a, anchor="mm")
        a = ease((t - 3.6) / 0.5)
        text(d, (W // 2, 574), "Checked against FEM (Kratos), Voro++, upstream CoACD and the fracture-modes reference code.", 21, False, GREY, a, anchor="mm")
        a = ease((t - 4.6) / 0.5)
        text(d, (W // 2, 650), "Here is why each part matters  →", 24, True, BLUE, a, anchor="mm")
        yield (img, "beat_pitch") if i == T - 1 else img

# fracture-modes demo models: (name, layout, headline, detail, weakness label)
MODELS = [("bracket", "row", "A clear weakness: the bracket snaps at its neck.",
           "Random Voronoi cuts wherever its seeds land, never where the part is weak.", "clear weakness"),
          ("bone", "row", "A long bone breaks across its shaft.",
           "The most common real femur fracture. Random cuts chip the ends or land at random heights.", None),
          ("creature", "grid", "Slender arms snap off; the heavy body stays whole.",
           "Random Voronoi slices body and arms together at arbitrary points.", None),
          ("chair", "row", "A chair loses its legs and back at the joints.",
           "Random cuts saw the seat and legs at arbitrary heights.", None)]

def s_modes():
    for m, layout, line, sub, weak in MODELS:
        T = 54
        for i in range(T):
            t = i / FPS; img, d = new()
            header(d, "1 · FRACTURE MODES", "It breaks where the object is weak", BLUE, 1.0,
                   "Fracture modes (Sellán et al. 2023) find the cuts that need the least energy")
            if layout == "row":
                PW, PH, y0 = 232, 392, 136
                xs = [30 + k * 244 for k in range(5)]
                boxes = [(x, y0, PW, PH) for x in xs]
            else:
                boxes = [(30, 132, 300, 200), (345, 132, 300, 200), (30, 345, 300, 200), (345, 345, 300, 200), (670, 132, 580, 413)]
            names = [f"model_{m}_input", f"model_{m}_r1", f"model_{m}_r2", f"model_{m}_r3", f"model_{m}_sw"]
            appear = [0.0, 0.5, 0.8, 1.1, 1.6]
            for k, ((x, y, w_, h_), nm) in enumerate(zip(boxes, names)):
                a = ease((t - appear[k]) / 0.4)
                if a <= 0: continue
                j = int(max(0, (t - 1.8) * FPS)) if k == 4 else 0
                paste(img, frame(nm, j, (w_, h_)), (x, y), a=a)
            # labels
            if layout == "row":
                text(d, (xs[0] + PW // 2, y0 + PH + 22), "Input", 22, True, INK, ease(t / 0.4), anchor="mm")
                text(d, (xs[2] + PW // 2, y0 + PH + 22), "Random Voronoi fractures", 22, True, RED, ease((t - 0.5) / 0.4), anchor="mm")
                d.line([(xs[1] + 10, y0 + PH + 40), (xs[3] + PW - 10, y0 + PH + 40)], fill=fade(RED, ease((t - 0.5) / 0.4)), width=3)
                pill(d, (xs[4] + PW // 2, y0 + PH + 8), "✓ Shardwright", GREEN, ease((t - 1.6) / 0.4), size=21, anchor="c")
            else:
                pill(d, (40, 140), "Input", INK, ease(t / 0.4), size=18)
                for k, (x, y, _, _) in enumerate(boxes[1:4]):
                    pill(d, (x + 10, y + 8), f"Random Voronoi #{k + 1}", RED, ease((t - appear[k + 1]) / 0.4), size=17)
                pill(d, (680, 140), "✓ Shardwright: fracture modes", GREEN, ease((t - 1.6) / 0.4), size=19)
            if weak and t > 0.4:
                a = ease((t - 0.4) / 0.4)
                x, y = boxes[0][0], boxes[0][1]
                c = (x + PW // 2 + 2, y + int(PH * 0.57))
                ring(d, c, 18, BLUE, 4, a)
                text(d, (x + 6, y - 28), weak, 18, True, BLUE, a)
                arrow(d, (x + 40, y - 6), (c[0] - 14, c[1] - 14), BLUE, 3, 12, a)
            a = ease((t - 2.2) / 0.5)
            text(d, (W // 2, 604), line, 30, True, INK, a, anchor="mm")
            text(d, (W // 2, 648), sub, 21, False, GREY, ease((t - 2.6) / 0.5), anchor="mm")
            yield (img, f"beat_model_{m}") if i == T - 1 else img

MATS = [("glass_annealed_pane", "Radial glass recipe", "Annealed glass cracks radially from the impact.", "Small shards near the impact point, large ones far away."),
        ("glass_tempered_pane_z", "Tempered dicing", "Tempered glass is pre-stressed, so it dices into small cubes.", "That's why it's used where people can get hurt."),
        ("timber_beam_z", "Grain-aligned splinters", "Wood splinters along its grain.", "Long, thin splinters: splitting along the grain takes ~27× less energy than across it."),
        ("brick_wall_window_z", "Masonry courses", "Mortar is ~10× weaker than brick.", "So walls come apart along the joints, brick by brick.")]

def mini_glass(d, x, y, w, h, t):
    sw = ev["glass_annealed_median_size_m_by_distance"]["shardwright"]; un = ev["glass_annealed_median_size_m_by_distance"]["uniform"]
    text(d, (x, y), "median piece size vs distance from impact", 18, True, INK, ease(t / 0.4))
    x0, y0, x1, y1 = x + 50, y + 34, x + w - 130, y + h - 30
    d.line([(x0, y1), (x1, y1)], fill=fade(GREY, ease(t / 0.4)), width=2); d.line([(x0, y0), (x0, y1)], fill=fade(GREY, ease(t / 0.4)), width=2)
    text(d, (x0 - 8, y0), "25 cm", 14, False, GREY, ease(t / 0.4), anchor="rm"); text(d, (x0 - 8, y1), "0", 14, False, GREY, ease(t / 0.4), anchor="rm")
    text(d, (x0, y1 + 14), "impact", 14, False, GREY, ease(t / 0.4), anchor="lm"); text(d, (x1, y1 + 14), "far edge", 14, False, GREY, ease(t / 0.4), anchor="rm")
    for vals, col, nm in ((un, RED, "random"), (sw, GREEN, "Shardwright")):
        pts = [(x0 + (x1 - x0) * (k + 0.5) / len(vals), y1 - (y1 - y0) * v / 0.26) for k, v in enumerate(vals) if v is not None]
        m = max(2, int(len(pts) * ease((t - 0.3) / 1.0)))
        if t > 0.3: d.line(pts[:m], fill=col, width=4)
        if t > 1.3: text(d, (pts[-1][0] + 8, pts[-1][1] + (-12 if col == GREEN else 12)), nm, 16, True, col, ease((t - 1.3) / 0.4), anchor="lm")

def s_materials():
    PX, PY, PW, PH = 40, 130, 560, 344
    for k, (m, recipe, line, sub) in enumerate(MATS):
        T = 52
        base = m.replace("_z", "")
        un = f"{base}_un" + ("_z" if m.endswith("_z") else ""); sw = f"{base}_sw" + ("_z" if m.endswith("_z") else "")
        for i in range(T):
            t = i / FPS; img, d = new()
            header(d, "2 · MATERIALS", "Each material breaks its own way", PURPLE, 1.0)
            paste(img, frame(un, i, (PW, PH)), (PX, PY))
            paste(img, frame(sw, i, (PW, PH)), (PX + 640, PY))
            pill(d, (PX + 16, PY - 20), "✗  Random Voronoi", RED)
            pill(d, (PX + 656, PY - 20), f"✓  {recipe}", GREEN)
            a = ease((t - 0.6) / 0.5)
            yy = para(d, (PX, 500), line, 28, 600, INK, a, bold=True)
            para(d, (PX, yy + 4), sub, 21, 600, GREY, ease((t - 1.0) / 0.5))
            tc = t - 1.2
            cx, cy, cw = 680, 498, 560
            if k == 0:
                mini_glass(d, cx, cy, cw, 190, tc)
            elif k == 1:
                tm = ev["tempered"]
                text(d, (cx, cy), "pieces from the same pane", 18, True, INK, ease(tc / 0.4))
                bars(d, cx, cy + 34, 420, [("random", tm["uniform_pieces"], RED, f'{tm["uniform_pieces"]}  (median {tm["uniform_median_m"] * 100:.0f} cm)'),
                                         ("Shardwright", tm["shardwright_pieces"], GREEN, f'{tm["shardwright_pieces"]:,}  (median {tm["shardwright_median_m"] * 100:.0f} cm)')],
                     tc, 2600, row_h=48, label_w=140, size=19)
            elif k == 2:
                aa = ease((t - 1.0) / 0.4)
                if aa > 0:
                    c = (PX + 640 + 300, PY + 175)
                    ring(d, c, 26, BLUE, 4, aa)
                    arrow(d, (PX + 640 + 430, PY + 62), (c[0] + 20, c[1] - 20), BLUE, 4, 14, aa)
                    pill(d, (PX + 640 + 430, PY + 30), "long splinters", BLUE, aa, size=19, anchor="c")
                wd = ev["wood_elongation_along_grain"]
                text(d, (cx, cy), "splinter length along ÷ across the grain", 18, True, INK, ease(tc / 0.4))
                bars(d, cx, cy + 34, 520, [("random", wd["uniform"], RED, f'{wd["uniform"]:.1f}×'),
                                         ("Shardwright", wd["shardwright"], GREEN, f'{wd["shardwright"]:.1f}×')], tc, 11, row_h=48, label_w=140, size=19)
            else:
                bk = ev["brick_break_surface_on_mortar_joints"]
                text(d, (cx, cy), "share of break surface on mortar joints", 18, True, INK, ease(tc / 0.4))
                bars(d, cx, cy + 34, 520, [("random", bk["uniform"] * 100, RED, f'{int(bk["uniform"] * 100 + 0.5)}%'),
                                         ("Shardwright", bk["shardwright"] * 100, GREEN, f'{int(bk["shardwright"] * 100 + 0.5)}%')], tc, 110, row_h=48, label_w=140, size=19)
            yield (img, f"beat_material_{k}") if i == T - 1 else img

KINDS = [((0.70, 0.70, 0.72), "monolithic"), ((0.65, 0.25, 0.20), "mortar joint"), ((0.20, 0.45, 0.85), "cold joint"),
         ((0.20, 0.65, 0.70), "part connection"), ((0.95, 0.55, 0.10), "bearing"), ((0.05, 0.05, 0.05), "ground anchor")]

def s_bonds():
    n = nframes("bonds")
    fields = ["exact area, centroid and normal", "second moments: bending + torsion", "material mix and rebar crossings",
              "joint type: mortar, cold joint, bearing…", "Weibull strength (size effect)", "anchors to the ground"]
    T = 64
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "3 · BONDS", "It knows what holds it together", GREEN, 1.0, "Every contact between two pieces becomes a bond (two-storey building, 3,657 chunk bonds)")
        im = frame("bonds", min(i, n - 1))
        im = im.crop((0, 0, im.width, min(im.height, 560 - 36)))
        s = min(700 / im.width, 520 / im.height); im = im.resize((int(im.width * s), int(im.height * s)), Image.LANCZOS)
        paste(img, im, (30, 118), border=False)
        text(d, (760, 130), "Each bond carries", 26, True, INK, ease(t / 0.4))
        for k, f_ in enumerate(fields):
            a = ease((t - 0.4 - 0.45 * k) / 0.4)
            tick(d, (774, 190 + k * 46), 12, GREEN, a)
            text(d, (796, 190 + k * 46), f_, 21, False, INK, a, anchor="lm")
        a = ease((t - 3.4) / 0.5)
        for k, (c, nm) in enumerate(KINDS):
            x = 764 + (k % 2) * 230; y = 490 + (k // 2) * 34
            d.rectangle([x, y, x + 22, y + 22], fill=fade(tuple(int(v * 255) for v in c), a))
            text(d, (x + 32, y + 11), nm, 19, False, GREY, a, anchor="lm")
        yield img
    T = 60
    pt = ev["bond_patch_test_p95"]
    ex, cl, ad = [sum(pt[k]) / 2 * 100 for k in ("exact", "centre_line_normals", "adjacency_only")]
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "3 · BONDS", "Why the bonds must come from the same exact cells", GREEN, 1.0,
               "Stress error in a uniform-stress patch test: p95, same solver, same 1,537 pieces (lower is better)")
        bars(d, 60, 170, 1160, [("Exact bonds (Shardwright)", ex, GREEN, f"{ex:.1f}%"),
                                ("Normals from centre lines", cl, ORANGE, f"{cl:.1f}%"),
                                ("Adjacency only (equal areas)", ad, RED, f"{ad:.1f}%")], t, 20, row_h=86, label_w=360, size=24, bar_h=44)
        a = ease((t - 2.0) / 0.6)
        para(d, (60, 470), "A connectivity tool that only knows which pieces touch has to guess each contact's area and orientation. Those guesses put stresses off by about 10×, so the solver breaks the wrong bonds.", 24, 1160, INK, a)
        a = ease((t - 3.4) / 0.6)
        text(d, (60, 620), "Shardwright cuts bonds from the same exact cells as the fragments: child bonds even sum to their parent exactly.", 21, False, GREY, a)
        yield (img, "why_bonds_error") if i == T - 1 else img
    T = 56
    wk = Image.open(ROOT + "/docs/readme/weakening_two_storey.png").convert("RGB")
    wk = wk.crop((0, 0, wk.width, 640))
    s = 760 / wk.width; wk = wk.resize((int(wk.width * s), int(wk.height * s)), Image.LANCZOS)
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "3 · BONDS", "Predefined cuts are deliberately weaker", GREEN, 1.0, "Weibull weakest-link size effect plus a configurable bias")
        paste(img, wk, (30, 120), a=ease(t / 0.5))
        text(d, (820, 150), "Chunk bond capacity", 24, True, INK, ease(t / 0.4))
        text(d, (820, 182), "relative to the interior median", 19, False, GREY, ease(t / 0.4))
        bars(d, 820, 228, 440, [("interior", 1.0, (150, 150, 158), "×1.00"), ("L1 cuts", 0.37, ORANGE, "×0.37"), ("part joints", 0.13, RED, "×0.13")],
             t - 0.5, 1.15, row_h=52, label_w=120, size=19)
        a = ease((t - 2.2) / 0.5)
        para(d, (820, 420), "The stress solver breaks along the structural cuts and joints first, then splits chunks.", 22, 430, INK, a)
        yield img

def s_hierarchy():
    n = nframes("partial"); T = 108
    flat = ev["rigid_bodies_partial_break"]["flat_leaf"]
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "4 · HIERARCHY", "Detail only where it breaks", ORANGE, 1.0, "Four nested levels: a fragment is swapped for its children only where it breaks")
        j = min(i, n - 1)
        paste(img, frame("partial", j, (700, 520)), (30, 115), border=False)
        c = counts[j]; tot = sum(c)
        text(d, (770, 140), "Rigid bodies for this damage", 25, True, INK, ease(t / 0.4))
        bars(d, 770, 196, 490, [("flat shatter", flat, RED, f"{flat:,}"), ("Shardwright", tot, GREEN, f"{tot:,}")], 1.0 if t > 0.2 else t * 5, flat * 1.05,
             row_h=58, label_w=150, size=21, bar_h=34)
        text(d, (770, 330), "  +  ".join(f"{v} L{k}" for k, v in enumerate(c)), 19, False, GREY, 1.0)
        a = ease((t - 4.0) / 0.5)
        text(d, (770, 392), f"{flat / tot:.0f}× fewer bodies", 40, True, GREEN, a)
        para(d, (770, 452), "Far from the impact, whole parts stay one body. Near it, pieces get finer level by level, down to convex cells.", 21, 480, GREY, ease((t - 4.6) / 0.5))
        yield (img, "why_hierarchy") if i == T - 1 else img

def clip(P, x0, keep_left):
    """Sutherland-Hodgman clip of polygon P by the vertical line x = x0."""
    inside = (lambda p: p[0] <= x0) if keep_left else (lambda p: p[0] >= x0)
    out = []
    for k in range(len(P)):
        a, b = P[k - 1], P[k]
        ia, ib = inside(a), inside(b)
        if ia != ib:
            t = (x0 - a[0]) / (b[0] - a[0]); out.append((x0, a[1] + t * (b[1] - a[1])))
        if ib: out.append(b)
    return out

def poly(d, pts, fill=None, outline=None, width=3):
    d.polygon(pts, fill=fill, outline=outline, width=width)

def s_colliders():
    T = 88
    A = [(0, 0), (120, -10), (150, 30), (135, 70), (160, 110), (140, 160), (10, 170), (-10, 80)]
    B = [(150, 30), (260, 10), (290, 150), (140, 160), (160, 110), (135, 70)]
    def hull(pts):
        pts = sorted(set(pts))
        def cross(o, a, b): return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
        lo, up = [], []
        for p in pts:
            while len(lo) >= 2 and cross(lo[-2], lo[-1], p) <= 0: lo.pop()
            lo.append(p)
        for p in reversed(pts):
            while len(up) >= 2 and cross(up[-2], up[-1], p) <= 0: up.pop()
            up.append(p)
        return lo[:-1] + up[:-1]
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "5 · COLLIDERS", "Colliders that don't fight each other", TEAL, 1.0, "Convex hulls (≤ 64 vertices) that never overlap their neighbours")
        for col, (ox, title, good) in enumerate(((70, "Naive hulls", False), (400, "Shardwright hulls", True))):
            a = ease((t - 0.3 - 0.8 * col) / 0.5)
            oy = 220
            tr = lambda P, dx=0: [(ox + x * 1.0 + dx, oy + y * 1.0) for x, y in P]
            sep = 10 if good else 0
            poly(d, tr(A), fill=fade((150, 190, 240), a), outline=fade(BLUE, a))
            poly(d, tr(B, sep), fill=fade((170, 225, 180), a), outline=fade(GREEN, a))
            ha, hb = hull(A), hull(B)
            if good:
                # clip both hulls at the separating line (x = 145) with a margin
                ha = clip(ha, 142, keep_left=True); hb = clip(hb, 148, keep_left=False)
            d.line(tr(ha + ha[:1]), fill=fade(BLUE, a), width=3)
            d.line(tr(hb + hb[:1], sep), fill=fade(GREEN, a), width=3)
            if not good and a > 0:
                ov = [(135, 70), (150, 30), (160, 110), (140, 160), (135, 70)]
                poly(d, tr(ov), fill=fade((245, 160, 160), a), outline=fade(RED, a))
                arrow(d, (ox + 60, oy + 85), (ox - 15, oy + 85), RED, 4, 12, a)
                arrow(d, (ox + 230, oy + 85), (ox + 305, oy + 85), RED, 4, 12, a)
                text(d, (ox + 150, oy + 205), "overlap → pieces pop apart", 19, True, RED, a, anchor="mm")
                text(d, (ox + 150, oy + 232), "and jitter when they spawn", 19, False, RED, a, anchor="mm")
            if good and a > 0:
                d.line([(ox + 145 + sep / 2, oy - 30), (ox + 145 + sep / 2, oy + 190)], fill=fade(TEAL, a), width=2)
                text(d, (ox + 150, oy + 205), "separating plane + margin", 19, True, GREEN, a, anchor="mm")
                text(d, (ox + 150, oy + 232), "checked by a hard gate", 19, False, GREEN, a, anchor="mm")
            text(d, (ox + 150, oy - 60), title, 24, True, RED if not good else GREEN, a, anchor="mm")
        tc = t - 2.4
        text(d, (770, 160), "vs upstream CoACD (equal hull budget)", 22, True, INK, ease(tc / 0.4))
        cv = ev["coacd_equal_budget"]["outside_volume_median"]
        text(d, (770, 204), "hull volume outside the fragment (median)", 18, False, GREY, ease(tc / 0.4))
        bars(d, 770, 230, 480, [("CoACD", cv[1] * 100, RED, f"{cv[1] * 100:.0f}%"), ("Shardwright", cv[0] * 100, GREEN, f"{cv[0] * 100:.0f}%")],
             tc, 25, row_h=48, label_w=140, size=19)
        a = ease((tc - 1.0) / 0.4)
        text(d, (770, 352), "neighbouring hulls overlap", 18, False, GREY, a)
        tick(d, (790, 400), 14, RED, a, cross=True); text(d, (812, 400), "CoACD: yes", 20, True, RED, a, anchor="lm")
        tick(d, (990, 400), 14, GREEN, a); text(d, (1012, 400), "Shardwright: never", 20, True, GREEN, a, anchor="lm")
        para(d, (770, 460), "Built in pure Rust from a CoACD port, with a tighter fit at the same hull count.", 20, 470, GREY, ease((tc - 1.6) / 0.5))
        yield (img, "why_colliders") if i == T - 1 else img

def s_trust():
    T = 70
    tiles = [("14 hard gates", "run on every bake"), ("Voro++", "cell complexes match"),
             ("Fracture-modes reference", "exact on 3 of 4 meshes"), ("Kratos FEM", "column bond targets met"),
             ("Khronos glTF validator", "0 errors, 0 warnings"), ("Deterministic", "byte-identical, Linux + macOS")]
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "TRUST", "Checked against physics, not eyeballed", INK, 1.0)
        for k, (a_, b_) in enumerate(tiles):
            a = ease((t - 0.3 - 0.4 * k) / 0.4)
            x = 60 + (k % 3) * 400; y = 150 + (k // 3) * 200
            d.rounded_rectangle([x, y, x + 370, y + 160], radius=14, fill=mix(BG, WHITE, a), outline=fade(LIGHT, a), width=2)
            tick(d, (x + 50, y + 80), 26, GREEN, a)
            text(d, (x + 92, y + 62), a_, 21, True, INK, a, anchor="lm")
            text(d, (x + 92, y + 100), b_, 19, False, GREY, a, anchor="lm")
        a = ease((t - 3.2) / 0.5)
        text(d, (W // 2, 610), "Results, bounds and the targets not yet met are all in docs/VALIDATION.md.", 21, False, GREY, a, anchor="mm")
        yield img

def s_end():
    T = 50
    for i in range(T):
        t = i / FPS; img, d = new()
        a = ease(t / 0.6)
        text(d, (W // 2, 230), "Shardwright", 76, True, INK, a, anchor="mm")
        text(d, (W // 2, 318), "One bake → fragments · bonds · colliders · render meshes", 30, False, INK, ease((t - 0.4) / 0.6), anchor="mm")
        a = ease((t - 1.0) / 0.6)
        if a > 0:
            s = "prefracture bake --input building.glb --config support_clusters.toml --out out/"
            f = F(22, mono=True); tw = f.getlength(s)
            d.rounded_rectangle([W // 2 - tw / 2 - 24, 380, W // 2 + tw / 2 + 24, 432], radius=10, fill=mix(BG, INK, a))
            d.text((W // 2, 406), s, font=f, fill=mix(mix(BG, INK, a), (230, 230, 240), a), anchor="mm")
        text(d, (W // 2, 500), "github.com/Glavin001/Shardwright", 24, True, BLUE, ease((t - 1.6) / 0.6), anchor="mm")
        yield img

SCENES = [s_title, s_problem, s_pitch, s_modes, s_materials, s_bonds, s_hierarchy, s_colliders, s_trust, s_end]
XF = 6
if __name__ == "__main__":
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    os.makedirs(OUT)
    DOCS = ROOT + "/docs/readme"
    n = 0; prev = None; marks = []; stills = {}
    for sc in SCENES:
        marks.append((sc.__name__, n))
        for k, im in enumerate(sc()):
            if isinstance(im, tuple):
                im, name = im
                stills[name] = im.copy()
            if prev is not None and k < XF:
                im = Image.blend(prev, im, (k + 1) / (XF + 1))
            im.save(f"{OUT}/{n:05d}.png", compress_level=1); n += 1
            last = im
        prev = last
    json.dump(marks, open(DATA + "/marks.json", "w"))
    # README stills: single beats, the three shapes stacked, the four materials
    for name, im in stills.items():
        if name.startswith("why_"): im.save(f"{DOCS}/{name}.png", optimize=True)
    # fracture-modes figure: one row per model, paper style
    CW, CH, TOP = 250, 300, 80
    fig = Image.new("RGB", (5 * CW + 40, TOP + 4 * CH), BG)
    fd = ImageDraw.Draw(fig)
    cx = lambda k: 20 + k * CW + CW // 2
    fd.text((cx(0), 30), "Input", font=F(24, True), fill=INK, anchor="mm")
    fd.text((cx(2), 30), "Random Voronoi fractures", font=F(24, True), fill=RED, anchor="mm")
    fd.text((cx(2), 60), "same cells, random seeds", font=F(17), fill=GREY, anchor="mm")
    fd.text((cx(4), 30), "Shardwright", font=F(24, True), fill=GREEN, anchor="mm")
    fd.text((cx(4), 60), "fracture modes", font=F(17), fill=GREY, anchor="mm")
    for k in (1, 4):
        fd.line([(20 + k * CW - 2, 14), (20 + k * CW - 2, fig.height - 14)], fill=LIGHT, width=2)
    for r, (m, *_r) in enumerate(MODELS):
        for k, nm in enumerate([f"model_{m}_input", f"model_{m}_r1", f"model_{m}_r2", f"model_{m}_r3", f"model_{m}_sw"]):
            im = frame(nm, nframes(nm) - 1, (CW - 10, CH - 10))
            fig.paste(im, (25 + k * CW, TOP + r * CH))
    fig.save(f"{DOCS}/why_modes_models.png", optimize=True)
    grid = Image.new("RGB", (W, H), BG)
    for k in range(4): grid.paste(stills[f"beat_material_{k}"].resize((W // 2, H // 2), Image.LANCZOS), ((k % 2) * W // 2, (k // 2) * H // 2))
    grid.save(f"{DOCS}/why_materials.png", optimize=True)
    print(n, "frames,", n / FPS, "s", marks)
