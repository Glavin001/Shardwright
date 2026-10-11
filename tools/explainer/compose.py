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
OVERFLOW = []   # (scene, text) of anything drawn past the canvas or its box
CUR = {"scene": ""}
def check(d, xy, s, font, anchor="la", maxw=None):
    x0, y0, x1, y1 = d.textbbox(xy, s, font=font, anchor=anchor)
    if x0 < 6 or x1 > W - 6 or y1 > H - 2 or (maxw is not None and x1 - x0 > maxw):
        OVERFLOW.append((CUR["scene"], s[:60]))

def text(d, xy, s, size=24, bold=False, color=INK, a=1.0, anchor="la", mono=False, maxw=None):
    f = F(size, bold, mono)
    if a > 0.05: check(d, xy, s, f, anchor, maxw)
    d.text(xy, s, font=f, fill=fade(color, a), anchor=anchor)

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
    if a > 0.05 and (x < 6 or x + tw + 28 > W - 6): OVERFLOW.append((CUR["scene"], "pill: " + s))
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
    T = 44
    for i in range(T):
        t = i / FPS; img, d = new()
        paste(img, frame("title", i, (540, 430)), (720, 140), a=ease(t / 0.8), border=False)
        a = ease(t / 0.6)
        text(d, (70, 200 + 12 * (1 - a)), "Why Shardwright?", 60, True, INK, a)
        a2 = ease((t - 0.5) / 0.6)
        text(d, (72, 290), "Destruction that breaks", 38, False, INK, a2)
        text(d, (72, 338), "like the real thing.", 38, False, INK, a2)
        a3 = ease((t - 1.1) / 0.6)
        para(d, (72, 420), "Author a structure and its materials. Get fragments, colliders and bonds for your stress solver, validated against physics.", 23, 580, GREY, a3)
        a4 = ease((t - 1.8) / 0.6)
        text(d, (72, 540), "For games, simulation and film.", 24, True, BLUE, a4)
        yield img

def s_problem():
    T = 116
    span = ev.get("problem_spanning", [80, 129])
    cells = [("model_bracket_r1", "Cracks ignore the shape", "Random seeds miss the bracket's obvious weak neck."),
             ("materials", "Every material breaks the same", "Glass, wood and brick all become the same blobs."),
             ("problem_random", "Nothing knows what holds what", f"{span[0]} of {span[1]} random pieces (red) fuse columns, slabs and walls, and there are no bonds."),
             ("problem_leaf", "One flat level of pieces", f"Every piece is a body from the start: {ev['rigid_bodies_partial_break']['flat_leaf']:,} for this building.")]
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "THE PROBLEM", "Random pre-fracture looks fake and breaks wrong", RED, ease(t / 0.4))
        for k, (nm, ttl, sub) in enumerate(cells):
            a = ease((t - 0.4 - 1.1 * k) / 0.45)
            if a <= 0: continue
            x = 30 + (k % 2) * 620; y = 100 + (k // 2) * 258
            bw, bh = 300, 238
            if nm == "materials":
                im = Image.new("RGB", (bw, bh), BG)
                dd = ImageDraw.Draw(im)
                for j, (sq, lab) in enumerate((("glass_tempered_pane_un_z", "glass"), ("timber_beam_un_z", "wood"), ("brick_wall_window_un_z", "brick"))):
                    tile = frame(sq, 11)
                    cx, cy = int(tile.width * (0.3 if lab == "brick" else 0.5)), tile.height // 2
                    im.paste(tile.crop((cx - 48, cy - 105, cx + 48, cy + 105)), (j * 102, 0))
                    dd.text((j * 102 + 48, 224), lab, font=F(17, True), fill=INK, anchor="mm")
            else:
                im = frame(nm, 0, (bw, bh))
            paste(img, im, (x, y), a=a)
            if nm == "model_bracket_r1":
                ring(d, (x + bw // 2, y + int(bh * 0.57)), 14, BLUE, 3, a)
            tick(d, (x + bw + 30, y + 30), 15, RED, a, cross=True)
            yy = para(d, (x + bw + 54, y + 14), ttl, 23, 252, INK, a, bold=True)
            para(d, (x + bw + 54, yy + 6), sub, 18, 252, GREY, a)
        ab = ease((t - 5.4) / 0.5)
        if ab > 0:
            d.rectangle([0, 630, W, 706], fill=mix(BG, INK, ab))
            text(d, (40, 668), "TL;DR", 28, True, YELLOW, ab, anchor="lm")
            d.text((150, 668), "Random pre-fracture looks fake, and the physics can't tell what should break.",
                   font=F(25), fill=mix(mix(BG, INK, ab), WHITE, ab), anchor="lm")
        yield (img, "why_problem") if i == T - 1 else img

def card(d, x, y, w, h, color, title, items, t0, t, mark=True):
    a = ease((t - t0) / 0.5)
    if a <= 0: return
    d.rounded_rectangle([x, y, x + w, y + h], radius=16, fill=mix(BG, WHITE, a), outline=fade(LIGHT, a), width=2)
    d.rounded_rectangle([x, y, x + w, y + 10], radius=5, fill=fade(color, a))
    text(d, (x + 24, y + 30), title, 27, True, INK, a)
    yy = y + 84
    for k, it in enumerate(items):
        ak = ease((t - t0 - 0.3 - 0.25 * k) / 0.4)
        if mark: tick(d, (x + 34, yy + 12), 11, color, ak)
        yy = para(d, (x + 56, yy), it, 20, w - 80, INK, ak) + 10

def s_pitch():
    T = 100
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "THE PITCH", "You author the structure. Shardwright does the rest.", GREEN, ease(t / 0.4))
        card(d, 40, 120, 440, 400, BLUE, "You author", ["Meshes of the parts (glTF, OBJ, PLY, STL)",
             "A material per part: concrete, brick, glass, wood, steel…",
             "Optional: joints, ground anchors, rebar, wood grain, impact points"], 0.3, t)
        a = ease((t - 1.4) / 0.4)
        arrow(d, (500, 320), (790, 320), INK, 5, 18, a)
        f = F(19, mono=True); s_ = "prefracture bake"; tw = f.getlength(s_)
        if a > 0:
            d.rounded_rectangle([645 - tw / 2 - 14, 268, 645 + tw / 2 + 14, 302], radius=8, fill=mix(BG, INK, a))
            d.text((645, 285), s_, font=f, fill=mix(mix(BG, INK, a), (230, 230, 240), a), anchor="mm")
        text(d, (645, 350), "seconds to minutes", 18, False, GREY, a, anchor="mm")
        card(d, 810, 120, 430, 400, GREEN, "You get", ["Fragments at 4 nested levels",
             "Convex colliders that never overlap",
             "A bond graph with strengths, weaker along joints and cuts",
             "Render meshes with interiors and LODs",
             "A report: 14 hard gates passed"], 1.8, t)
        a = ease((t - 3.6) / 0.5)
        text(d, (W // 2, 580), "Ready for game engines, stress solvers, simulation and film VFX.", 25, True, INK, a, anchor="mm")
        a = ease((t - 4.4) / 0.5)
        text(d, (W // 2, 634), "Here is each capability, and why it matters  →", 23, True, BLUE, a, anchor="mm")
        yield (img, "beat_pitch") if i == T - 1 else img

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
        T = 50
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
        T = 48
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
    yield from s_bond_schematic()
    n = nframes("bonds")
    fields = ["exact area, centroid and normal", "second moments: bending + torsion", "material mix and rebar crossings",
              "joint type: mortar, cold joint, bearing…", "Weibull strength (size effect)", "anchors to the ground"]
    T = 64
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "5 · BONDS", "It knows what holds it together", GREEN, 1.0, "Every contact between two pieces becomes a bond (two-storey building, 3,657 chunk bonds)")
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
        header(d, "5 · BONDS", "Why the bonds must come from the same exact cells", GREEN, 1.0,
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
        header(d, "5 · BONDS", "Predefined cuts are deliberately weaker", GREEN, 1.0, "Weibull weakest-link size effect plus a configurable bias")
        paste(img, wk, (30, 120), a=ease(t / 0.5))
        text(d, (820, 150), "Chunk bond capacity", 24, True, INK, ease(t / 0.4))
        text(d, (820, 182), "relative to the interior median", 19, False, GREY, ease(t / 0.4))
        bars(d, 820, 228, 440, [("interior", 1.0, (150, 150, 158), "×1.00"), ("L1 cuts", 0.37, ORANGE, "×0.37"), ("part joints", 0.13, RED, "×0.13")],
             t - 0.5, 1.15, row_h=52, label_w=120, size=19)
        a = ease((t - 2.2) / 0.5)
        para(d, (820, 420), "The stress solver breaks along the structural cuts and joints first, then splits chunks.", 22, 430, INK, a)
        yield img

def s_hierarchy():
    """The Rapier destruction sim (tools/sim/make.sh) at half speed, with the
    number of rigid bodies it needs against a flat shatter."""
    sim = json.load(open(ROOT + "/out/sim/collapse.json"))["frames"]
    simdir = ROOT + "/out/sim/render"
    files = sorted(f for f in os.listdir(simdir) if f.endswith(".png"))
    flat = ev["rigid_bodies_partial_break"]["flat_leaf"]
    step = 60 / FPS / 2                     # sim frames per explainer frame at 1/2 speed
    T = int(len(files) / step) + 14
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "6 · HIERARCHY", "Detail only where it breaks", ORANGE, 1.0,
               "A Rapier simulation on the baked chunks, bonds and hulls (tools/sim), at half speed")
        j = min(int(i * step), len(files) - 1)
        im = Image.open(f"{simdir}/{files[j]}").convert("RGB")
        im = im.resize((800, 500), Image.LANCZOS)
        paste(img, im, (20, 112), border=False)
        fr = sim[j]
        text(d, (850, 130), "Rigid bodies right now", 25, True, INK, ease(t / 0.4))
        bars(d, 850, 186, 410, [("flat shatter", flat, RED, f"{flat:,}"), ("Shardwright", fr["islands"], GREEN, f"{fr['islands']:,}")],
             1.0 if t > 0.2 else t * 5, flat * 1.05, row_h=58, label_w=150, size=21, bar_h=34)
        text(d, (850, 320), f"t = {fr['t']:.2f} s", 22, True, INK, 1.0)
        text(d, (850, 352), f"bonds broken  {fr['broken']:,}", 20, False, GREY, 1.0)
        text(d, (850, 380), f"chunks refined to Level 3  {fr['refined']:,}", 20, False, GREY, 1.0)
        for k, (col, lab) in enumerate((((204, 201, 194), "Level-2 chunk"), ((245, 143, 51), "Level-3 cell, refined on impact"))):
            d.rectangle([850, 424 + 32 * k, 872, 446 + 32 * k], fill=col, outline=INK)
            text(d, (884, 435 + 32 * k), lab, 19, False, INK, 1.0, anchor="lm")
        para(d, (850, 512), "The ball breaks a column; the floor above loses its support and comes down. Only the chunks that are hit get finer.",
             20, 410, GREY, ease((t - 1.0) / 0.5))
        yield (img, "why_hierarchy") if i == int(len(files) * 0.55 / step) else img

def s_support():
    """What the bonds tell a solver: the support graph, the self-weight load
    path, and what a missing column overloads (tools/explainer/support.sh)."""
    st0 = json.load(open(ROOT + "/out/why/graph/l0_stats.json"))
    st2 = json.load(open(ROOT + "/out/why/graph/l2_stats.json"))
    ims = {k: Image.open(ROOT + f"/out/why/render/support_{k}/" + sorted(os.listdir(ROOT + f"/out/why/render/support_{k}"))[0]).convert("RGB")
           for k in ("graph", "load", "removed")}
    big = {k: v.resize((700, int(v.height * 700 / v.width)), Image.LANCZOS) for k, v in ims.items()}
    beats = [
        ("graph", "A support graph you can check",
         [f"{st0['nodes']} parts are the nodes, {st0['bonds']} bonds the edges (Level 0; every level has its own)",
          "Each edge runs through the contact it stands for",
          "Every piece reaches the ground through some path: nothing floats"],
         [((242, 140, 31), "joint between parts"), ((140, 77, 191), "bearing (compression only)"), ((20, 20, 26), "anchor to the ground")]),
        ("load", "Where the weight goes",
         [f"An exact stress solve of all {st2['bonds']:,} chunk bonds under self-weight",
          "Colour = the force through each chunk",
          "The ground-floor columns carry the building; the slabs carry little"],
         [((230, 232, 237), "little force"), ((140, 173, 230), "more"), ((31, 41, 140), "most")]),
        ("removed", "Take one column away",
         ["The column the wrecking ball hits, removed (pink)",
          f"{st2['over_capacity_bonds']} bonds go over capacity (red chunks)",
          "Those break first: this is where the collapse starts"],
         [((219, 26, 31), "over capacity"), ((247, 199, 199), "removed column"), ((140, 173, 230), "within capacity")]),
    ]
    T = 50
    for key, title, lines, legend in beats:
        for i in range(T):
            t = i / FPS; img, d = new()
            header(d, "5 · BONDS", "What the bonds tell your solver", GREEN, 1.0, "Two-storey building: connectivity, load path and failure, from the baked bonds alone")
            paste(img, big[key], (20, 120), a=ease(t / 0.4), border=False)
            text(d, (760, 140), title, 28, True, INK, ease(t / 0.4))
            y = 196
            for k, ln in enumerate(lines):
                a = ease((t - 0.5 - 0.5 * k) / 0.4)
                tick(d, (774, y + 13), 11, GREEN, a)
                y = para(d, (796, y), ln, 21, 450, INK, a) + 14
            a = ease((t - 2.2) / 0.4)
            for k, (col, lab) in enumerate(legend):
                d.rectangle([764, 520 + 34 * k, 786, 542 + 34 * k], fill=fade(col, a), outline=fade(INK, a))
                text(d, (798, 531 + 34 * k), lab, 19, False, GREY, a, anchor="lm")
            yield img
    # the three side by side (README still)
    T = 40
    small = {k: v.resize((404, int(v.height * 404 / v.width)), Image.LANCZOS) for k, v in ims.items()}
    caps = [("1. Support graph", "every part connected to the ground"), ("2. Load path", "self-weight flows down the columns"),
            ("3. One column removed", f"{st2['over_capacity_bonds']} bonds over capacity break first")]
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "5 · BONDS", "What the bonds tell your solver", GREEN, 1.0, "Two-storey building: connectivity, load path and failure, from the baked bonds alone")
        for k, key in enumerate(("graph", "load", "removed")):
            x = 20 + k * 418
            paste(img, small[key], (x, 140), a=ease((t - 0.2 * k) / 0.4), border=False)
            text(d, (x + 202, 140 + small[key].height + 26), caps[k][0], 24, True, INK, ease((t - 0.2 * k) / 0.4), anchor="mm")
            text(d, (x + 202, 140 + small[key].height + 58), caps[k][1], 19, False, GREY, ease((t - 0.2 * k) / 0.4), anchor="mm")
            a = ease((t - 0.6 - 0.2 * k) / 0.4)
            for r, (col, lab) in enumerate(beats[k][3]):
                yy = 140 + small[key].height + 96 + 30 * r
                d.rectangle([x + 110, yy, x + 128, yy + 18], fill=fade(col, a), outline=fade(INK, a))
                text(d, (x + 140, yy + 9), lab, 17, False, GREY, a, anchor="lm")
        yield (img, "why_support") if i == T - 1 else img

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
        header(d, "4 · COLLIDERS", "Colliders that don't fight each other", TEAL, 1.0, "Convex hulls (≤ 64 vertices) that never overlap their neighbours")
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


INPUT_BEATS = [
    ("material", [("in_pane_annealed", '"material": "glass_annealed"'), ("in_pane_tempered", '"material": "glass_tempered"'),
                  ("in_pane_concrete", '"material": "concrete_c30"')],
     "Same pane, one line changed: the material picks the fracture pattern."),
    ("impact", [("in_pane_annealed", '"impact_center": [0.45, 0.65, 0]'), ("in_pane_corner", '"impact_center": [0.2, 0.25, 0]')],
     "Move the impact point and the radial cracks move with it."),
    ("grain", [("in_beam_x_z", '"grain": [1, 0, 0]'), ("in_beam_y_z", '"grain": [0, 1, 0]')],
     "Rotate the wood grain and the splinters follow it."),
]
PROPS = [("recipe, grain, impact point", "the cell pattern: how it shatters"),
         ("Young's modulus, Poisson's ratio", "fracture modes (FEM) and bond stiffness"),
         ("fracture energy (grain, joints)", "where cuts are cheap, so where it splits"),
         ("tensile and compressive strength, cohesion", "bond capacity for the stress solver"),
         ("Weibull modulus", "size-effect weakening of large cut surfaces"),
         ("density", "mass and inertia of every piece"),
         ("noise, chipping, interior colours", "the render meshes")]

def s_inputs():
    for key, panels, line in INPUT_BEATS:
        T = 46
        n = len(panels); pw = 380 if n == 3 else 540; gap = (W - 60 - n * pw) // max(1, n - 1)
        for i in range(T):
            t = i / FPS; img, d = new()
            header(d, "3 · YOUR INPUTS", "Your inputs steer the result", ORANGE, 1.0,
                   "Edit one line of the part's metadata and re-bake")
            for k, (nm, code) in enumerate(panels):
                x = 30 + k * (pw + gap)
                a = ease((t - 0.2 - 0.5 * k) / 0.4)
                if a <= 0: continue
                paste(img, frame(nm, 0, (pw, 380)), (x, 160), a=a)
                f = F(17, mono=True); tw = f.getlength(code)
                d.rounded_rectangle([x, 116, x + tw + 24, 146], radius=8, fill=mix(BG, INK, a))
                d.text((x + 12, 131), code, font=f, fill=mix(mix(BG, INK, a), (240, 220, 140), a), anchor="lm")
            a = ease((t - 1.4) / 0.5)
            text(d, (W // 2, 600), line, 28, True, INK, a, anchor="mm")
            yield (img, f"beat_input_{key}") if i == T - 1 else img
    T = 66
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "3 · YOUR INPUTS", "Material properties, and what each one drives", ORANGE, 1.0,
               "Built-in library in materials.toml, or pass your own with --materials")
        text(d, (60, 128), "Property", 21, True, GREY, ease(t / 0.4))
        text(d, (640, 128), "Drives", 21, True, GREY, ease(t / 0.4))
        for k, (p_, e_) in enumerate(PROPS):
            a = ease((t - 0.3 - 0.35 * k) / 0.4)
            y = 170 + k * 62
            d.line([(60, y - 12), (1220, y - 12)], fill=fade(LIGHT, a), width=1)
            text(d, (60, y + 8), p_, 22, True, INK, a, anchor="lm", maxw=560)
            text(d, (640, y + 8), e_, 22, False, INK, a, anchor="lm", maxw=580)
        yield img

def s_convex():
    T = 60
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "4 · COLLIDERS", "Physics engines need convex shapes", TEAL, 1.0,
               "Fracture-mode pieces follow the shape, so they are non-convex")
        paste(img, frame("hull_pieces", 0, (560, 400)), (30, 140))
        pill(d, (46, 122), "Fracture-mode pieces (non-convex)", BLUE, 1.0, size=19)
        a = ease((t - 1.0) / 0.5)
        arrow(d, (605, 340), (675, 340), INK, 5, 16, a)
        paste(img, frame("hull_hulls", 0, (560, 400)), (690, 140), a=a)
        pill(d, (706, 122), "Convex hulls, ≤ 64 vertices each", GREEN, a, size=19)
        a = ease((t - 2.0) / 0.5)
        text(d, (W // 2, 590), "Each piece is decomposed into convex hulls: 9 pieces → 105 hulls.", 27, True, INK, a, anchor="mm")
        text(d, (W // 2, 632), "Convex cells skip this step; hulls of neighbouring pieces never overlap.", 21, False, GREY, ease((t - 2.6) / 0.5), anchor="mm")
        yield (img, "why_convex") if i == T - 1 else img
    yield from s_colliders()

def s_bond_schematic():
    T = 56
    A = [(0, 0), (140, 0), (200, 230), (0, 230)]          # left piece; shared face (140,0)-(200,230)
    B_ = [(140, 0), (330, 0), (330, 230), (200, 230)]
    def cen(P): return (sum(p[0] for p in P) / len(P), sum(p[1] for p in P) / len(P))
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "5 · BONDS", "Bonds come from the exact contact surfaces", GREEN, 1.0,
               "A stress solver decides what breaks from each bond's area, orientation and strength")
        for col, (ox, good, ttl) in enumerate(((90, False, "Guessed from adjacency"), (720, True, "Shardwright: exact contact"))):
            a = ease((t - 0.3 - 1.0 * col) / 0.5)
            if a <= 0: continue
            oy = 200
            tr = lambda P: [(ox + x, oy + y) for x, y in P]
            d.polygon(tr(A), fill=fade((190, 210, 240), a), outline=fade(BLUE, a), width=3)
            d.polygon(tr(B_), fill=fade((200, 230, 205), a), outline=fade(GREEN, a), width=3)
            ca, cb = cen(A), cen(B_)
            d.ellipse([ox + ca[0] - 5, oy + ca[1] - 5, ox + ca[0] + 5, oy + ca[1] + 5], fill=fade(INK, a))
            d.ellipse([ox + cb[0] - 5, oy + cb[1] - 5, ox + cb[0] + 5, oy + cb[1] + 5], fill=fade(INK, a))
            text(d, (ox + 165, oy - 52), ttl, 24, True, RED if not good else GREEN, a, anchor="mm")
            if not good:
                d.line(tr([ca, cb]), fill=fade(GREY, a), width=2)
                m = ((ca[0] + cb[0]) / 2, (ca[1] + cb[1]) / 2)
                d.line(tr([(m[0], m[1] - 60), (m[0], m[1] + 60)]), fill=fade(RED, a), width=7)
                arrow(d, (ox + m[0], oy + m[1]), (ox + m[0] + 70, oy + m[1]), RED, 4, 12, a)
                para(d, (ox - 10, oy + 260), "Normal from the line between centres, one average area: the wrong angle and size.", 19, 360, RED, a)
            else:
                d.line(tr([(140, 0), (200, 230)]), fill=fade(GREEN, a), width=8)
                mx, my = 170, 115
                nx, ny = 230, -60
                L = (nx * nx + ny * ny) ** 0.5
                arrow(d, (ox + mx, oy + my), (ox + mx + nx / L * 70, oy + my + ny / L * 70), GREEN, 4, 12, a)
                d.ellipse([ox + mx - 6, oy + my - 6, ox + mx + 6, oy + my + 6], fill=fade(GREEN, a))
                para(d, (ox - 10, oy + 260), "Exact polygon: true area, centroid, normal, second moments, rebar crossings.", 19, 360, GREEN, a)
        a = ease((t - 2.6) / 0.5)
        text(d, (W // 2, 640), "Wrong contact geometry means wrong stresses. How wrong? →", 24, True, INK, a, anchor="mm")
        yield (img, "why_bond_contact") if i == T - 1 else img

CHAIN = [("chain_input", "Input", "48 authored parts + materials"),
         ("chain_cells", "Cells", "6,860 exact cells from recipes"),
         ("chain_l1", "Fracture modes (L1)", "618 pieces, weakest cuts first"),
         ("chain_l2", "Solver chunks (L2)", "1,182 chunks about 0.5 m across"),
         ("chain_hulls", "Convex hulls", "3,655 hulls, never overlapping"),
         ("chain_bonds", "Bonds", "3,657 bonds; joints, cuts weaker"),
         (None, "Export", ".glb + .fracphys + report, in 28 s")]

def s_chain():
    T = 112
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "END TO END", "Drop in a mesh. Get chunks, colliders and bonds.", BLUE, 1.0,
               "The whole pipeline on the two-storey building, in one bake")
        for k, (nm, ttl, sub) in enumerate(CHAIN):
            a = ease((t - 0.3 - 0.55 * k) / 0.4)
            if a <= 0: continue
            row, colk = (0, k) if k < 4 else (1, k - 4)
            x = 30 + colk * 310 + (155 if row else 0); y = 140 + row * 268
            if nm:
                im = frame(nm, 0, (280, 180)) if nm != "chain_bonds" else None
                if nm == "chain_bonds":
                    raw = Image.open(f"{R}/chain_bonds/000.png").convert("RGB")
                    raw = raw.crop((0, 36, raw.width // 2, 36 + 420))
                    s_ = min(280 / raw.width, 180 / raw.height)
                    raw = raw.resize((int(raw.width * s_), int(raw.height * s_)), Image.LANCZOS)
                    im = Image.new("RGB", (280, 180), BG); im.paste(raw, ((280 - raw.width) // 2, (180 - raw.height) // 2))
                paste(img, im, (x, y), a=a)
            else:
                d.rounded_rectangle([x, y, x + 280, y + 180], radius=12, fill=mix(BG, INK, a))
                for j, f_ in enumerate(("building.glb", "building.fracphys", "building.report.md")):
                    d.text((x + 140, y + 50 + j * 40), f_, font=F(20, mono=True), fill=mix(mix(BG, INK, a), (230, 230, 240), a), anchor="mm")
            d.ellipse([x, y - 36, x + 28, y - 8], fill=fade(BLUE, a))
            text(d, (x + 14, y - 22), str(k + 1), 17, True, WHITE if a > 0.5 else BG, a, anchor="mm")
            text(d, (x + 38, y - 22), ttl, 20, True, INK, a, anchor="lm", maxw=240)
            text(d, (x, y + 192), sub, 17, False, GREY, a, maxw=290)
            if k not in (3, 6):
                nx = x + 290 if k != 3 else None
                if nx: arrow(d, (x + 284, y + 90), (x + 304, y + 90), GREY, 3, 9, a)
        a = ease((t - 4.6) / 0.5)
        text(d, (W // 2, 668), "Every level nests inside the one above, and every level gets hulls and bonds.", 22, True, INK, a, anchor="mm")
        yield (img, "why_chain") if i == T - 1 else img

BUILT = [("Voronoi fracture", "classic", "material recipes (radial glass, dicing, grain, masonry), exact clipping"),
         ("Fracture modes", "Sellán et al. 2023", "weakest cuts first, material-aware, open-source solver, building scale"),
         ("Convex decomposition", "CoACD, Wei et al. 2022", "pure Rust; neighbours never overlap; tighter fit (6% vs 21%)"),
         ("Stress-solver bonds", "support graphs, e.g. Blast", "exact geometry (~10× lower stress error); weaker joints and cuts"),
         ("Hierarchy", "chunk hierarchies, e.g. Blast", "4 nested levels with consistent bonds; detail only where it breaks"),
         ("Validation", "—", "14 hard gates; FEM, Voro++ and CoACD oracles; byte-identical builds")]

def s_builton():
    T = 92
    for i in range(T):
        t = i / FPS; img, d = new()
        header(d, "BUILT ON PROVEN IDEAS", "Each step improved for destruction", PURPLE, 1.0)
        for x_, lab in ((50, "Technique"), (310, "Builds on"), (600, "What Shardwright adds")):
            text(d, (x_, 112), lab, 20, True, GREY, ease(t / 0.4))
        for k, (a_, b_, c_) in enumerate(BUILT):
            a = ease((t - 0.3 - 0.45 * k) / 0.4)
            y = 150 + k * 78
            d.line([(50, y - 8), (1230, y - 8)], fill=fade(LIGHT, a), width=1)
            text(d, (50, y + 22), a_, 19, True, INK, a, anchor="lm", maxw=250)
            text(d, (310, y + 22), b_, 19, False, GREY, a, anchor="lm", maxw=280)
            para(d, (600, y + 6), c_, 19, 620, INK, a)
        a = ease((t - 3.6) / 0.5)
        text(d, (W // 2, 640), "Not one algorithm: the whole chain, consistent from one exact cell complex.", 23, True, PURPLE, a, anchor="mm")
        yield img

def s_trust():
    T = 62
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
        text(d, (W // 2, 356), "For games, simulation and film.", 22, False, GREY, ease((t - 0.7) / 0.6), anchor="mm")
        a = ease((t - 1.0) / 0.6)
        if a > 0:
            s = "prefracture bake --input building.glb --config support_clusters.toml --out out/"
            f = F(22, mono=True); tw = f.getlength(s)
            d.rounded_rectangle([W // 2 - tw / 2 - 24, 380, W // 2 + tw / 2 + 24, 432], radius=10, fill=mix(BG, INK, a))
            d.text((W // 2, 406), s, font=f, fill=mix(mix(BG, INK, a), (230, 230, 240), a), anchor="mm")
        text(d, (W // 2, 500), "github.com/Glavin001/Shardwright", 24, True, BLUE, ease((t - 1.6) / 0.6), anchor="mm")
        yield img

SCENES = [s_title, s_problem, s_pitch, s_modes, s_materials, s_inputs, s_convex, s_bonds, s_support, s_hierarchy, s_chain, s_builton, s_trust, s_end]
XF = 6
if __name__ == "__main__":
    if os.path.isdir(OUT): shutil.rmtree(OUT)
    os.makedirs(OUT)
    DOCS = ROOT + "/docs/readme"
    n = 0; prev = None; marks = []; stills = {}
    for sc in SCENES:
        marks.append((sc.__name__, n)); CUR["scene"] = sc.__name__
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
    over = sorted(set(OVERFLOW))
    for o in over: print("TEXT OVERFLOW:", o)
    print(len(over), "text overflows")
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
    rows = [stills[f"beat_input_{k}"] for k in ("material", "impact", "grain")]
    fig = Image.new("RGB", (W, H + 2 * (H - 100)), BG)
    fig.paste(rows[0], (0, 0)); fig.paste(rows[1].crop((0, 100, W, H)), (0, H)); fig.paste(rows[2].crop((0, 100, W, H)), (0, 2 * H - 100))
    fig.resize((1024, fig.height * 1024 // W), Image.LANCZOS).save(f"{DOCS}/why_inputs.png", optimize=True)
    grid = Image.new("RGB", (W, H), BG)
    for k in range(4): grid.paste(stills[f"beat_material_{k}"].resize((W // 2, H // 2), Image.LANCZOS), ((k % 2) * W // 2, (k // 2) * H // 2))
    grid.save(f"{DOCS}/why_materials.png", optimize=True)
    print(n, "frames,", n / FPS, "s", marks)
