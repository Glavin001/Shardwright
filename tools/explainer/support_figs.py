"""README figures for the support-graph section, from support.sh's renders:

  docs/readme/support_graph.png    the Level-0 support graph, with a legend
  docs/readme/support_load.png     the self-weight load path, with a legend
  docs/readme/support_columns.gif  each ground-floor column removed in turn:
                                   the chunks whose bonds go over capacity"""
import glob, json, os
from PIL import Image, ImageDraw, ImageFont

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
R, G, DOCS = ROOT + "/out/why/render", ROOT + "/out/why/graph", ROOT + "/docs/readme"
FD = "/usr/share/fonts/truetype/dejavu/"
F = lambda s, b=False: ImageFont.truetype(FD + ("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf"), s)
INK, GREY, BG = (24, 24, 32), (90, 90, 102), (247, 247, 250)

def panel(img, title, sub, legend):
    """Title above, legend below a render."""
    im = Image.open(img).convert("RGB")
    W = im.width
    out = Image.new("RGB", (W, im.height + 70 + 44), BG)
    d = ImageDraw.Draw(out)
    d.text((16, 10), title, font=F(22, True), fill=INK)
    d.text((16, 40), sub, font=F(15), fill=GREY)
    out.paste(im, (0, 70))
    x, y = 16, 70 + im.height + 12
    for col, lab in legend:
        d.rectangle((x, y + 2, x + 16, y + 18), fill=col, outline=INK)
        d.text((x + 24, y), lab, font=F(15), fill=INK)
        x += 48 + int(F(15).getlength(lab))
    return out

st0 = json.load(open(G + "/l0_stats.json"))
st2 = json.load(open(G + "/l2_stats.json"))
panel(R + "/support_graph/000.png", "Support graph (Level 0)",
      f"{st0['nodes']} parts, {st0['bonds']} bonds; each edge runs through its contact",
      [((242, 140, 31), "joint between parts"), ((140, 77, 191), "bearing"), ((20, 20, 26), "anchor to ground")]
      ).save(DOCS + "/support_graph.png", optimize=True)
panel(sorted(glob.glob(R + "/support_load/*.png"))[0], "Load path under self-weight (Level 2)",
      f"Exact stress solve of {st2['bonds']:,} chunk bonds; colour = force through each chunk",
      [((230, 232, 237), "little force"), ((140, 173, 230), "more"), ((31, 41, 140), "most")]
      ).save(DOCS + "/support_load.png", optimize=True)

# column positions on plan, for an inset marking the removed one
asset = json.load(open(ROOT + "/out/support_clusters/two_storey_building.asset.json"))
plan = {}
for c in asset["components"]:
    if c["name"].startswith("f0_col_"):
        v = c["solid"]["verts"]
        plan[c["name"]] = (sum(p[0] for p in v) / len(v), sum(p[2] for p in v) / len(v))
xs, zs = [p[0] for p in plan.values()], [p[1] for p in plan.values()]
def inset(img, removed):
    """Plan of the ground-floor columns (top view), the removed one red."""
    d = ImageDraw.Draw(img)
    S, x0, y0 = 110, img.width - 140, img.height - 44 - 160
    d.rectangle((x0 - 14, y0 - 14, x0 + S + 14, y0 + S + 30), fill=BG, outline=(200, 200, 208))
    for name, (x, z) in plan.items():
        px = x0 + S * (x - min(xs)) / (max(xs) - min(xs))
        py = y0 + S * (z - min(zs)) / (max(zs) - min(zs))
        if name == removed:
            d.line((px - 9, py - 9, px + 9, py + 9), fill=(219, 26, 31), width=4)
            d.line((px - 9, py + 9, px + 9, py - 9), fill=(219, 26, 31), width=4)
        else:
            d.rectangle((px - 6, py - 6, px + 6, py + 6), fill=(90, 100, 130))
    d.text((x0 + S / 2, y0 + S + 22), "columns, plan view", font=F(13), fill=GREY, anchor="mm")
    return img

frames = []
cols = sorted(glob.glob(R + "/support_columns/*.png"))
for k, p in enumerate(cols):
    c = os.path.basename(p)[:-len("_0000.png")]
    st = json.load(open(f"{G}/col_{c}_stats.json"))
    frames.append(panel(p, f"Remove one column ({k + 1} of {len(cols)}): {st['over_capacity_bonds']} bonds over capacity",
                        "Ground-floor column removed, structure re-solved: red chunks break first",
                        [((219, 26, 31), "over capacity"), ((247, 199, 199), "removed column"), ((140, 173, 230), "within capacity")]))
    inset(frames[-1], c)
frames = [f.resize((640, int(f.height * 640 / f.width)), Image.LANCZOS) for f in frames]
# one palette for every frame (from all of them side by side), so the legend
# colours stay true
strip = Image.new("RGB", (frames[0].width * len(frames), frames[0].height))
for k, f in enumerate(frames):
    strip.paste(f, (k * f.width, 0))
palette = strip.quantize(colors=255, method=Image.Quantize.MEDIANCUT)
pal = [f.quantize(palette=palette, dither=Image.Dither.NONE) for f in frames]
pal[0].save(DOCS + "/support_columns.gif", save_all=True, append_images=pal[1:], duration=1600, loop=0, optimize=True)
print("wrote support_graph.png, support_load.png, support_columns.gif (", len(frames), "columns )")
