"""Caption the rendered frames of a destruction sim (tools/sim/make.sh):
title, legend (level colours), a clock in sim time with the slow-motion
factor, and live counters from the sim JSON (bonds broken, rigid bodies,
chunks refined to the leaf level). With --minimal (the README hero), only a
small legend and the speed; with --compact, one strip along the bottom with
the legend and the counters (for close-ups, where the structure fills the
frame).

Usage: compose.py SIM.json RENDER_DIR OUT_DIR SLOWMO [--minimal | --compact]"""
import json, os, sys
from PIL import Image, ImageDraw, ImageFont

sim, src, out, slow = sys.argv[1], sys.argv[2], sys.argv[3], float(sys.argv[4])
minimal = "--minimal" in sys.argv
compact = "--compact" in sys.argv
os.makedirs(out, exist_ok=True)
for f in os.listdir(out):
    if f.endswith(".png"): os.remove(os.path.join(out, f))
frames = json.load(open(sim))["frames"]
FD = "/usr/share/fonts/truetype/dejavu/"
F = lambda s, b=False: ImageFont.truetype(FD + ("DejaVuSans-Bold.ttf" if b else "DejaVuSans.ttf"), s)
INK, GREY = (24, 24, 32), (90, 90, 102)
L2, L3 = (204, 201, 194), (245, 143, 51)
files = sorted(f for f in os.listdir(src) if f.endswith(".png"))
assert len(files) == len(frames), (len(files), len(frames))
leaf_total = 6860
for k, (fn, fr) in enumerate(zip(files, frames)):
    im = Image.open(os.path.join(src, fn)).convert("RGB")
    W, H = im.size
    d = ImageDraw.Draw(im)
    if compact:
        d.rectangle((0, H - 44, W, H), fill=(236, 236, 240))
        x = 18
        for col, label in ((L2, "L2 chunk"), (L3, "L3, refined on impact")):
            d.rectangle((x, H - 29, x + 14, H - 15), fill=col, outline=INK)
            d.text((x + 22, H - 31), label, font=F(15), fill=INK)
            x += 40 + int(F(15).getlength(label))
        d.text((W - 18, H - 31), f"t {fr['t']:.2f} s ({slow:g}x slow) · {fr['broken']:,} bonds broken · "
               f"{fr['islands']:,} bodies · {fr['refined']:,} refined", font=F(15), fill=GREY, anchor="ra")
        im.save(os.path.join(out, f"{k:05d}.png"))
        continue
    if minimal:
        x = 22
        for col, label in ((L2, "chunk"), (L3, "refined where it breaks")):
            d.rectangle((x, H - 36, x + 14, H - 22), fill=col, outline=INK)
            d.text((x + 22, H - 38), label, font=F(15), fill=GREY)
            x += 44 + int(F(15).getlength(label))
        d.text((W - 22, H - 38), f"Rapier · {slow:g}x slow motion", font=F(15), fill=GREY, anchor="ra")
        im.save(os.path.join(out, f"{k:05d}.png"))
        continue
    d.text((24, 18), "A wrecking ball through a two-storey building", font=F(26, True), fill=INK)
    d.text((24, 52), "Rapier rigid bodies, broken by a stress solver on Shardwright's bonds", font=F(17), fill=GREY)
    # legend
    y = 86
    for col, label in ((L2, "Level-2 chunk (1,182 to start)"), (L3, "Level-3 cell, refined where it breaks")):
        d.rectangle((26, y + 3, 42, y + 19), fill=col, outline=INK)
        d.text((52, y), label, font=F(16), fill=INK)
        y += 26
    # clock + counters, top right
    bodies = len({b[0] for b in fr["bodies"]})
    lines = [(f"t = {fr['t']:.2f} s", True), (f"{slow:g}x slow motion" if slow > 1 else "real time", False),
             (f"bonds broken  {fr['broken']:,}", False), (f"rigid bodies  {fr['islands']:,}", False),
             (f"chunks refined to Level 3  {fr['refined']:,}", False)]
    y = 20
    for text, bold in lines:
        d.text((W - 24, y), text, font=F(18, bold), fill=INK if bold else GREY, anchor="ra")
        y += 24
    d.text((W - 24, H - 30), "tools/sim  ·  reproducible: tools/sim/make.sh", font=F(14), fill=GREY, anchor="ra")
    im.save(os.path.join(out, f"{k:05d}.png"))
print("composed", len(frames), "frames into", out)
