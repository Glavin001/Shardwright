"""Cut one scene of the explainer (out/why/frames, scene starts in
out/why/data/marks.json) into its own GIF for a README section.

Usage: snippet.py SCENE OUT.gif [WIDTH] [FPS]"""
import json, os, subprocess, sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
scene, out = sys.argv[1], sys.argv[2]
width = int(sys.argv[3]) if len(sys.argv) > 3 else 760
fps = int(sys.argv[4]) if len(sys.argv) > 4 else 10
marks = json.load(open(ROOT + "/out/why/data/marks.json"))
names = [m[0] for m in marks]
k = names.index(scene)
start = marks[k][1]
end = marks[k + 1][1] if k + 1 < len(marks) else len(os.listdir(ROOT + "/out/why/frames"))
start += 6   # skip the cross-fade from the previous scene
subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-framerate", "12", "-start_number", str(start),
                "-i", ROOT + "/out/why/frames/%05d.png", "-frames:v", str(end - start),
                "-vf", f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];"
                       "[a]palettegen=max_colors=112:stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                out], check=True)
print(out, end - start, "frames from", scene)
