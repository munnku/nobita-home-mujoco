import math, os, sys
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from world import World
w = World(duck_pose=(5.6 * 0.91, 3.0 * 0.91, 0.0), n_persons=0, policy=os.environ.get("POLICY", "alpha_walking"))
for _ in range(50): w.duck.step()
for cmd in [(0, 0, 1.0), (0, 0, -1.0), (0, 0, 0.5), (0, 0, -0.5), (0.1, 0, -0.8), (0.2, 0, 0), (0, 0, -1.5)]:
    w.duck.set_cmd(*cmd)
    _, _, y0 = w.duck.pose(); x0, yy0, _ = w.duck.pose()
    tot = 0.0; prev = y0
    for _ in range(200):
        w.duck.step(); _, _, yy = w.duck.pose()
        tot += (yy - prev + math.pi) % (2 * math.pi) - math.pi; prev = yy
    x1, y1, _ = w.duck.pose()
    print(cmd, f"turned {math.degrees(tot):.0f} deg in 4s, moved {math.hypot(x1-x0, y1-yy0):.2f} m")
