import math, os, sys
import mujoco, numpy as np
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "sim"))
from scene import compose
from duck import Duck
spec = compose(os.path.join(ROOT, "houses", "nobita"))
duck = Duck(spec)
duck.reset(2.9, 4.5, 0.0)
for _ in range(100): duck.step()
c = mujoco.mj_name2id(duck.model, mujoco.mjtObj.mjOBJ_CAMERA, "head_camera")
R = duck.data.cam_xmat[c].reshape(3, 3)
print("pose", duck.pose(), "cam pos", duck.data.cam_xpos[c].round(3))
print("right(x)", R[:, 0].round(2), "up(y)", R[:, 1].round(2), "forward(-z)", (-R[:, 2]).round(2))
print("fovy", duck.model.cam_fovy[c])
