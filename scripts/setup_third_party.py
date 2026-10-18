"""外部のモデルと学習済み方策を third_party/ に取ってくる (リポジトリには含めない)。

- pollen-robotics/microduck_rl (決まったコミット): Microduck の MJCF とメッシュ (CC BY-NC-SA 4.0)
- pollen-robotics/microduck-policies (Hugging Face): 学習済みの方策 ONNX (Apache-2.0)
uv run python scripts/setup_third_party.py
"""
import os
import subprocess
import urllib.request

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TP = os.path.join(ROOT, "third_party")
RL_URL = "https://github.com/pollen-robotics/microduck_rl.git"
RL_COMMIT = "b02fdea4902c2b93b7d34a83a6f05da4ececb7d3"
HF = "https://huggingface.co/pollen-robotics/microduck-policies/resolve/main/"
POLICIES = ["alpha_walking.onnx", "velstand.onnx", "alpha_stand.onnx", "alpha_sitstand.onnx",
            "alpha_ground_pick.onnx", "manifest.json"]


def main():
    os.makedirs(TP, exist_ok=True)
    rl = os.path.join(TP, "microduck_rl")
    if not os.path.isdir(rl):
        subprocess.check_call(["git", "clone", "-q", RL_URL, rl])
    subprocess.check_call(["git", "-C", rl, "checkout", "-q", RL_COMMIT])
    pol = os.path.join(TP, "policies")
    os.makedirs(pol, exist_ok=True)
    for f in POLICIES:
        dst = os.path.join(pol, f)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(HF + f, dst)
    print("third_party ready:", rl, pol)


if __name__ == "__main__":
    main()
