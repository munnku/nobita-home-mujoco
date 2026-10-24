"""残差の拾い方を PPO で学習する (CPU。MuJoCo Warp は noslip を使えないので、評価と同じ物理で CPU で回す)。

    PYTHONUTF8=1 .venv/Scripts/python.exe learning/rl_pick/env.py --n 1200          # 始まりの状態を作る (約 5 分)
    PYTHONUTF8=1 .venv/Scripts/python.exe learning/rl_pick/train.py --iters 300     # 学習

環境は並列のプロセス (numpy だけで方策を動かす。torch は学習の本体だけ)。出力:
  learning/rl_pick/policy.npz (成功率が一番良かったときの方策)、build/rl_pick/train_log.json (学習曲線)。
"""
import argparse
import json
import os
import pickle
import sys
import time
from multiprocessing import Pipe, Process

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "learning"))
HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(ROOT, "build", "rl_pick")
HIDDEN = 128


def worker(conn, pool_path, seed, eval_path=None):
    import warnings
    warnings.filterwarnings("ignore")
    from rl_pick.env import PickEnv
    from rl_pick.residual import MLPPolicy
    with open(pool_path, "rb") as f:
        pool = pickle.load(f)
    eval_pool = []
    if eval_path:
        with open(eval_path, "rb") as f:
            eval_pool = pickle.load(f)
    env = PickEnv(pool, seed)
    rng = np.random.default_rng(seed + 12345)
    obs = env.reset()
    while True:
        cmd, params, T = conn.recv()
        if cmd == "close":
            break
        pol = MLPPolicy(params=params)
        if cmd == "eval":                    # 乱数を足さない方策で、取っておいた場面 (T = 自分の受け持ちの番号) を試す
            res = []
            for i in T:
                e = eval_pool[i]
                o = env.reset(e)
                while True:
                    o, r, done, info = env.step(pol(o))
                    if done:
                        break
                res.append((e["kind"], info["success"]))
            obs = env.reset()
            conn.send(res)
            continue
        std = np.exp(params["log_std"])
        O, A, L, R, D, infos = [], [], [], [], [], []
        for _ in range(T):
            mu = pol(obs)
            a = mu + std * rng.standard_normal(mu.shape)
            logp = -0.5 * np.sum(((a - mu) / std) ** 2 + 2 * np.log(std) + np.log(2 * np.pi))
            nobs, r, done, info = env.step(a)
            O.append(obs); A.append(a); L.append(logp); R.append(r); D.append(done)
            if done:
                infos.append(info)
                nobs = env.reset()
            obs = nobs
        conn.send((np.array(O), np.array(A), np.array(L), np.array(R), np.array(D, bool), obs, infos))


class RunningStat:
    def __init__(self, n):
        self.mean, self.var, self.count = np.zeros(n), np.ones(n), 1e-4

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), len(x)
        delta, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + delta * bc / tot
        self.var = (self.var * self.count + bv * bc + delta ** 2 * self.count * bc / tot) / tot
        self.count = tot

    @property
    def std(self):
        return np.sqrt(self.var) + 1e-6


def main():
    import torch
    import torch.nn as nn
    from rl_pick.residual import N_ACT, OBS_DIM
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", default=os.path.join(OUT_DIR, "pool_s1.pkl"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--steps", type=int, default=1024, help="1 回の集めで、環境ごとのステップ数")
    ap.add_argument("--iters", type=int, default=300)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--gamma", type=float, default=0.995)
    ap.add_argument("--lam", type=float, default=0.95)
    ap.add_argument("--clip", type=float, default=0.2)
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--minibatch", type=int, default=2048)
    ap.add_argument("--log-std", type=float, default=-1.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-pool", default=os.path.join(OUT_DIR, "pool_s2.pkl"),
                    help="学習に使わない場面。乱数なしの方策の成功率で一番良いものを残す")
    ap.add_argument("--eval-every", type=int, default=25)
    ap.add_argument("--log", default=os.path.join(OUT_DIR, "train_log.json"))
    ap.add_argument("--out", default=os.path.join(HERE, "policy.npz"))
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    torch.set_num_threads(4)
    os.makedirs(OUT_DIR, exist_ok=True)

    def mlp(out):
        net = nn.Sequential(nn.Linear(OBS_DIM, HIDDEN), nn.Tanh(), nn.Linear(HIDDEN, HIDDEN), nn.Tanh(),
                            nn.Linear(HIDDEN, out))
        return net
    actor, critic = mlp(N_ACT), mlp(1)
    with torch.no_grad():                    # 初めは補正 0 (= 今の拾い方) から
        actor[-1].weight.mul_(0.01)
        actor[-1].bias.zero_()
    log_std = nn.Parameter(torch.full((N_ACT,), a.log_std))
    opt = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()) + [log_std], lr=a.lr)
    rs = RunningStat(OBS_DIM)

    def params():
        p = {"obs_mean": rs.mean, "obs_std": rs.std, "log_std": log_std.detach().numpy().copy()}
        for i, li in enumerate((0, 2, 4)):
            p[f"w{i}"] = actor[li].weight.detach().numpy().T.copy()
            p[f"b{i}"] = actor[li].bias.detach().numpy().copy()
        return p

    conns, procs = [], []
    for i in range(a.workers):
        c1, c2 = Pipe()
        pr = Process(target=worker, args=(c2, a.pool, a.seed * 1000 + i, a.eval_pool), daemon=True)
        pr.start()
        conns.append(c1)
        procs.append(pr)

    with open(a.eval_pool, "rb") as f:
        n_eval = len(pickle.load(f))

    def det_eval(p):
        for i, c in enumerate(conns):
            c.send(("eval", p, list(range(i, n_eval, len(conns)))))
        res = [x for c in conns for x in c.recv()]
        by = {}
        for k, s in res:
            by.setdefault(k, []).append(s)
        return float(np.mean([s for _, s in res])), {k: float(np.mean(v)) for k, v in sorted(by.items())}

    log, best, t_start = [], -1.0, time.time()
    recent = []
    for it in range(a.iters):
        t0 = time.time()
        p = params()
        for c in conns:
            c.send(("run", p, a.steps))
        outs = [c.recv() for c in conns]
        t_col = time.time() - t0
        mean_t, std_t = torch.tensor(rs.mean, dtype=torch.float32), torch.tensor(rs.std, dtype=torch.float32)

        def norm(x):
            return torch.clamp((torch.as_tensor(x, dtype=torch.float32) - mean_t) / std_t, -10, 10)
        O, A, L, ADV, RET = [], [], [], [], []
        infos = []
        for o, act, lp, r, dn, last, inf in outs:
            with torch.no_grad():
                v = critic(norm(o)).squeeze(-1).numpy()
                v_last = float(critic(norm(last[None])).squeeze())
            adv = np.zeros(len(r))
            g = 0.0
            for t in reversed(range(len(r))):
                nv = v_last if t == len(r) - 1 else v[t + 1]
                nonterm = 0.0 if dn[t] else 1.0
                delta = r[t] + a.gamma * nv * nonterm - v[t]
                g = delta + a.gamma * a.lam * nonterm * g
                adv[t] = g
            O.append(o); A.append(act); L.append(lp); ADV.append(adv); RET.append(adv + v)
            infos += inf
        O, A, L = np.concatenate(O), np.concatenate(A), np.concatenate(L)
        ADV, RET = np.concatenate(ADV), np.concatenate(RET)
        ADV = (ADV - ADV.mean()) / (ADV.std() + 1e-8)
        On, At = norm(O), torch.as_tensor(A, dtype=torch.float32)
        Lt, ADVt, RETt = (torch.as_tensor(x, dtype=torch.float32) for x in (L, ADV, RET))
        n = len(O)
        for _ in range(a.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, a.minibatch):
                idx = perm[s:s + a.minibatch]
                mu = actor(On[idx])
                std = torch.exp(log_std)
                lp = (-0.5 * (((At[idx] - mu) / std) ** 2 + 2 * log_std + np.log(2 * np.pi))).sum(-1)
                ratio = torch.exp(lp - Lt[idx])
                pg = -torch.min(ratio * ADVt[idx], torch.clamp(ratio, 1 - a.clip, 1 + a.clip) * ADVt[idx]).mean()
                vl = ((critic(On[idx]).squeeze(-1) - RETt[idx]) ** 2).mean()
                loss = pg + 0.5 * vl
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(list(actor.parameters()) + list(critic.parameters()) + [log_std], 0.5)
                opt.step()
        rs.update(O)
        succ = [i["success"] for i in infos]
        recent = (recent + succ)[-400:]
        by_kind = {}
        for i in infos:
            by_kind.setdefault(i["kind"], []).append(i["success"])
        row = dict(iter=it, steps=(it + 1) * n, episodes=len(infos), success=float(np.mean(succ)) if succ else None,
                   success_recent=float(np.mean(recent)) if recent else None,
                   by_kind={k: float(np.mean(v)) for k, v in by_kind.items()},
                   std=np.exp(log_std.detach().numpy()).round(3).tolist(), t_collect=round(t_col, 1),
                   t_iter=round(time.time() - t0, 1), wall=round(time.time() - t_start))
        if it % a.eval_every == 0 or it == a.iters - 1:
            p = params()
            row["det_success"], row["det_by_kind"] = det_eval(p)
            if row["det_success"] > best:
                best = row["det_success"]
                np.savez(a.out, **p)
                row["saved"] = True
        log.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
        with open(a.log, "w", encoding="utf-8") as f:
            json.dump(log, f, ensure_ascii=False)
        np.savez(a.out.replace(".npz", "_last.npz"), **params())
    for c in conns:
        c.send(("close", None, 0))


if __name__ == "__main__":
    main()
