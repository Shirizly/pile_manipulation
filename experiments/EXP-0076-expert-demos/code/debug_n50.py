import sys, time, json, torch
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE))
import tasks as T
from simcore import Sim
import beam as BM, wide_planner as wp
sc = [s for s in json.load(open(HERE.parent / "results/scenarios.json"))["scenarios"] if s["n"] == 50][:2]
sim = Sim(50, 4)
st = torch.stack([torch.tensor(s["state0"], device="cpu").float() for s in sc])
t0 = time.time(); bank = sim.bank(st, 4096); print("bank", time.time() - t0, bank.shape, flush=True)
import argparse
a = argparse.Namespace(H=1, gd=20, pool="fl", n_pool=256, pop=256, iters=4, distinct_mm=3.0, max_depth=0, model="ens2bal", phases=None, phases_list=None, B=1, k=1)
scn = BM.Scenario(sc[0], a, None); nd = scn.beam[0]
S = nd["state"]; xy = S[:, :2].cuda()
t0 = time.time(); ex = wp.fetch_candidates(xy, scn.mask_t); print("fetch", time.time() - t0, ex.shape, flush=True)
from model.retrieval.frame import yaw_from_quat
legal = wp.make_legal_scorer(S[:, :2], yaw_from_quat(S[:, 3:7]))
t0 = time.time(); r = legal(ex[:256]); torch.cuda.synchronize(); print("legal 256", time.time() - t0, flush=True)
m = BM.get_model("ens2bal"); obj = wp.SeqObjective(m, S, scn.dist, scn.mask_t, 1.0)
t0 = time.time(); c = obj.cost(ex[:256, None, :]); torch.cuda.synchronize(); print("cost 256", time.time() - t0, flush=True)
t0 = time.time(); out = BM.propose(scn, nd, bank[0], a, 1); print("propose", time.time() - t0, flush=True)
t0 = time.time(); ns = sim.step_from(S[None], out[0][:1].cpu()); print("step", time.time() - t0, flush=True)
