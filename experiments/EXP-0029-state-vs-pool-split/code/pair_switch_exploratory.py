"""EXP-0029 exploratory follow-up (NOT pre-registered): the cross-fitted
switching gain between the two closest models only (nfd_3ch_randlen vs
nfd_warped_randlen), where switching is not dominated by a large quality gap.
Reads split_test.py's saved predictions/truth; same split procedure."""
import sys
from pathlib import Path
import numpy as np, torch
REPO = Path(__file__).resolve().parents[3]; sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).parent))
from split_test import ART, capture, VFS
from Baselines.common.goals import higher_is_better_for
from Genesis.binned_slate_dataset import BinnedSlateCorpus
rows = BinnedSlateCorpus.load(str(REPO / "Genesis/data/slates_binned/n20_scatter_s20a1000_L20-70mm")).step(0)
sl = rows.slate_idx.numpy(); S = sl.max() + 1
dvt = torch.load(ART / "truth.pt", weights_only=False)["dv"].numpy()
pair = ["nfd_3ch_randlen", "nfd_warped_randlen"]
dvp = {m: torch.load(ART / f"pred_{m}.pt", weights_only=False)["dv"].numpy() for m in pair}
rng = np.random.default_rng(1)
idx = [np.nonzero(sl == s)[0] for s in range(S)]
for K in (128, 500):
    for vi, vf in enumerate(VFS):
        hib = higher_is_better_for(vf); gains = []; per_state_gain = []
        for r in range(200):
            c = np.zeros((2, 2, S))
            for s in range(S):
                p = rng.permutation(idx[s]); pools = (p[:K], p[K:2 * K])
                for h, pool in enumerate(pools):
                    for mi, m in enumerate(pair):
                        c[h, mi, s] = np.nanmean(capture(dvp[m][pool, :, vi], dvt[pool, :, vi], hib))
            single = c[0].mean(1).argmax()
            pick = c[0].argmax(0)
            g = c[1][pick, np.arange(S)] - c[1][single]
            gains.append(g.mean()); per_state_gain.append(g)
        gains = np.array(gains)
        print(f"K={K} {vf:15s}: switching gain {gains.mean():+.4f} [{np.quantile(gains,.025):+.4f},{np.quantile(gains,.975):+.4f}] "
              f"positive in {(gains>0).mean():.0%} of splits")
