import sys, torch, numpy as np
sys.path.insert(0, "experiments/EXP-0074-wide-domain-zoom-nfd/code")
SHARDS = ["scattered_n20", "scattered_n50", "scattered_n100", "inbetween_n20", "inbetween_n50", "inbetween_n100", "piled_n20", "piled_n50"]
ts = torch.load("experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/test_sets.pt", weights_only=False)
key = lambda P0, P1: set(map(tuple, np.round(torch.cat([P0, P1], 1).numpy() * 1e6).astype(np.int64)))
for cache in sys.argv[1:] or ("zoom_r64", "zoom_r128", "world_r64", "world_r128"):
    try: D = torch.load(f"experiments/EXP-0074-wide-domain-zoom-nfd/artifacts/{cache}.pt")
    except Exception as e: print(cache, "missing"); continue
    tr = D["train"]; out = []
    for si, sh in enumerate(SHARDS):
        m = tr["shard"] == si; ktr = key(tr["P0"][m], tr["P1"][m]); t = ts[sh]["t"]; rows = torch.from_numpy(ts[sh]["rows"]); kte = key(t["P0"][rows], t["P1"][rows])
        out.append(len(kte & ktr) / len(kte))
    print(cache, "fraction of cached TEST rows (one-step set) also in the train cache per shard:", np.round(out, 2), "mean", round(float(np.mean(out)), 3), flush=True)
