"""Does the trained UNet use its action channel at all?

If the pile channel and the plate channel of a training sample are in mutually
transposed frames, the only way a network can condition on the action is to
learn the reflection between them. The cheapest way to find out whether it did
is to destroy the action channel at inference and see whether anything changes.
Inference only -- no training, no data collection.
"""
import torch, yaml, copy
from registry.model_registry import build_model
from registry.dataset_registry import build_dataset

run = "runs_cubes/unetfilm_corl_limited_100e"
cfg = yaml.safe_load(open(f"{run}/run_config.yaml").read())
wrapper = build_model(cfg["model"])
sd = torch.load(f"{run}/unet_best.pth", map_location="cpu", weights_only=False)
if isinstance(sd, dict) and "state_dict" in sd: sd = sd["state_dict"]
missing = wrapper.load_state_dict(sd, strict=False)
print("load:", missing)
wrapper.eval()

ds = build_dataset(cfg["dataset"], "val")
n = min(200, len(ds))
print(f"eval on {n} validation samples")

X, P, Y = [], [], []
for i in range(n):
    s = ds[i]; ig, ph, tg = s["input"], s["physics"], s["target"]
    X.append(ig); P.append(ph); Y.append(tg)
X, P, Y = torch.stack(X), torch.stack(P), torch.stack(Y)

def err(pred): return float(((pred - Y) ** 2).mean() ** 0.5)

with torch.no_grad():
    def fwd(x):
        out = wrapper({"input": x, "physics": P})
        if not torch.is_tensor(out): out = out["prediction"]
        return torch.sigmoid(out).squeeze(1)
    base = fwd(X)
    Xz = X.clone(); Xz[:, 1] = 0.0                    # action erased
    zero = fwd(Xz)
    Xs = X.clone(); Xs[:, 1] = X[torch.randperm(n), 1]  # action shuffled
    shuf = fwd(Xs)
    Xt = X.clone(); Xt[:, 0] = X[:, 0].transpose(1, 2)  # pile channel transposed
    trans = fwd(Xt)

print(f"\n{'variant':28s} {'rms':>9s} {'vs persistence':>15s}")
p = err(X[:, 0])
for name, v in [("persistence (copy input)", X[:, 0]), ("UNet, true action", base),
                ("UNet, action ZEROED", zero), ("UNet, action SHUFFLED", shuf),
                ("UNet, pile transposed", trans)]:
    print(f"{name:28s} {err(v):9.5f} {100*err(v)/p:14.1f}%")
print("\nIf 'action zeroed' / 'action shuffled' match 'true action', the network "
      "is not conditioning on the push at all.")
