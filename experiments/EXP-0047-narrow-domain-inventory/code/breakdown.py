"""Per-spawn-group breakdown for overnight_randlen and Sean n20 (adds entries to inventory.json)."""
import glob, json, os, sys
sys.path.insert(0, os.path.dirname(__file__))
from inventory import scan_files, summarise, phys_table, OUT, ROOT
res = json.load(open(OUT))
D = f"{ROOT}/Genesis/data"
for name, pat in [("overnight_randlen/mixed_n20", f"{D}/overnight_randlen/mixed/cube/n20/**/*_data.pt"),
                  ("overnight_randlen/piled_n20", f"{D}/overnight_randlen/piled/cube/n20/**/*_data.pt"),
                  ("overnight_randlen/scattered_n20", f"{D}/overnight_randlen/scattered/cube/n20/**/*_data.pt"),
                  ("Sean/inbetween_n20", f"{D}/Sean/inbetween*/*/cube/n20/**/*_data.pt"),
                  ("Sean/piled_n20", f"{D}/Sean/piled*/*/cube/n20/**/*_data.pt"),
                  ("Sean/scattered_n20", f"{D}/Sean/scattered*/*/cube/n20/**/*_data.pt")]:
    files = sorted(f for f in glob.glob(pat, recursive=True) if not f.endswith("_failed.pt"))
    acc, phys, ld = scan_files(files)
    res["breakdown:" + name] = summarise(acc.cat(), acc.links, acc.total, dict(path=pat, files=len(files), physics=phys_table(phys)))
    tmp = OUT + ".tmp"; json.dump(res, open(tmp, "w"), indent=1, default=str); os.replace(tmp, OUT)
    v = res["breakdown:" + name]
    print(name, {k: v[k] for k in ("files","n20_rows","n20_single_layer_rows","n20_sl_clump_rows","clump_frac_sl_quantiles","sl_18_22_perp2","sl_exact20_perp2","sl_clump_18_22_perp2","chain_links_narrow_18_22")}, [(x['settle_steps'],x['rows']) for x in v['physics']])
