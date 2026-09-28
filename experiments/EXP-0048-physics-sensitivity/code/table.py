"""Print the EXP-0048 markdown table from results/sensitivity.json."""
import json
from pathlib import Path
d = json.loads((Path(__file__).resolve().parents[1] / "results/sensitivity.json").read_text())
print(f"n_states={d['n_states']} x {d['n_actions']} pushes; ref displacement {d['ref_displacement_mm']}")
print("| condition | input change (L1/mass) | moved-cube final diff mm med / p90 | displacement diff mm med / p90 | per-push max diff mm med | frac cubes >2 mm | outcome image L1/mass med / p90 | abs dv diff / between-action sd med / p90 | Spearman med / p10 | top-1 same | top-1 regret / sd mean | pushes with mean diff > repeat |")
print("|---|---|---|---|---|---|---|---|---|---|---|---|")
for c, v in d["conditions"].items():
    m, dd, im, dv, r = (v["cube_diff_mm_moved_in_ref"], v["displacement_diff_mm_moved_in_ref"], v["img_L1_over_mass"],
                        v["abs_dv_diff_over_between_action_sd"], v["spearman_dv_across_actions"])
    print(f"| {c} | {v['input_perceptibility_L1_over_mass']:.3f} | {m['median']:.2f} / {m['p90']:.2f} | {dd['median']:.2f} / {dd['p90']:.2f} "
          f"| {v['per_push_max_cube_diff_mm']['median']:.2f} | {v['frac_cubes_diff_gt_2mm']:.3f} | {im['median']:.3f} / {im['p90']:.3f} "
          f"| {dv['median']:.3f} / {dv['p90']:.3f} | {r['median']:.3f} / {r['p10']:.3f} | {v['top1_same']:.2f} "
          f"| {v['top1_regret_over_sd']['mean']:.3f} | {v['frac_pushes_mean_diff_gt_repeat']:.2f} |")
