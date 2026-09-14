# Why these fields exist

Six things went wrong in this repo before the corresponding field existed.
Prose warnings did not stop any of them; a validator reading actual fields
would have. Kept here, out of the skill body, so field requirements don't
read as ceremony to a future reader who didn't live through these — and so
removing one is a deliberate decision made with the incident in view, not an
accidental tidy-up.

| What happened | Field that prevents it |
|---|---|
| Occupancy and plate channels were mutually transposed for an entire research programme; "nothing beats persistence" was an artifact of the bug, not a finding | `depends_on` + a backing invariant test |
| Two rows of one comparison came from two different rasterisers — the risk was warned about in prose, then it happened anyway | `provenance.code_path` |
| One view was compared at σ=1 blur against another at σ=0, so the view effect and the blur effect were observationally identical | `design.held_fixed` |
| An effect of ~0.001 was interpreted against a fold-to-fold sd of ~0.004 | `noise_floor`, stated before the result |
| Per-pixel error was measured for a long time before anyone asked whether it bore on the actual claim | `downgrades: [indirectness]` |
| When a transpose bug was eventually found, nothing indexed which claims it invalidated | `REGISTER.md`'s `depends_on` column |

The downgrade-domain table (`register-validator`) and this repo's git
history hold the fuller story of each episode; this file exists so the six
rows above survive without the full narrative needing to be reloaded every
time someone is tempted to drop a field as bureaucracy.
