# NGFLO Final Research Package

## Manuscript
- `NGFLO_Final_Manuscript.tex`: complete 7-section manuscript.
- `NGFLO_Final_Manuscript.pdf`: compiled manuscript.

## Main figures
- Candidate-restriction gap versus K.
- Variable retention versus K.
- Commitment-aware dynamic runtime comparison.
- Dynamic loaded-vehicle comparison.
- Hard-residual stress runtime savings.

## Reproducible result tables
The `results/` directory contains the exact benchmark certification summary, held-out scorer recall, reduced-MIP candidate frontier, hardness-router predictions, complete three-intensity dynamic aggregate results, and replicated hard-residual stress results.

## Core code
The `code/` directory contains the ordered-slot benchmark model, graph-scorer training script, hardness-router builder, commitment-aware rolling-horizon implementation, and delayed-commitment stress implementation.

## Scope and limitations
All current experiments are synthetic. The modeled ferry is a simplified ordered-slot, single-ramp representation with lane/deck capacity, deck mass, a transverse-balance envelope, compatibility, and discharge-order constraints. It does not yet model operator-specific ramp networks, hydrostatics/ballast, axle loads, fire zones, dangerous-goods separation, or empirical terminal calibration. The manuscript states these limitations explicitly.

## Core result
The evidence supports NGFLO as a state-selective acceleration layer embedded inside an exact feasibility-certified optimizer. It does not support universal speed dominance over Full-MIP.
