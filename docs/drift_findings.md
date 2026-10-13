# Drift findings (ALL DATA IS SYNTHETIC; rules pre-registered in docs/monitoring_plan.md)

- Frozen M2 at threshold 0.0927: crisis recall 1.000 on validation (optimistic; validation
  shares families with train), 0.841 [0.750, 0.903] on monitor weeks 17-26 (n_crisis=88).
- Localised: new D-* families recall 0.711 [0.552, 0.830], precision 0.243;
  old families recall 0.940 [0.838, 0.979], precision 0.734.
- R1 (label-free) fired week 18; R2 (delayed labels) fired week 20; drift began week 17.
  R1 is the earliest week a 2-consecutive-week rule allows.
- OOV is confounded with ASR noise (voice share rose 0.36 -> ~0.55); PSI hovers near its limit.
- Retrain at week 20 (333 monitor rows added), evaluated on weeks 21-26 (51 crisis):
  recall 0.863 -> 0.961, cost/1000 146.4 -> 65.2, but PR-AUC difference +0.075 has a family-cluster
  95% CI [-0.050, +0.182] that includes 0. Gain is NOT established. Drift families persist after the
  retrain week, so this is recovery from a persisting shift, not generalisation to unseen slang.
- Weekly recall is noisy (5-13 crisis cases per week); no single week should be read alone.
