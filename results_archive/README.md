# Results archive

Every result JSON produced by the Q1–Q5 experiments, pulled from the run host
(`grandriver.egr.msu.edu:/egr/research-dselab/lihang4/Coworker/hongji/GraphWorldModel`).
46 files, 784 KB. Committed because a result nobody can find is a result nobody
can check.

What is here and what is not:

  here      every metrics JSON — training runs, frozen OOD evaluations, ranking
            evaluations, the Q5 transfer, the depth study, the OOD manifest.
            Each one carries its own provenance block (checkpoint spec, split
            mode, graph ids, train/val overlap, seed).

  not here  transition JSONL datasets (~26 MB) and `.pt` checkpoints (12).
            Both regenerate exactly from `--seed 42` plus the recorded config,
            and both are archived on the run host under
            `../GWM_archive/gwm_{results,datasets,code}_20260822.tar.gz`.

Layout mirrors the run host:

    results/influence_maximization/ba100/   the canonical BA-100 x 100 source
        wm_structured_IC|LT/                Q2 treatment arms
        wm_linear_*_pw{off,auto}/           Q2 controls
        wm_depth_L{1,2,5,8}/                Q3 receptive-field study
        ranking_structured_IC.json          Q4 on the source distribution
    results/ood/                            Q3/Q4 targets, all eval_only
        eval_IC_*.json                      frozen-model evaluation per target
        ranking_IC_*.json                   Q4 ranking per target
        depth_L*_*.json                     depth x target grid
        manifest.json                       structural disjointness proof
    results/q5/                             IM -> Adaptive IM frozen transfer
