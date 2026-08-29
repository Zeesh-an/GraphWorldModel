"""
Which graph tasks exist and what each one needs.

`--task` validates against this registry, and `Layout` uses the name as the first
level of the results tree, so `results/<task>/<dataset>/<run>/` and
`research/<task>.md` always agree on spelling.

Every entry here is RUNNABLE. Five further tasks were screened, reviewed and
dropped in August 2026: cascading failure, graph completion, influence
estimation, network inference and temporal forecasting, and their literature
reviews were removed with them; `git log` is the record. `status` and `blocker`
survive so a future entry can be added in the un-shipped state without changing
the dataclass or the readers that print it.
"""

from dataclasses import dataclass

from data.wm_simulator import (
    default_action_ops,
    blocked,
    epidemic_dynamics,
    spent,
    valid_action_ops,
    valid_remove_semantics,
)

implemented = "implemented"
planned = "planned"
out_of_scope = "out_of_scope"

# Objective sense the planner optimizes. `recover` tasks invert the forward model
# instead of choosing an intervention, so they have no action ops; `forecast` tasks
# emit no action at all and predict a SCALAR the process produces.
maximize = "maximize"
minimize = "minimize"
recover = "recover"
forecast = "forecast"

# What `coding_agent.types.improves` should do with each family's reward. Held here
# rather than inferred at each call site because two of the four are not their own
# name: a `recover` task's reward is an F1 and MAXIMIZES, and a `forecast` task's is
# a prediction ERROR and MINIMIZES. A run that read the objective literally would
# optimize a cascade-prediction arm toward the worst MSLE it could find.
objective_sense = {
    maximize: maximize,
    minimize: minimize,
    recover: maximize,
    forecast: minimize,
}

default_run = "default"


@dataclass()
class Task:
    name: str
    title: str
    status: str
    objective: str | None
    dynamics: tuple
    action_ops: tuple
    summary: str
    # What remove_node means for this task. `spent` (the node is a used-up
    # spreader, stays counted) is right for maximization; every containment task
    # needs `blocked` (the node is deleted from the graph), because `spent`
    # counts each immunized node as infected and biases the spread by +k.
    remove_semantics: str = spent
    # Arms this task sweeps by default. None = pipeline.conditions.default_arms
    # (the six-condition ladder). A task overrides this only when its question
    # needs a second method held alongside the ladder, the way adaptive IM needs
    # a matched non-adaptive control to divide by.
    default_arms: tuple | None = None
    # Condition-1 pool. None = pipeline.conditions.default_baselines (the static
    # IM classics). A task whose published baselines are a different KIND of
    # algorithm overrides it: adaptive IM's are per-round policies, and running
    # only static ones would leave its own literature off the table.
    default_baselines: tuple | None = None
    # Which ops a generated strategy may EMIT, and which the data generator
    # injects. `allowed_ops` defaults to `action_ops`; `gen_action_ops` defaults
    # to the repo-wide `default_action_ops` (the two node ops), so a task opts
    # into the edge ops or into none at all explicitly. CND narrows the planner
    # to `remove_node` (edge removals ride along inside the deletion bag, and
    # are not the planner's to spend budget on).
    default_allowed_ops: tuple | None = None
    default_gen_action_ops: tuple | None = None
    # The op one unit of budget buys. `add_node` for a seeding task, `remove_node`
    # for a containment one; the executor counts it and the prompt states it.
    budget_op: str = "add_node"
    # Budget sweep this task defaults to, as percentages of N. None = the
    # pipeline's 1/5/10/20 ladder. A task overrides it when its k is a property of
    # the INSTANCE rather than a choice: source localization is handed the source
    # count it has to recover, so a four-point budget sweep would be four runs of
    # the same experiment.
    default_budget_pcts: tuple | None = None
    # ...or an ABSOLUTE sweep, which takes precedence over the percentage one. Only
    # influence blocking sets it, and for a documented reason rather than taste:
    # research/influence_blocking.md §8.2 records that percent-of-N budgets are used
    # by NOBODY in that literature, while k in {10..50} is the shared convention of
    # SandIMIN, both Xie papers and TC-AIBM. A percentage sweep there would produce a
    # table comparable to no published number at all.
    default_budgets: tuple | None = None
    # Fraction of N the exogenous outbreak seeds, for a task whose cascade the
    # planner does not start. 0 = the planner seeds it (every maximize task).
    outbreak_pct: float = 0.0
    # TWO cascades rather than one: the planner answers a rumour with a counter-
    # cascade of its own, which needs the competitive simulator, an 8-channel state
    # and a 4-target head. Only influence blocking sets it, and it is what
    # `TaskSpec.blocks` is built from: the difference from plain containment is that
    # the budget can buy something that SPREADS, not only something that deletes.
    competitive: bool = False
    # The recovered object is a TRAJECTORY rather than a set, which changes the
    # contract as much as `competitive` changes the simulator: `reconstruct()`
    # instead of `localize()`, a tree-weighted reward instead of F1, and a
    # transmission edge in the data that NDlib does not otherwise produce. It
    # narrows `recovers` exactly as `blocks` narrows `contains`: both members of
    # the family invert something, and only this one has to name an edge.
    reconstructs: bool = False
    # FOUR EXCLUSIVE COMPARTMENTS rather than two overlapping indicators, and the
    # one task in this registry whose dynamics are not monotone: `I -> R` under
    # SIR/SEIR and `I -> S` under SIS both SHRINK the infectious set. That is what
    # `TaskSpec.immunizes` is built from, and it narrows `contains` exactly as
    # `blocks` does: the difference from plain containment is not what the budget
    # buys but what the process DOES, and it is why the structured heads' monotone
    # composition had to be replaced by a per-node transition matrix rather than
    # extended (research/epidemic_control.md §2.4).
    epidemic: bool = False
    # The transitions are REPLAYED FROM A LOG rather than produced by a simulator,
    # and it is the one field in this registry that changes where the data comes
    # from at all. `data/wm_cascades.py` replays a real cascade corpus into the same
    # JSONL every other task simulates, which costs three things nothing else here
    # gives up: the targets go HARD (a real cascade happened once, so
    # `--mc-marginals` has nothing to average), the counterfactual forks have
    # nothing to fork on, and the split must be CHRONOLOGICAL because the cascades
    # share a wall clock (research/cascade_prediction.md §2.4, §8.3).
    #
    # It is a field rather than a property of `forecast` because the two are
    # genuinely separable: a forecast task on SIMULATED episodes would keep soft
    # targets and the counterfactual forks, which is a different data path from a
    # replayed log even though both predict a scalar.
    observational: bool = False
    blocker: str | None = None

    @property
    def research_doc(self) -> str:
        return f"research/{self.name}.md"

    @property
    def sense(self) -> str:
        """
        maximize or minimize, for the reward THIS task's arms actually report.

        Never the objective read literally: a `recover` task reports an F1 and a
        `forecast` task reports a prediction error, so two of the four families
        would be optimized backwards by a call site that used `objective` directly.
        """
        return objective_sense.get(self.objective or maximize, maximize)

    @property
    def runnable(self) -> bool:
        return self.status == implemented

    # `is None` rather than `or`: an EMPTY override is meaningful. Source
    # localization generates diffusion-only episodes on purpose (no injected
    # actions at all, so the transition degenerates to f(G, s_t) -> s_{t+1}), and
    # `or` would read that empty tuple as "unset" and inject node ops anyway.
    @property
    def allowed_ops(self) -> tuple:
        return self.action_ops if self.default_allowed_ops is None else self.default_allowed_ops

    @property
    def gen_action_ops(self) -> tuple:
        # None = the repo-wide default, the two node ops; a task opts into the
        # edge ops (influence blocking, epidemic control) or into none at all
        # (the three tasks whose episodes must be pure diffusion) explicitly
        return (
            default_action_ops
            if self.default_gen_action_ops is None
            else self.default_gen_action_ops
        )

    @property
    def contains(self) -> bool:
        """
        True when the planner fights a cascade it did not start.

        The one structural difference between this task family and the seeding
        one: an exogenous outbreak has to be injected, the budget buys removals
        rather than seeds, and every "is this better" comparison flips sign.
        """
        return self.objective == minimize

    @property
    def recovers(self) -> bool:
        """
        True when the program INVERTS the transition instead of steering it.

        The structural difference is bigger than the containment one: there is no
        rollout, no action bag and no intervention at all. The program is handed a
        graph and an observed diffusion state and returns the seed set that
        produced it, scored on F1 against ground truth. The world model stops being
        the thing being optimized against and becomes a subroutine the generated
        program calls (research/source_localization.md §2.5).
        """
        return self.objective == recover

    @property
    def forecasts(self) -> bool:
        """
        True when the program PREDICTS a scalar the process produces rather than
        steering or inverting it.

        The fourth family, and the one that gives up the most: `a_t` is NULL at
        every step, so `T_exo` is the identity, none of the five ops fire, and the
        world model degenerates from an action-conditioned simulator into a
        forecaster (research/cascade_prediction.md §2.1). That is a bad property
        for a headline task and the exact property that makes it a stress test,
        conditions 3-6 still differ, because what varies down that ladder is the
        FORWARD MODEL the predictor may call, not the intervention it may choose.
        """
        return self.objective == forecast


tasks = {
    "influence_maximization": Task(
        name="influence_maximization",
        title="Influence Maximization",
        status=implemented,
        objective=maximize,
        # The INTERVENTION this task is defined by. The two `default_*_ops` below
        # are the wider set the pipeline has always shipped, and they are pinned
        # rather than derived: `remove_node` under `spent` is a legal (if almost
        # always bad) move, and the model is trained on both node ops so the
        # action channels are exercised. Every checkpoint in `results/` was
        # produced under these, so narrowing them to `action_ops` would silently
        # change what a rerun means.
        dynamics=("IC", "LT"),
        action_ops=("add_node",),
        default_allowed_ops=("add_node", "remove_node"),
        default_gen_action_ops=("add_node", "remove_node"),
        summary="Choose k seeds maximizing expected spread sigma(S).",
    ),
    "influence_blocking": Task(
        name="influence_blocking",
        title="Influence Blocking",
        status=implemented,
        objective=minimize,
        dynamics=("IC", "LT"),
        # All four of §1.1's levers, and the one task that makes `remove_edge` and
        # `set_edge_weight` load-bearing rather than implemented-and-idle. Which one
        # an arm actually spends its budget on is `--blocking-lever`, which sets
        # `budget_op` and `allowed_ops` together; `add_node` is the default because
        # counter-seeding is the founding and by far the largest sub-literature.
        action_ops=("add_node", "remove_node", "remove_edge", "set_edge_weight"),
        remove_semantics=blocked,
        competitive=True,
        summary="Two competing cascades; place blockers to minimize the negative one.",
        # The blocker only ever seeds POSITIVELY: S_N is an input to the episode
        # rather than an action (research/influence_blocking.md §2.1), which is what
        # lets the three action channels keep the meaning they have everywhere else.
        default_allowed_ops=("add_node",),
        budget_op="add_node",
        # All four levers are generated so ONE checkpoint serves every lever: the
        # head has to have seen a counter-seed, a deletion, a cut and a reweight to
        # predict any of them. `add_edge` is absent because no row of §1.1 uses it.
        default_gen_action_ops=(
            "add_node",
            "remove_node",
            "remove_edge",
            "set_edge_weight",
        ),
        # |S_N|, not the blocker budget. §5.4 is the reason it is 1% and not more:
        # CLDAG's Table 2 shows that at |S_N| = 1000 on NetHEPT even 1000 blockers
        # remove 17% of the negative spread, so a big rumour puts every method in a
        # regime where nothing works and every arm ties at "barely anything".
        outbreak_pct=1.0,
        # §8.2, and the one convention this task breaks with every other: percent-of-N
        # budgets speak to NO blocking paper. SandIMIN, both Xie papers and TC-AIBM
        # all report absolute k in {10..50} or {10..100}, so that is what we report,
        # and the informative ratio |S_P| / |S_N| is a reported column rather than the
        # sweep axis.
        default_budgets=(10, 20, 30, 40, 50),
        # Left None DELIBERATELY, unlike every other task's: condition 1's pool here
        # depends on the LEVER, not the task, because a lever can only emit what its
        # own members return: `proximity` hands back node ids and
        # `kimura_link_blocking` hands back arcs. `pipeline.run.resolve_baselines`
        # therefore reads `blocking_algorithms.default_blocking_baselines[lever]`,
        # where each list leads with the row that actually has to be beaten
        # (`proximity`, `imin_lhga`, `kimura_link_blocking`) and carries
        # `degree_blocking` as the published FAILURE mode rather than as a floor.
        default_baselines=None,
        blocker=None,
    ),
    "critical_node_detection": Task(
        name="critical_node_detection",
        title="Critical Node Detection",
        status=implemented,
        objective=minimize,
        dynamics=("IC", "LT"),
        action_ops=("remove_node", "remove_edge"),
        remove_semantics=blocked,
        summary="Remove k nodes to minimize the eventual spread of an outbreak.",
        # The DIFFUSION variant (research/critical_node_detection.md §2.2), not
        # the structural one. §2.1 argues at length that structural CNDP is a bad
        # fit for a world model: its T_endo is nothing, and its ground truth
        # (`nx.connected_components`, O(N+E)) is cheaper than one forward pass, so
        # a learned surrogate has nothing to amortize. The connectivity
        # functionals are still computed and reported per arm as descriptive
        # context (§8.3), just never as the learned target.
        #
        # A planner emits `remove_node` only. Edge removals are the deletion bag
        # `containment.expand_removals` builds around each one, so charging the
        # planner for them would make k mean deg(v) different things per node.
        # Generation injects the default two node ops: the outbreak itself is an
        # `add_node` bag at t=0, so the head has to have seen that channel too.
        default_allowed_ops=("remove_node",),
        default_gen_action_ops=("add_node", "remove_node"),
        budget_op="remove_node",
        # The cascade is exogenous and the planner is TOLD where it is, which makes
        # this reactive containment rather than the literature's blind dismantling
        # (§2.1). The budget must then sit BELOW the outbreak's one-hop ring, or
        # deleting the ring contains everything and every outbreak-aware arm ties at
        # |S|: at 1% on power_grid the ring was 129 nodes and three of the four
        # sweep budgets cleared it. 10% of N puts the ring above the 20% budget on
        # every sparse benchmark graph; `outbreak_ring` in each result and the
        # report's note say whether a budget was trivial, whatever this is set to.
        outbreak_pct=10.0,
        # §8.3 and §9.3, in ascending order of danger. `adaptive_degree` is the
        # one that hurts: MIND's Table 5 puts plain HDA at 119.9 against FINDER's
        # 115.0, so a learned dismantler that does not clearly beat it has
        # demonstrated nothing. `iterative_betweenness` is Wandelt's best-in-70-80%
        # method that almost no learned paper reports, and omitting it would
        # reproduce the exact methodological gap that survey calls out.
        # One representative per family, and the reinserting variant wherever the
        # published method HAS one: §8.2 trap 2 is that `X` and `X+R` are cited
        # under one name and are not the same method. `bpd` and `min_sum` are the
        # message-passing pair; `corehd` and `decycling` are the greedy versions of
        # the same two stages, so the four together isolate what inference buys.
        default_baselines=(
            "adaptive_degree",
            "iterative_betweenness",
            "collective_influence_r",
            "corehd",
            "bpd_r",
            "decycling",
            "explosive_immunization",
            "gnd",
            "netshield",
            # The outbreak-aware control: every other row is blind to where the
            # cascade is, and the agent's first power_grid win was this ring
            "frontier_removal",
            "degree_removal",
            "random_removal",
        ),
        blocker=None,
    ),
    "epidemic_control": Task(
        name="epidemic_control",
        title="Epidemic Control",
        status=implemented,
        objective=minimize,
        # The three COMPARTMENTAL dynamics, simulated by data/wm_epidemic.py rather
        # than NDlib. research/epidemic_control.md §2.2 is the reason we wrote our
        # own: NDlib's SIRModel/SISModel/SEIRModel all declare an EMPTY edge
        # parameter dict and compare a draw against one scalar beta, which deletes
        # `set_edge_weight`, degenerates `GraphInput.edge_weight` to ones, and makes
        # `structured_residual` (defined as an anchor on logit(w)) inexpressible.
        dynamics=("SIR", "SIS", "SEIR"),
        # §2.5's four levers, minus contact tracing (which changes the OBSERVATION,
        # not the graph, and belongs in a POMDP observation model we do not have).
        # Which one an arm spends its budget on is `--epi-lever`, which sets
        # `budget_op` and `allowed_ops` together; `vaccinate` is the default because
        # permanent immunization is what every published method in §3 does.
        action_ops=("remove_node", "remove_edge", "set_edge_weight"),
        # Vaccination: immune, non-infectious, never counted in the outbreak. Note
        # this is a DIFFERENT thing from SIR's own Removed compartment, where a
        # recovered node must stay counted: §8.2 trap 7's "recovered is not
        # removed", so the simulator keeps `blocked` (the intervention) and `R`
        # (the compartment) as separate sets and the head reads both.
        remove_semantics=blocked,
        epidemic=True,
        summary="Vaccinate, quarantine, or reduce contact to minimize an outbreak.",
        # A planner emits BARE lever ops. Under `vaccinate` the incident edge
        # removals ride along inside the deletion bag `epidemic.expand_immunization`
        # builds, exactly as they do for critical node detection, so charging the
        # planner for them would make k mean deg(v) different things per node.
        default_allowed_ops=("remove_node",),
        # All three ops are GENERATED so one checkpoint serves every lever: the head
        # has to have seen a dose, a cut and a reweight to predict any of them.
        default_gen_action_ops=("remove_node", "remove_edge", "set_edge_weight"),
        budget_op="remove_node",
        # The outbreak is exogenous. 1% of N is the standard immunization setup and
        # matches the smallest point of the --budget-pcts ladder, so the k=1% row is
        # "one dose per index case".
        outbreak_pct=1.0,
        # Left None DELIBERATELY, like influence blocking's: condition 1's pool here
        # depends on the LEVER, not the task, because a lever can only emit what its
        # own members return: `netshield` hands back node ids and `netmelt` hands
        # back arcs. `pipeline.run.resolve_baselines` reads
        # `immunization_algorithms.default_immunization_baselines[lever]`, where each
        # list leads with the row that actually has to be beaten
        # (`degree_immunization`, `netmelt`) and carries `netshield` and `dava`
        # together because §8.2 trap 1 is only visible when both are run.
        default_baselines=None,
        blocker=None,
    ),
    "source_localization": Task(
        name="source_localization",
        title="Source Localization",
        status=implemented,
        objective=recover,
        dynamics=("IC", "LT"),
        # The recovered source set IS an `add_node` bag: the environment converts
        # x_hat into exactly the seed commit the generator writes at t=0 for any
        # re-simulation it needs (research/source_localization.md §2.4.1). Nothing
        # is ever emitted as an intervention: a localize() program returns node
        # ids, but naming the op keeps `predict_marginals` and the --compare
        # referee speaking the same action vocabulary as every other task.
        action_ops=("add_node",),
        summary="Recover the seed set s_0 from an observed diffusion state s_T.",
        # k is a property of the INSTANCE here, not a choice: the localizer is told
        # how many sources to name. A four-point budget sweep would be four runs of
        # one experiment, so the default is SL-VAE's single 10%-of-N convention.
        # Pass --budget-pcts 5 10 20 with --sl-budget sweep for §8.5.1's
        # source-fraction axis.
        default_budget_pcts=(10.0,),
        # Diffusion-only episodes: no injected actions at any t > 0, so the
        # transition degenerates to f(G, s_t) -> s_{t+1} and every episode's whole
        # observable history is caused by its t=0 seed commit alone. That is what
        # makes the (x, y) label pair well defined: an episode with a mid-cascade
        # injection has an observation its seed set did not produce (§2.1).
        default_gen_action_ops=(),
        # §2.6 arm 1, one representative per family of §3, ordered by how dangerous
        # each is. `lpsi` heads the list because it is the row that actually has to
        # be beaten: SIDSL's Table 1 puts a 2017 label-propagation method with no
        # learning at F1 0.544 on Digg against SL-VAE's 0.479 and DDMSL's 0.517
        # [verified, §5.5], and LPSI sits INSIDE the agent's expressible space, so
        # "the search rediscovers LPSI" is the realistic floor (§2.9 risk 1).
        # `rumor_centrality` is single-source and scores near zero under a
        # multi-source protocol by construction (§8.2): it is here because it
        # founded the field and because a --budgets 1 run makes it admissible.
        # `resim_greedy` is deliberately absent for the same reason `celf` and
        # `greedy_blocking` are: it re-simulates every candidate on its own private
        # simulator, which would dominate startup for a table that exists to set a
        # bar. Run it as its own --baselines arm when you want its number.
        default_baselines=(
            "lpsi",
            "netsleuth",
            "ojc",
            "jordan_center",
            "dmp_localize",
            "dynamic_age",
            "effective_distance",
            "infected_degree",
            "rumor_centrality",
            "random_sources",
        ),
        # The six-condition ladder plus arm A (§2.6): §2.2's framing, in which the
        # world model is FROZEN and a relaxed source vector is gradient-descended
        # against it. That is SL-VAE with our likelihood plugged in, which the seed
        # paper has already declared a no-op component swap, so it is the CONTROL
        # program search is measured against, not the method. 6 vs A is the
        # methodological claim this task exists to make.
        default_arms=(
            "routing",
            "evolve_free@native",
            "evolve_free@monte_carlo",
            "evolve_free@oracle",
            "evolve_free@world_model",
            "gradient_free@world_model",
        ),
        blocker=None,
    ),
    "cascade_reconstruction": Task(
        name="cascade_reconstruction",
        title="Cascade Reconstruction",
        status=implemented,
        objective=recover,
        reconstructs=True,
        dynamics=("IC", "LT"),
        # Named for the same reason source localization names it: nothing is ever
        # emitted as an intervention: a reconstruct() program returns
        # `{node: (time, parent)}`, but the recovered SOURCES (the nodes whose
        # parent is None) are an `add_node` bag, which is what the --compare
        # re-simulation referee replays. Naming the op keeps every arm speaking one
        # action vocabulary (research/cascade_reconstruction.md §2.5.1).
        action_ops=("add_node",),
        summary="Recover the hidden trajectory from partial observations.",
        # `budget` is NOT spent on anything here: a decoder emits no action and is
        # handed no k. The single point exists because the sweep axis is the
        # OBSERVATION RATE (--cr-observation-rate) and the SETTING (--cr-setting),
        # and running the same experiment four times under four unused budgets
        # would be four identical rows (§8.3).
        default_budget_pcts=(10.0,),
        # Diffusion-only episodes, for a stronger version of source localization's
        # reason: actions are NULL throughout so T_exo = identity and the whole
        # factorization collapses to s_{t+1} = T_endo(s_t) (§2.1). A mid-cascade
        # injection would put a jump in the trajectory that no transition kernel
        # can explain, and the decoder would be scored on inverting it.
        default_gen_action_ops=(),
        # §2.9 arm 1, one representative per family of §3, ordered by how dangerous
        # each is. `delayed_bfs` heads the list because it is the row that actually
        # has to be beaten: Xiao SDM'18 gets node precision > 0.8 from an
        # O(m + k log k) BFS variant, and it sits INSIDE the agent's expressible
        # space. `personalized_pagerank` is second because §8.2 trap 4 records that
        # it BEATS tree sampling on `ca_grqc` specifically (assortativity 0.164),
        # if the search cannot clear it there, the result is not real.
        # `observed_only` is Rozenshtein's `Reports` control: precision 1.0 by
        # construction, and the row that proves the reward is not gameable.
        # `mcmc_decode` and `forward_backward` are deliberately absent for the same
        # reason `celf` and `resim_greedy` are: they call the transition kernel
        # 10^4 times per instance and would dominate startup for a table that
        # exists to set a bar. Run either as its own --baselines arm.
        default_baselines=(
            "delayed_bfs",
            "personalized_pagerank",
            "ordered_steiner_closure",
            "greedy_ordered",
            "tree_sampling",
            "cult",
            "consistent_tree_wpct",
            "dhrec",
            "cri",
            "netfill",
            "observed_only",
            "random_reconstruction",
        ),
        # The six-condition ladder plus arm A (§2.9): DITTO's decoder with our
        # kernel substituted for its mean-field beta-hat. That is §2.3's framing,
        # the component swap the tempting move would make, so it is the CONTROL
        # decoder SEARCH is measured against, not the method. 6 vs A is the
        # methodological claim this task exists to make, and §2.11 risk 6 is why it
        # is built well: DITTO beats a supervised model trained with the TRUE beta
        # on two of eight rows, so a weak arm A makes the comparison meaningless.
        default_arms=(
            "routing",
            "evolve_free@native",
            "evolve_free@monte_carlo",
            "evolve_free@oracle",
            "evolve_free@world_model",
            "decode_free@world_model",
        ),
        blocker=None,
    ),
    "adaptive_online_im": Task(
        name="adaptive_online_im",
        title="Adaptive and Online Influence Maximization",
        status=implemented,
        objective=maximize,
        dynamics=("IC", "LT"),
        action_ops=("add_node",),
        # Same reasoning as influence_maximization: the pipeline's shipped pair,
        # pinned so the checkpoints already on disk keep meaning what they meant
        default_allowed_ops=("add_node", "remove_node"),
        default_gen_action_ops=("add_node", "remove_node"),
        summary="Choose seeds over rounds, observing realized activations between them.",
        # The published adaptive algorithms (AdaptGreedy, EPIC) plus the per-round
        # heuristic floor and `static_split`, the same-machinery control that
        # isolates the timing penalty from the adaptivity benefit. `celf_pp` and
        # `imm` ride along as the strongest STATIC seed sets, which is what the
        # adaptivity gap divides by (research/adaptive_online_im.md §8.1).
        default_baselines=(
            "adapt_greedy",
            "adapt_epic",
            "adapt_degree_discount",
            "adapt_random",
            "static_split",
            "celf_pp",
            "imm",
        ),
        # Every adaptive arm needs the non-adaptive arm at the same k to divide
        # by: the adaptivity gap is a RATIO, and a multi-round spread number on
        # its own says nothing (research/adaptive_online_im.md 5.1, 9.3 item 3).
        # Same evaluator on both sides of each pair, so the gap is not confounded
        # by evaluator fidelity.
        default_arms=(
            "routing",
            "evolve_free@native",
            "adaptive_free@native",
            "evolve_free@monte_carlo",
            "adaptive_free@monte_carlo",
            "evolve_free@oracle",
            "adaptive_free@oracle",
            "evolve_free@world_model",
            "adaptive_free@world_model",
        ),
        blocker=None,
    ),
    "cascade_prediction": Task(
        name="cascade_prediction",
        title="Cascade / Popularity Prediction",
        status=implemented,
        objective=forecast,
        # REAL LOGGED CASCADES, not NDlib. This is the one task in `research/` whose
        # literature refuses to use a simulator, and §9.1 is why that is the point
        # rather than an obstacle: every other task here evaluates a learned model
        # against traces drawn from the same simulator that trained it, a closed loop
        # that can only measure LEARNING error and never MODELLING error. Weibo
        # retweets break the loop: our structured head's
        # `p_new(v) = 1 - prod(1 - q(u->v) * frontier_u)` is either an adequate
        # approximation of whatever produced them or it is not, and that is
        # measurable. `observational` is what routes the data stage into
        # `data/wm_cascades.py` instead of the simulator.
        observational=True,
        # The dynamics label names the KERNEL the transitions are read under, not a
        # claim about what produced them. §2.2: the IC composition rule is a
        # modelling commitment rather than a learned fact, so fitting the IC head to
        # real retweets IS the experiment. `metadata.json` carries an `observed`
        # block so nobody reads `transitions_IC_train.jsonl` as simulated.
        dynamics=("IC", "LT"),
        # EMPTY, and it is the defining property of this task rather than an
        # omission (§2.1): nothing intervenes, the cascade is only watched. None of
        # the five ops fire, `action_sensitivity` is undefined, and the
        # counterfactual forks that carry most of our training signal have nothing
        # to fork on. What still varies across conditions 3-6 is the forward model
        # the predictor may CALL, which is why the ladder survives at all.
        action_ops=(),
        summary="Predict a real cascade's final size from its early window.",
        # Diffusion-only, and here the constraint is stronger than source
        # localization's: a logged cascade HAS no injection to replay, so an action
        # op is not merely unhelpful but unrepresentable.
        default_gen_action_ops=(),
        # `budget` buys nothing: a predictor emits no action and is handed no k. The
        # single point exists because the sweep axis is the OBSERVATION WINDOW
        # (--cp-observation) and the SPLIT (--cp-split), and §8.2 requires two
        # windows rather than four budgets: "a single-window result is not
        # publishable in this literature".
        default_budget_pcts=(10.0,),
        # §3, one representative per family, ordered by how dangerous each is.
        # `szabo_huberman` heads the list because it is the row that actually has to
        # be beaten: one feature, one line, from 2008, and every paper in §5 still
        # prints it as "Feature-S&H". `feature_linear` and `feature_gbt` are CTCP's
        # own MLP/XGBoost rows, and CasFlow's ablation records that feature models
        # "in some cases even beat deep learning models". `seismic` and `hawkes` are
        # §3.2's generative pair: the closest classical analogue to a world model,
        # and the ones that DECLINE to score supercritical cascades, which is the
        # `n_failed` column §8.4 says almost nobody publishes. `persistence` and
        # `mean_size` are the two floors MSLE rewards far more than people expect.
        # `mc_forward` is deliberately absent for the same reason `celf` and
        # `mcmc_decode` are: it pays kernel calls per instance and would dominate
        # startup for a table that exists to set a bar. Run it as its own
        # --baselines arm when you want its number.
        default_baselines=(
            "szabo_huberman",
            "feature_linear",
            "feature_gbt",
            "seismic",
            "hawkes",
            "hawkes_hybrid",
            "rpp",
            "hip",
            "branching_factor",
            "weng_communities",
            "persistence",
            "mean_size",
            "random_prediction",
        ),
        blocker=None,
    ),

}


# A typo in an action op above would silently produce a task whose ops the
# simulator rejects only at generation time
for _task in tasks.values():
    for _label, _ops in (
        ("action_ops", _task.action_ops),
        ("allowed_ops", _task.allowed_ops),
        ("gen_action_ops", _task.gen_action_ops),
    ):
        _unknown = set(_ops) - set(valid_action_ops)
        if _unknown:
            raise ValueError(
                f"task {_task.name!r} declares unknown {_label} {sorted(_unknown)}; "
                f"valid ops are {sorted(valid_action_ops)}"
            )

    if _task.remove_semantics not in valid_remove_semantics:
        raise ValueError(
            f"task {_task.name!r} declares unknown remove_semantics "
            f"{_task.remove_semantics!r}; valid values are {valid_remove_semantics}"
        )

    # A budget op the planner may not emit is a task whose every candidate is
    # rejected at validation: cheaper to catch here than one budget in. Guarded on
    # `runnable` so an entry added in the un-shipped state can carry the ops it
    # WOULD need before `budget_op` is settled.
    #
    # A forecast task is exempt because it spends NO budget at all: `a_t` is NULL at
    # every step, no plan is ever validated, and naming an op it may emit would
    # misdescribe the one property that defines it (§2.1).
    if _task.runnable and not _task.forecasts and _task.budget_op not in _task.allowed_ops:
        raise ValueError(
            f"task {_task.name!r} spends budget on {_task.budget_op!r} but does "
            f"not allow the planner to emit it (allowed_ops={_task.allowed_ops})"
        )

    # `spent` counts an immunized node as infected, which biases any containment
    # objective by exactly +k (research/critical_node_detection.md §2.3)
    if _task.contains and _task.remove_semantics != blocked:
        raise ValueError(
            f"task {_task.name!r} minimizes spread but uses remove_semantics="
            f"{_task.remove_semantics!r}; a containment task needs {blocked!r}"
        )

    # An inverse task labels each episode with its t=0 seed commit, so an episode
    # carrying a mid-cascade injection has an observation its seed set did not
    # produce and its (x, y) pair is a lie (research/source_localization.md §2.1)
    if _task.runnable and _task.recovers and _task.gen_action_ops:
        raise ValueError(
            f"task {_task.name!r} inverts the transition but generates with "
            f"action ops {_task.gen_action_ops}; a recover task needs "
            f"default_gen_action_ops=() so every episode's observation is caused "
            f"by its t=0 seed set alone"
        )

    # A competitive task's whole premise is a rumour it did not start, so an empty
    # S_N would leave the blocker answering nothing and every arm tied at zero
    if _task.competitive and not _task.outbreak_pct:
        raise ValueError(
            f"task {_task.name!r} is competitive but seeds no negative cascade "
            f"(outbreak_pct=0); a blocker with no rumour to answer scores the same "
            f"as every other blocker"
        )

    # A trajectory decoder is scored on a TREE, and a tree needs a parent that only
    # `recover`-family data carries; a reconstructs entry that did not invert would
    # be asking the planner to steer a cascade and be graded on explaining it
    if _task.reconstructs and not _task.recovers:
        raise ValueError(
            f"task {_task.name!r} sets reconstructs=True but its objective is "
            f"{_task.objective!r}; recovering a hidden trajectory is a "
            f"{recover!r} objective, not an intervention"
        )

    # A compartmental task fights an outbreak it did not start, exactly as every
    # other containment task does; an epidemic entry with no outbreak would leave
    # every arm dosing a graph nothing ever spreads on and tied at zero
    if _task.epidemic and not _task.outbreak_pct:
        raise ValueError(
            f"task {_task.name!r} is compartmental but seeds no outbreak "
            f"(outbreak_pct=0); a dose allocation with no epidemic to stop scores "
            f"the same as every other allocation"
        )

    # ...and its dynamics have to BE compartmental, or the compartment head would be
    # built for a simulator that never produced its targets
    if _task.epidemic and set(_task.dynamics) - set(epidemic_dynamics):
        raise ValueError(
            f"task {_task.name!r} is compartmental but declares dynamics "
            f"{_task.dynamics}; only {epidemic_dynamics} produce the four exclusive "
            f"compartments the head composes"
        )

    # ...and a compartmental task is never also two-cascade: the layouts disagree on
    # what every column means and no head reads both
    if _task.epidemic and _task.competitive:
        raise ValueError(
            f"task {_task.name!r} sets both epidemic and competitive; a dataset is "
            f"either four-COMPARTMENT or two-CASCADE, never both"
        )

    # A logged cascade has no injection to replay, so an action op here is not
    # merely unhelpful but unrepresentable (research/cascade_prediction.md §2.1)
    if _task.runnable and _task.forecasts and _task.gen_action_ops:
        raise ValueError(
            f"task {_task.name!r} forecasts an observed process but generates with "
            f"action ops {_task.gen_action_ops}; a_t is NULL at every step, so it "
            f"needs default_gen_action_ops=()"
        )

    # ...and a replayed corpus is the only thing that makes the falsification test
    # of §9.1 a test at all: replaying NDlib would close the loop it exists to break
    if _task.observational and not _task.forecasts:
        raise ValueError(
            f"task {_task.name!r} sets observational=True but its objective is "
            f"{_task.objective!r}; replaying a real log rather than simulating is a "
            f"{forecast!r} objective, and every other family needs the "
            f"counterfactual forks a log cannot provide"
        )

    # ...and it cannot also fight an exogenous cascade: the sources ARE the unknown
    if _task.recovers and _task.outbreak_pct:
        raise ValueError(
            f"task {_task.name!r} inverts the transition but declares an outbreak; "
            f"the source set is what it is solving for, not something handed to it"
        )


def task_names() -> list[str]:
    return sorted(tasks)


def runnable_task_names() -> list[str]:
    return sorted(name for name, task in tasks.items() if task.runnable)


def get_task(name: str) -> Task:
    if name not in tasks:
        raise ValueError(f"unknown task {name!r}; choose one of {task_names()}")

    return tasks[name]


def require_runnable(name: str) -> Task:
    task = get_task(name)

    if not task.runnable:
        raise ValueError(
            f"task {name!r} is {task.status}, not runnable by this pipeline. "
            f"{task.blocker or ''} "
            f"See {task.research_doc} for the full analysis. "
            f"Runnable today: {runnable_task_names()}"
        )

    return task
