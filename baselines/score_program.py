"""
Score one candidate program on the plain Monte Carlo simulator.

This is the fitness function every condition-9 framework calls, one subprocess
per candidate, from inside its own venv. The program is wrapped as a canned
Strategy and pushed through `run_experiment` exactly as a classical baseline or
an external seed set would be, so the number it gets is the number the arm being
compared against would get for the same program: same outbreak, same lever, same
selection split, same seed count. The world model is never constructed here.

    python -m baselines.score_program --context <work_dir>/discovery_context.json \
        --program candidate.py --json-out candidate.json
"""

import argparse
import json
import os
import time
from pathlib import Path

from baselines.discovery import oriented_fitness, program_script
from coding_agent.run import ExperimentConfig, run_experiment

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Score one discovered program")
    parser.add_argument(
        "--context",
        type=str,
        required=True,
        help="discovery_context.json written by the pipeline (default: required).",
    )
    parser.add_argument(
        "--program",
        type=str,
        required=True,
        help="path of the candidate program (default: required).",
    )
    parser.add_argument(
        "--json-out",
        type=str,
        default=None,
        help="write the score record here as JSON (default: None).",
    )

    args = parser.parse_args()
    context = json.loads(Path(args.context).read_text())
    experiment = dict(context["experiment"])
    experiment["allowed_ops"] = tuple(experiment["allowed_ops"])
    # One canned pass on the selection split, nothing else: no credit rollouts,
    # no agreement replay, no checkpoint, no results file of its own (the referee
    # replay is the canned evaluation itself, reused, so it costs nothing extra)
    experiment.update(
        outer_iters=1,
        mc_agreement=False,
        credit=False,
        out_json=None,
        resume=False,
        routing=False,
        baseline=None,
        transfer_from=None,
    )
    config = ExperimentConfig(**experiment)
    program = Path(args.program).read_text()
    script = program_script(program, context["task"], context["budget_op"], context["lever"])

    start = time.perf_counter()
    result = run_experiment(config, canned_script=script)
    reward = float(result["selection_reward"])
    record = {
        "reward": reward,
        "fitness": oriented_fitness(
            context["task"], result["objective"], reward, result["graph"]["num_nodes"]
        ),
        "sense": result["objective"],
        "reward_name": context["contract"]["reward_name"],
        "num_nodes": result["graph"]["num_nodes"],
        "evaluator_calls": result.get("evaluator_calls"),
        "evaluator_seconds": result.get("evaluator_seconds"),
        "seconds": round(time.perf_counter() - start, 3),
    }

    if args.json_out:
        os.makedirs(Path(args.json_out).parent, exist_ok=True)
        Path(args.json_out).write_text(json.dumps(record, indent=2))

    print(json.dumps(record))
    print(reward)
