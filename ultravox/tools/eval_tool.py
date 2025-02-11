import argparse
import dataclasses
from typing import IO
import wandb
import os
from tqdm import tqdm
import numpy as np

import simple_parsing

from ultravox.evaluation import eval
from ultravox.evaluation import eval_types


@dataclasses.dataclass
class EvalArgs:
    # model name for wandb
    model: str = simple_parsing.field(alias="-M")
    # Path to the audio file
    file: IO = simple_parsing.field(type=argparse.FileType("rb"), alias="-f")
    # Metric to use for evaluation (e.g., "asr" or "boolq")
    metric: str = simple_parsing.field(default="asr", alias="-m")
    # Verbose output
    verbose: bool = simple_parsing.field(default=False, alias="-v")


def main(args: EvalArgs):
    
    if args.metric == "instruct":
        from ultravox.evaluation.gpt_eval_instruct import INSTRUCT_SYSTEM_PROMPT, INSTRUCT_USER_PROMPT
        from ultravox.evaluation.gpt_eval import RATING_MODEL

        config = {
            "rating_model": RATING_MODEL,
            "instruct_system_prompt": INSTRUCT_SYSTEM_PROMPT,
            "instruct_user_prompt": INSTRUCT_USER_PROMPT,
        }

    wandb.init(
        project="smolvox",
        name=args.model,
        config={
            "model": args.model,
            "metric": args.metric,
            **config,
        },
        dir=os.path.join(os.getcwd(), "evals"),
        save_code=True,
    )

    scores = []
    lines = args.file.readlines()
    pbar = tqdm(enumerate(lines), total=len(lines))
    for i, line in pbar:
        sample = eval_types.Sample.from_json(line)
        result = eval.evaluate_answer(sample, metric=args.metric)
        assert result.score is not None, "Rating failed."
        scores.append(result.score)
        if args.verbose and isinstance(result, eval_types.InstructResult):
            average = sum(scores) / len(scores)
            print(f"{i}: score={result.score:.2f} average={average:.2f}")
            print(f"  reason={result.reason}")
        if len(scores) > 0:
            pbar.set_postfix({'avg_score': np.mean(scores)})
        wandb.log(
            {"eval_samples": wandb.Table(
                data=[[sample.question, sample.expected_answer, sample.generated_answer, result.score]], 
                columns=["question", "expected_answer", "generated_answer", "score"])}
        )
    wandb.log({"average_score": np.mean(scores)})        


if __name__ == "__main__":
    main(simple_parsing.parse(EvalArgs))
