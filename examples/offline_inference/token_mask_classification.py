# SPDX-License-Identifier: Apache-2.0
"""Example for repeated token classification on the last [MASK] token.

This demonstrates how to use vLLM with an encoder model such as BERT
for a repeated classification task. A static prefix is combined with
variable tokens and a final ``[MASK]`` token. The script repeatedly
runs the model and prints the probability for the positive class after
applying a sigmoid to the raw logit of the ``[MASK]`` position.
"""

from __future__ import annotations

from argparse import Namespace
from pathlib import Path
from typing import Iterable

import torch
from transformers import AutoTokenizer

from vllm import LLM, EngineArgs
from vllm.utils import FlexibleArgumentParser


_DEF_MODEL = "bert-base-uncased"


def _read_lines(fp: Path) -> Iterable[str]:
    if not fp.exists():
        return []
    with fp.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield line


def parse_args():
    parser = FlexibleArgumentParser()
    parser = EngineArgs.add_cli_args(parser)
    parser.add_argument(
        "--static-prefix",
        type=str,
        default="",
        help="Text that forms the first 400 tokens of every prompt.",
    )
    parser.add_argument(
        "--variable-file",
        type=Path,
        default=None,
        help="Optional file with one variable prompt per line.",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=1000,
        help="Number of iterations to run.",
    )
    parser.set_defaults(
        model=_DEF_MODEL,
        task="classify",
        enforce_eager=True,
        override_pooler_config="{\"softmax\": false}",
    )
    return parser.parse_args()


def main(args: Namespace) -> None:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    mask_id = tokenizer.mask_token_id
    if mask_id is None:
        raise ValueError("The tokenizer does not define a [MASK] token.")

    prefix_ids = tokenizer(args.static_prefix, add_special_tokens=False)[
        "input_ids"]

    if args.variable_file:
        variable_texts = list(_read_lines(args.variable_file))
    else:
        variable_texts = [f"iteration {i}" for i in range(args.iterations)]

    llm = LLM(**vars(args))

    for i, text in enumerate(variable_texts[: args.iterations]):
        variable_ids = tokenizer(text, add_special_tokens=False)["input_ids"]
        input_ids = prefix_ids + variable_ids
        input_ids = input_ids[:511]
        input_ids.append(mask_id)
        prompt = tokenizer.decode(input_ids)

        (output,) = llm.classify(prompt)
        logits = output.outputs.probs
        if not logits:
            raise RuntimeError("Received empty logits from model.")
        prob = torch.sigmoid(torch.tensor(logits[0])).item()
        print(f"Step {i}: P=\u007bprob:.4f\u007d")


if __name__ == "__main__":
    args = parse_args()
    main(args)
