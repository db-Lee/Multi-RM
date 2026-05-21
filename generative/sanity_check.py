"""
Sanity check: verify dataset formatting and tokenization for a given model.

Usage:
    python -m generative.sanity_check --model_id Qwen/Qwen3-8B --task_type gORM
    python -m generative.sanity_check --model_id deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --task_type gORM
"""
import argparse

from datasets import load_dataset
from transformers import AutoTokenizer

from generative.prompt_formats import CHAT_TEMPLATE, ORM_PROMPT_FORMAT, PRM_PROMPT_FORMAT
from generative.data import add_eos, tokenize


def build_example(task_type: str, category: str) -> dict:
    data_path = "dongboklee/train_gORM" if task_type == "gORM" else "dongboklee/train_gPRM"
    prompt_format = ORM_PROMPT_FORMAT if task_type == "gORM" else PRM_PROMPT_FORMAT

    print(f"Loading one example from {data_path} / {category} ...")
    dataset = load_dataset(data_path, split=category)
    data = dataset[0]

    prompt = prompt_format(category, data["question"], data["cot"])
    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": data["critique"]}],
    }


def color(text: str, code: int) -> str:
    return f"\033[{code}m{text}\033[0m"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True)
    parser.add_argument("--task_type", type=str, default="gORM", choices=["gORM", "gPRM"])
    parser.add_argument("--category", type=str, default="math")
    args = parser.parse_args()

    print(f"\n{'='*70}")
    print(f"Model    : {args.model_id}")
    print(f"Task     : {args.task_type}")
    print(f"Category : {args.category}")
    print(f"{'='*70}\n")

    # ── Tokenizer ────────────────────────────────────────────────────────────
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if "deepseek" in args.model_id.lower():
        tokenizer.chat_template = CHAT_TEMPLATE
        print("Applied custom DeepSeek chat template.\n")
    else:
        print("Using model's built-in chat template.\n")

    # ── Build example ────────────────────────────────────────────────────────
    example = build_example(args.task_type, args.category)
    example = add_eos(example, tokenizer.eos_token)

    print("── Raw prompt (first 300 chars) ──────────────────────────────────")
    user_content = example["prompt"][0]["content"]
    print(user_content)

    print("── Raw completion (first 300 chars) ──────────────────────────────")
    asst_content = example["completion"][0]["content"]
    print(asst_content)

    # ── Tokenize ─────────────────────────────────────────────────────────────
    import warnings
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        tokenized = tokenize(
            example,
            processing_class=tokenizer,
            dataset_text_field="text",
            assistant_only_loss=False,
        )

    if caught:
        for w in caught:
            print(color(f"WARNING: {w.message}", 33))
        print()

    input_ids = tokenized["input_ids"]
    mask = tokenized["completion_mask"]

    print(f"── Token counts ──────────────────────────────────────────────────")
    n_prompt = sum(1 for m in mask if m == 0)
    n_compl = sum(1 for m in mask if m == 1)
    print(f"  Total tokens   : {len(input_ids)}")
    print(f"  Prompt tokens  : {n_prompt}  (loss=0)")
    print(f"  Completion tokens: {n_compl}  (loss=1)\n")

    # ── Decode and display with colour ───────────────────────────────────────
    print("── Token-by-token (green = completion / red = prompt, first 60) ──")
    for i, (tok_id, m) in enumerate(zip(input_ids[:60], mask[:60])):
        tok = tokenizer.decode([tok_id])
        tok_repr = repr(tok)
        if m == 1:
            print(color(tok_repr, 32), end=" ")
        else:
            print(color(tok_repr, 31), end=" ")
        if (i + 1) % 8 == 0:
            print()
    print("\n")

    # ── Decode full sections ──────────────────────────────────────────────────
    prompt_ids = [t for t, m in zip(input_ids, mask) if m == 0]
    compl_ids  = [t for t, m in zip(input_ids, mask) if m == 1]

    print("── Decoded prompt ────────────────────────────────────────────────")
    print(tokenizer.decode(prompt_ids))
    print()
    print("── Decoded completion ────────────────────────────────────────────")
    print(tokenizer.decode(compl_ids))
    print()

    # ── Boundary check ────────────────────────────────────────────────────────
    print("── Boundary token (last prompt → first completion) ───────────────")
    if n_prompt > 0 and n_compl > 0:
        last_prompt = tokenizer.decode([input_ids[n_prompt - 1]])
        first_compl = tokenizer.decode([input_ids[n_prompt]])
        print(f"  last prompt token : {repr(last_prompt)} (id={input_ids[n_prompt - 1]})")
        print(f"  first compl token : {repr(first_compl)} (id={input_ids[n_prompt]})")
    print()

    # ── Chat template preview (inference prompt) ─────────────────────────────
    print("── Inference prompt preview (add_generation_prompt=True) ─────────")
    chat_template_kwargs = {"enable_thinking": False} if "qwen3" in args.model_id.lower() else {}
    inference_prompt = tokenizer.apply_chat_template(
        example["prompt"],
        tokenize=False,
        add_generation_prompt=True,
        add_special_tokens=False,
        **chat_template_kwargs,
    ) + "Let's verify step by step:"
    print(inference_prompt)
    print()

    print(color("Sanity check complete.", 32))


if __name__ == "__main__":
    main()
