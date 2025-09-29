import warnings

from trl import is_conversational

def tokenize(example, processing_class, dataset_text_field, assistant_only_loss):
    # tokenize function from SFTTrainer._prepare_dataset()
    if "prompt" in example:  # prompt-completion case
        if is_conversational(example):
            prompt_ids = processing_class.apply_chat_template(
                example["prompt"],
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
            prompt_completion_ids = processing_class.apply_chat_template(
                example["prompt"] + example["completion"],
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
        else:
            prompt_ids = processing_class(text=example["prompt"]).input_ids
            prompt_completion_ids = processing_class(
                text=example["prompt"] + example["completion"]
            ).input_ids

        # Check if the tokenized prompt starts with the tokenized prompt+completion
        if not prompt_completion_ids[: len(prompt_ids)] == prompt_ids:
            warnings.warn(
                "Mismatch between tokenized prompt and the start of tokenized prompt+completion. "
                "This may be due to unexpected tokenizer behavior, whitespace issues, or special "
                "token handling. Verify that the tokenizer is processing text consistently."
            )

        # Create a completion mask
        completion_mask = [0] * len(prompt_ids) + [1] * \
            (len(prompt_completion_ids) - len(prompt_ids))
        processed = {"input_ids": prompt_completion_ids,
                     "completion_mask": completion_mask}
    else:  # language modeling case
        if is_conversational(example):
            processed = processing_class.apply_chat_template(
                example["messages"],
                return_dict=True,
                return_assistant_tokens_mask=assistant_only_loss,
                tools=example.get("tools"),
                **example.get("chat_template_kwargs", {}),
            )
            if "assistant_masks" in processed and 1 not in processed["assistant_masks"]:
                raise RuntimeError(
                    "You're using `assistant_only_loss=True`, but at least one example has no "
                    "assistant tokens. This usually means the tokenizer's chat template doesn't "
                    "generate assistant masks — it may be missing the `{% generation %}` keyword. Please "
                    "check the template and ensure it's correctly configured to support assistant "
                    "masking."
                )
            processed = {k: processed[k] for k in (
                "input_ids", "assistant_masks") if k in processed}
        else:
            processed = {"input_ids": processing_class(
                text=example[dataset_text_field]).input_ids}
    return processed

def add_eos(example, eos_token):
    # language modeling case
    if "text" in example and not example["text"].endswith(eos_token):
        example["text"] = example["text"] + eos_token
    elif "completion" in example and not example["completion"].endswith(eos_token):
        example["completion"] = example["completion"] + eos_token
    return example

def preprocess_dataset(dataset, tokenizer):
     # add eos token
    map_kwargs = {}
    map_kwargs["desc"] = f"Adding EOS to dataset"
    dataset = dataset.map(add_eos,
                fn_kwargs={"eos_token": tokenizer.eos_token},
                remove_columns=None,
                **map_kwargs,
                )
    # tokenize dataset
    map_kwargs["desc"] = "Tokenize train dataset"
    dataset = dataset.map(
        tokenize,
        fn_kwargs={
            "processing_class": tokenizer,
            "dataset_text_field": "text",
            "assistant_only_loss": False,
        },
        **map_kwargs
    )
    return dataset