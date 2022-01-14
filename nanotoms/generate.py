from transformers import GPT2Tokenizer, GPTNeoForCausalLM


def get_model(path: str) -> GPTNeoForCausalLM:
    return GPTNeoForCausalLM.from_pretrained(path)


def get_tokenizer(path: str) -> GPT2Tokenizer:
    return GPT2Tokenizer.from_pretrained(path)


def generate(
    model: GPTNeoForCausalLM,
    tokenizer: GPT2Tokenizer,
    prompt: str,
    kwargs: dict = dict(),
) -> str:
    input_ids = tokenizer(prompt, return_tensors="pt").input_ids

    if "max_length" in kwargs:
        kwargs["max_length"] += len(input_ids[0])

    tokens = model.generate(input_ids, **kwargs)

    return tokenizer.batch_decode(tokens)[0]
