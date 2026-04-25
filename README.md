# CPT for PINN Papers on Gemma 3 4B

## Overview

`google/gemma-3-4b-pt` is a 4B base model suitable for custom post-training. In this project, we apply continuous pretraining (CPT) on a metadata-only [dataset](https://huggingface.co/datasets/pymlex/pinn-arxiv) of Physics-Informed Neural Networks (PINNs) papers from arXiv. The goal is to make the model respond more precisely to questions from this domain and improve its familiarity with PINN-related terminology, topics, and paper metadata.

## Dataset

The dataset contains 2,598 metadata records for PINN-related papers. Each record was flattened into a text block with the following fields:

- Title
- Authors
- Published
- Updated
- Summary
- Categories
- Primary category

Each sample is no longer than 2300 symbols so we set the `max_seq_length` value to 1024: 

![output_8_0](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/kKW0I1tGx207AGZeC37gW.png)

The dataset was split into train, validation, and test subsets with a `90/5/5` ratio.

## CPT

Continued pretraining was performed with RTX 5090, Ryzen 9 9950X, 64 GB RAM. The whole process took only 10 minutes. We used a supervised fine-tuning infrastructure from `trl` and a causal language modelling objective with the following settings:

- max sequence length: `1024`
- batch size: `6`
- gradient accumulation: `4`
- epochs: `2`
- learning rate: `1e-4`
- scheduler: cosine
- optimiser: `adamw_torch`

The best checkpoint is selected by validation loss.

## Loss curves

The training and validation losses show a general downward trend with visible oscillations, which is typical for small-domain CPT runs.

![output_20_0](https://cdn-uploads.huggingface.co/production/uploads/6957bafe54c6b170be4df9cb/nIV1CnrWio0MhW1wk_gDE.png)

## Inference

Use these two cells for inference:

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel

base_model_id = "google/gemma-3-4b-pt"
adapter_id = "pymlex/gemma3-4b-pinn-expert"

tokenizer = AutoTokenizer.from_pretrained(base_model_id)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

base_model = AutoModelForCausalLM.from_pretrained(
    base_model_id,
    device_map="auto",
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float16,
)

model = PeftModel.from_pretrained(base_model, adapter_id)
model.eval()
```

```python
def build_prompt(record):
    return (
        f"Title: {record.get('Title', '')}\n"
        f"Authors: {record.get('Authors', '')}\n"
        f"Published: {record.get('Published', '')}\n"
        f"Updated: {record.get('Updated', '')}\n"
        f"Summary: "
    )


def generate_continuation(model, tokenizer, prompt, max_new_tokens=220):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.inference_mode():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            repetition_penalty=1.1,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id,
        )

    prompt_len = inputs["input_ids"].shape[1]
    continuation_ids = outputs[0, prompt_len:]

    continuation = tokenizer.decode(continuation_ids, skip_special_tokens=True)
    return continuation


sample_record = {
    "Title": "fPINNs: Fractional Physics-Informed Neural Networks",
    "Authors": "Guofei Pang, Lu Lu, George Em Karniadakis",
    "Published": "2018-11-20T02:48:36Z",
    "Updated": "2018-11-20T02:48:36Z",
}

prompt = build_prompt(sample_record)
output = generate_continuation(model, tokenizer, prompt, max_new_tokens=400)
print("Prompt:")
print(prompt)
print("\nGenerated continuation:")
print(output)
```

## Perplexity evaluation

We evaluated both the base model and the tuned model on the test split with 130 samples. The results are:

| Model | Perplexity |
|---|---|
| Base Model | 9.200 |
| Tuned Model | 6.646 |

Perplexity improvement is about `28%`. This shows that CPT makes the model less surprised by PINN-domain text even on held-out examples.
