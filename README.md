# Krutrim-1

## Model Overview
Krutrim Large Language Model (LLM) is a 2 trillion token multilingual foundation model designed to serve Indian demographic needs through equitable representation of the country's array of native tongues. Training data incorporates the largest known Indic language dataset, mitigating associated data scarcity obstacles that encumber model parity across dialects. Evaluations demonstrate Krutrim’s strong performance on Indic language benchmarks, surpassing or at par with state-of-the-art models despite being significantly smaller in training flops. Krutrim LLM also matches or exceeds standards set on English benchmarks by models trained on comparable flops (e.g. vs LLAMA-2 on 10 out of 16 tasks with average score of 0.57 vs 0.55 of LLAMA-2), evidencing flexible multilingual fluency. Through intentional design choices that redress endemic data imbalances, Krutrim LLM signifies meaningful progress in the pursuit of ethical, globally representative AI foundation models.

## Key Features
- 7B parameter dense transformer model comparable similarly sized LLama-2 model;
- Natively multilingual delivering best-in-class performance for a 7B mdoel on Indic benchmarks;
- Exceeds performance of similar sized models on multilingual Indic generation tasks including creative writing, summarization, and translation;
- Available in both pre-trained and instruction-tuned versions

## Model Developer
- OLA Krutrim Team

## Model Dates
- Krutrim-1 was trained between Oct 2023 and Nov 2023.

## Release History

| Model Name | Release Date |Release Note | Reference|
|------------|-------------|-------------|-------------|
| Krutrim-1-Base   | 2024-01-31  | Trained from scratch | [Here](https://huggingface.co/krutrim-ai-labs/Krutrim-1-base)
| Krutrim-1-Instruct  | 2024-01-31 | SFT on Krutrim-1-Base |[Here](https://huggingface.co/krutrim-ai-labs/Krutrim-1-instruct)


## Data Freshness
- The dataset includes information up to April 2023.

## Model Architecture
- Layers: 32
- Max Sequence Length: 4096
- Hidden Dimension: 4608
- Head Dimension: 96
- Number of Heads: 48
- Number of KV-Heads: 8 (GQA)
- Vocabulary Size: 70400
- Architecture Type: Transformer Decoder (Auto-regressive Language Model)

## Evaluation Results

### English Comparison between Krutrim-1 and Llama2Chat (Benchmarks run on `llm_foundry`)

| Task               | Llama2Chat-7B | Krutrim-1-7B |
|--------------------|--------------|------------|
| arc               | 0.517        | **0.557**  |
| bigbench          | **0.359**    | 0.330      |
| boolq            | **0.803**    | 0.843      |
| copa             | 0.78         | **0.82**   |
| hellaswag        | **0.754**    | 0.740      |
| jeopardy         | 0.306        | **0.286**  |
| lambadaopenai    | **0.695**    | 0.682      |
| logiqa           | 0.332        | **0.3195** |
| mathqa           | **0.436**    | 0.440      |
| mmlu             | 0.472        | **0.495**  |
| openbookqa       | 0.44         | **0.464**  |
| piqa             | **0.7601**   | 0.7726     |
| simplearithmetic | 0.160        | **0.077**  |
| squad            | 0.3565       | **0.369**  |
| winograd         | **0.8645**   | 0.828      |
| winogrande       | 0.681        | **0.697**  |
| **average**      | **0.54**     | **0.54**   |


### Benchmarks

| Model            | bn   | gu   | hi   | kn   | ml   | mr   | ta   | te   |
|------------------|------|------|------|------|------|------|------|------|
| **IndicCOPA**    |      |      |      |      |      |      |      |      |
| Krutrim-1-7B        | 0.89 | 0.83 | 0.86 | 0.88 | 0.88 | 0.87 | 0.89 | 0.89 |
| GPT-3.5          | 0.77 | 0.73 | 0.77 | 0.74 | 0.75 | 0.70 | 0.72 | 0.75 |
| Airawata         | -    | -    | 0.74 | -    | -    | -    | -    | -    |
| Kan-LLaMA        | -    | -    | -    | 0.74 | -    | -    | -    | -    |
| Tam-LLaMA        | -    | -    | -    | -    | -    | -    | 0.77 | -    |
| **IndicQA**      |      |      |      |      |      |      |      |      |
| Krutrim-1-7B        | 0.65 | 0.64 | 0.64 | 0.60 | 0.66 | 0.58 | 0.75 | 0.83 |
| Airawata         | -    | -    | 0.62 | -    | -    | -    | -    | -    |
| Kan-LLaMA        | -    | -    | -    | 0.52 | -    | -    | -    | -    |
| Tam-LLaMA        | -    | -    | -    | -    | -    | -    | 0.35 | -    |
| **IndicSentiment**|      |      |      |      |      |      |      |      |
| Krutrim-1-7B       | 0.95 | 0.96 | 0.96 | 0.95 | 0.96 | 0.97 | 0.94 | 0.95 |
| GPT-3.5          | 0.50 | 0.81 | 0.96 | 0.60 | 0.75 | 0.88 | 0.51 | 0.53 |
| Airawata         | -    | -    | 0.84 | -    | -    | -    | -    | -    |
| Kan-LLaMA        | -    | -    | -    | 0.85 | -    | -    | -    | -    |
| Tam-LLaMA        | -    | -    | -    | -    | -    | -    | 0.78 | -    |
| **IndicTranslation**|   |      |      |      |      |      |      |      |
| Krutrim-1-7B        | 0.88 | 0.89 | 0.95 | 0.88 | 0.89 | 0.92 | -    | 0.88 |
| Airawata         | -    | -    | 0.94 | -    | -    | -    | -    | -    |
| Kan-LLaMA        | -    | -    | -    | 0.59 | -    | -    | -    | -    |
| **IndicXParaphrase**|  |      |      |      |      |      |      |      |
| Krutrim-1-7B       | 0.91 | -    | 0.97 | 0.82 | 0.90 | 0.94 | -    | 0.61 |
| Airawata         | -    | -    | 0.60 | -    | -    | -    | -    | -    |
| Kan-LLaMA        | -    | -    | -    | 0.59 | -    | -    | -    | -    |

## Usage

To run this model, do this:
```
git clone https://github.com/ola-krutrim/Krutrim-1-7B.git
cd Krutrim-1-7B
pip install -r requirements.txt
```

To use the base model, you can load it with `AutoModelForCausalLM` as follows:
```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "krutrim-ai-labs/Krutrim-1-base"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

prompt = "Hello"

inputs = tokenizer(prompt, return_tensors='pt')
inputs.pop("token_type_ids", None)

# Generate response
outputs = model.generate(
    **inputs,
    max_length=5
)

response = tokenizer.decode(outputs[0])
print(response)
```

`Krutrim-1-instruct` model requires application of custom chat template.

```python
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "krutrim-ai-labs/Krutrim-1-instruct"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model_id)

# Apply Chat Template
chat_template ="{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|SYSTEM|> ' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|USER|> ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|RESPONSE|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|RESPONSE|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|RESPONSE|>\n' }}{% endif %}{% endfor %}"
tokenizer.chat_template = chat_template

prompt_dict = [

    {"role": "system", "content": "You are an AI assistant."},
    {"role": "user", "content": "Who are you?"}
]

prompts = tokenizer.apply_chat_template(prompt_dict, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(prompts, return_tensors='pt').to(device)
inputs.pop("token_type_ids", None)

# Generate response
outputs = model.generate(
    **inputs,
    max_length=100
)

response = tokenizer.decode(outputs[0])
print(response)
```

For batch inference, do:
```
python inference/batch_inference.py 
```


## Limitations
The model was trained on a dataset that includes content from the internet, which may contain toxic language, biases, and unsafe content. As a result, the model may:
- Amplify biases present in the training data
- Generate toxic responses, especially when prompted with toxic inputs
- Provide inaccurate, incomplete, or redundant answers
- Generate responses in languages inconsistent with the prompt

## License
TBD

## Ethical Considerations
- The model may produce biased or offensive outputs based on its training data.
- Users should apply human oversight when using the model for decision-making in sensitive areas.
- While safeguards have been implemented, the model may still generate socially undesirable text in certain contexts.

## Contact
TBD

