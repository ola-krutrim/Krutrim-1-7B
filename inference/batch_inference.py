import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "krutrim-ai-labs/Krutrim-1-instruct" 
device = "cuda" if torch.cuda.is_available else "cpu"
# Load model and tokenizer
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.bfloat16, trust_remote_code=True)
model = model.to(device)
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)


chat_template ="{% for message in messages %}{% if message['role'] == 'system' %}{{ '<|SYSTEM|> ' + message['content'] + '\n' }}{% elif message['role'] == 'user' %}{{ '<|USER|> ' + message['content'] + '\n' }}{% elif message['role'] == 'assistant' %}{% if not loop.last %}{{ '<|RESPONSE|>\n'  + message['content'] + eos_token + '\n' }}{% else %}{{ '<|RESPONSE|>\n'  + message['content'] + eos_token }}{% endif %}{% endif %}{% if loop.last and add_generation_prompt %}{{ '<|RESPONSE|>\n' }}{% endif %}{% endfor %}"
tokenizer.chat_template = chat_template

prompt_dict = [
    [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "Who are you?"}
    ],
    [
        {"role": "system", "content": "You are an AI assistant."},
        {"role": "user", "content": "What's up?"}
    ]
]

prompts = tokenizer.apply_chat_template(prompt_dict, add_generation_prompt=True, tokenize=False)
inputs = tokenizer(prompts, return_tensors='pt').to(device)
inputs.pop("token_type_ids", None)

print(prompts)
# Generate response
outputs = model.generate(
    **inputs,
    max_length=100
)

responses = tokenizer.batch_decode(outputs, skip_special_tokens=True)
responses = [r.split(p)[1] for r, p in zip(responses, prompts)]

print(responses)
