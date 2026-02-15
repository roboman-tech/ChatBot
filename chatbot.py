import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

model_path = "D:/Source/models/llama2-7b-chat"

tokenizer = AutoTokenizer.from_pretrained(
    model_path,
    local_files_only=True,
    use_fast=False
)

tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    local_files_only=True,
    torch_dtype=torch.float16,
    device_map="auto",
    low_cpu_mem_usage=True
)

def format_prompt(user_input):
    return f"<s>[INST] {user_input.strip()} [/INST]"

def chat(user_input: str) -> str:
    prompt = f"<s>[INST] {user_input.strip()} [/INST]"

    inputs = tokenizer(
        prompt,
        return_tensors="pt"
    )

    input_ids = inputs["input_ids"].to(model.device)
    attention_mask = inputs["attention_mask"].to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=200,
            temperature=0.7,
            top_p=0.9,
            do_sample=True,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.eos_token_id
        )

    # 🔑 Slice off the prompt tokens
    generated_ids = outputs[0][input_ids.shape[-1]:]

    reply = tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True
    )

    reply = reply.replace("▁", " ").strip()   # ✅ cosmetic fix
    reply = ' '.join(reply.split())
    return reply

#print("LLaMA-2-7B-Chat ready (no warnings)")
#while True:
#    user_input = input("You: ")
#    if user_input.lower() in ("exit", "quit"):
#        break
#    print("AI:", chat(user_input))