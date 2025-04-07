from unsloth import FastLanguageModel, is_bfloat16_supported, train_on_responses_only
from datasets import load_dataset, Dataset
from trl import SFTTrainer, apply_chat_template
from transformers import TrainingArguments, DataCollatorForSeq2Seq, TextStreamer
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
base_model_path = "unsloth/Llama-3.2-3B-Instruct"
adapter_path = "/content/fine_tuned_llama3_3b"
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModelForCausalLM

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name = base_model_path,
    device_map = "auto",
    dtype = torch.float16,
)

FastLanguageModel.for_inference(model)
def chat_with_model(prompt, max_new_tokens=512, temperature=0.7):
    inputs = tokenizer(
        [
            f"<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n{prompt}<|eot_id|>"
            "<|start_header_id|>assistant<|end_header_id|>\n\n"
        ],
        return_tensors="pt",
    ).to("cuda")

    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        pad_token_id=tokenizer.eos_token_id,
    )

    response = tokenizer.batch_decode(outputs, skip_special_tokens=True)[0]
    response = response.split("<|start_header_id|>assistant<|end_header_id|>")[-1].strip()
    return response
while True:
    user_input = input("\nBạn: ")
    if user_input.lower() in ["exit", "quit", "stop"]:
        break
    response = chat_with_model(user_input)
    print("\nModel:", response)
