import os
import torch
from datasets import load_dataset
from transformers import ( AutoModelForCausalLM,
    BitsAndBytesConfig, TrainingArguments, Trainer
)
from peft import LoraConfig, get_peft_model

from transformers import AutoTokenizer

DEBUG = False

model_id = "./Llama-3.2-3B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token  # nécessaire pour DataCollator


# bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
bnb = BitsAndBytesConfig(load_in_8bit=True, bnb_8bit_use_double_quant=True)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    device_map="auto",
    quantization_config=bnb,
    trust_remote_code=True
)
model.config.use_cache = False

dataset = load_dataset("json", data_files="./MS/data/llama3_train_dataset_titlegen.jsonl", split="train")

# dataset = dataset.select(range(len(dataset) // 3))  # réduction à 1/3

if DEBUG:
    dataset = dataset.select(range(5))


def format_chat_template(row):
    prompt = f"<|start_header_id|>user<|end_header_id|>\n{row['input']}<|eot_id|>\n" \
             f"<|start_header_id|>assistant<|end_header_id|>\n{row['output']}<|eot_id|>"
    row["input_ids"] = tokenizer(prompt, return_tensors="pt", truncation=True).input_ids[0]
    return row

dataset = dataset.map(
    format_chat_template,
    num_proc=4,
    remove_columns=["input", "output"],
)

if DEBUG:
    from llama3.tokenizer import Tokenizer

    TOKENIZER_PATH = "../Llama-3.2-3B-Instruct/original/tokenizer.model"
    raw_tokenizer = Tokenizer(TOKENIZER_PATH)

    print("Dataset tokenisé (exemple):")
    taille_dataset = len(dataset["input_ids"])
    print("Taille du dataset:", taille_dataset)
    for i in range(taille_dataset):
        print(f"Exemple {i+1}:")
        print("Exemples de tokens d'entrée:", dataset["input_ids"][i])
        print("Exemples de tokens d'entrée:", raw_tokenizer.decode(dataset["input_ids"][i]))

from transformers import DataCollatorForLanguageModeling

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)

lora_config = LoraConfig(
    r=16, lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05, bias="none",
    task_type="CAUSAL_LM"
)
model = get_peft_model(model, lora_config)

training_args = TrainingArguments(
    output_dir="llama3_lora_finetuned",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    fp16=True,
    logging_steps=10,
    save_steps=500,
    save_total_limit=2,
    report_to="tensorboard",
    logging_dir="./logs_finetuning"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator
)

if not DEBUG :
    trainer.train()
    model.save_pretrained("./llama3_lora_finetuned_in")
    print("✅ Fine-tuning terminé (LoRA)")