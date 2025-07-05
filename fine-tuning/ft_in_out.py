from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, Trainer
from peft import get_peft_model, LoraConfig, TaskType
from transformers import DataCollatorForSeq2Seq

from llama3.tokenizer import Tokenizer, ChatFormat

DEBUG = False
V_mask = False

TOKENIZER_PATH = "../Llama-3.2-3B-Instruct/original/tokenizer.model"
raw_tokenizer = Tokenizer(TOKENIZER_PATH)
formatter = ChatFormat(raw_tokenizer)

model_name = "./Llama-3.2-3B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(model_name, load_in_4bit=True)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM
)
model = get_peft_model(model, lora_config)

# Chargement dataset JSONL
dataset = load_dataset("json", data_files="./MS/data/llama3_train_dataset_titlegen.jsonl")

# dataset = dataset.select(range(len(dataset) // 3))  # réduction à 1/3

if DEBUG:
    dataset["train"] = dataset["train"].select(range(5))

def preprocess(example):
    dialog = [{"role": "user", "content": example["input"]}]
    input_ids = formatter.encode_dialog_prompt(dialog)

    target_ids = tokenizer(
        example["output"],
        max_length=64,
        truncation=True,
        padding=False
    )["input_ids"]

    if target_ids and target_ids[0] == tokenizer.bos_token_id:
        target_ids = target_ids[1:]

    if V_mask:
        full_input = input_ids + target_ids
        full_input = full_input[:256]
    else:
        full_input = input_ids[:256]
        full_input += [tokenizer.pad_token_id] * (256 - len(full_input))

    if V_mask:
        labels = [tokenizer.pad_token_id] * len(input_ids) + target_ids
        labels = labels[:256]
    else:
        labels = target_ids[:256]
        labels += [tokenizer.pad_token_id] * (256 - len(labels))

    if V_mask:
        pad_len = 256 - len(full_input)
        full_input += [tokenizer.pad_token_id] * pad_len
        labels += [tokenizer.pad_token_id] * pad_len

    attention_mask = [1 if token != tokenizer.pad_token_id else 0 for token in full_input]
    for i in range(len(attention_mask)):
        if attention_mask[i] == 0:
            attention_mask[i] = 1
            break


    if DEBUG:
        print("Input:", tokenizer.decode(full_input))
        print("Full input IDs:", full_input)
        print("_"* 80)
        print("Attention Mask:", attention_mask)
        print("_"* 80)
        print("Labels:", labels)
        print("Decoded Labels:", tokenizer.decode(labels))

    return {
        "input_ids": full_input,
        "attention_mask": attention_mask,
        "labels": labels
    }


# Tokenisation dataset sans batch (exemple par exemple)
tokenized_dataset = dataset.map(preprocess, batched=False, remove_columns=["input", "output"])

if not DEBUG :
    # Data collator pour pad batch sans retokenizer
    data_collator = DataCollatorForSeq2Seq(
        tokenizer,
        model=model,
        label_pad_token_id=tokenizer.pad_token_id,
        pad_to_multiple_of=8,
    )

    # TrainingArguments
    training_args = TrainingArguments(
        output_dir="llama3_lora_finetuned",
        per_device_train_batch_size=4,
        per_device_eval_batch_size=4,
        learning_rate=2e-4,
        num_train_epochs=1,
        logging_steps=50,
        save_steps=500,
        save_total_limit=2,
        logging_dir="./logs",
        fp16=True,
        report_to="none",
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_dataset["train"],
        data_collator=data_collator,
    )

    # Lancement du fine-tuning
    trainer.train()
    model.save_pretrained("./llama3_lora_finetuned_in_out")
    print("✅ Fine-tuning terminé (LoRA)")
