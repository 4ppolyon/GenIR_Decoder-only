from transformers import AutoModelForCausalLM
from peft import PeftModel
import torch

usage = "save"
# usage = "use"

lora_model = "llama3_lora_finetuned_in"
# lora_model = "llama3_lora_finetuned_in_out"

base_model = "./Llama-3.2-3B-Instruct"

# Charger le modèle de base et le modèle LoRA fine-tuné
base_model = AutoModelForCausalLM.from_pretrained(base_model, torch_dtype=torch.float16)
model = PeftModel.from_pretrained(base_model, lora_model)

if usage.lower() == "save":
    # model devient un modèle fusionné pour économiser de la mémoire
    model = model.merge_and_unload()
    # Sauvegarder le modèle fusionné
    state_dict = model.state_dict()
    # Enregistrer le modèle fusionné
    torch.save(state_dict, lora_model + "/llama3_finetuned_fused.pth")
    print("Model saved as 'llama3_finetuned_fused.pth'.")

elif usage.lower() == "use":
    # Tokenizer et formatter pour le chat
    from llama3.tokenizer import Tokenizer, ChatFormat
    raw_tokenizer = Tokenizer("../Llama-3.2-3B-Instruct/original/tokenizer.model")
    formatter = ChatFormat(raw_tokenizer)
    dialog = [{"role": "user", "content": ")what was the immediate impact of the success of the manhattan project?"}]
    # dialog = [{"role": "user", "content": "Everything about cancer"}]
    prompt = formatter.encode_dialog_prompt(dialog)
    decoded = raw_tokenizer.decode(prompt)
    input_ids = torch.tensor([prompt]).to(model.device)

    # Génération
    outputs = model.generate(
        input_ids=input_ids,
        max_new_tokens=64,
        temperature=1.0
    )

    generated_ids = outputs[0][input_ids.shape[-1]:]
    generated_text = raw_tokenizer.decode(generated_ids.tolist())

    print("- Prompt:", decoded, "\n\n", "- Response:", generated_text)