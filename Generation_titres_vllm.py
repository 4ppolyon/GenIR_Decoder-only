import random
from vllm import LLM, SamplingParams
import csv
import time
from tqdm import tqdm
from transformers import AutoTokenizer

SYSTEM_PROMPT = "Summarize in a nominal phrase with no repetition, at most 10 words and at least 6 words the following paragraph"
# SYSTEM_PROMPT = "Summarize in a nominal phrase with no repetition, at most 10 words and at least 6 words the following paragraph :"
# SYSTEM_PROMPT = "Summarize in a nominal phrase containing only lowercase letters, no repetition, at most 10 words and at least 6 words the following paragraph:"
# SYSTEM_PROMPT = "Summarize in a nominal phrase with no repetition, at most 6 words and at least 2 words the following paragraph"

def load_model(model_id="./Llama-3.2-3B-Instruct"):
    return LLM(model=model_id)

def generate_titles(model, passages, model_name="./Llama-3.2-3B-Instruct"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    prompts = []
    for passage in passages:
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": passage}
            # {"role": "user", "content": "\"" + passage + "\""}
        ]
        prompt = tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        prompts.append(prompt)

    sampling_params = SamplingParams(temperature=0, top_p=0.9, max_tokens=150)
    outputs = model.generate(prompts, sampling_params)

    results = []
    for output in outputs:
        results.append(output.outputs[0].text.strip())
    return results

def read_tsv(file_path):
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        return [(row[0], row[1]) for row in tqdm(reader)]

if __name__ == "__main__":
    batch_size = 250
    data = read_tsv("MS/data/MS_collection.tsv")

    # random.shuffle(data)

    all_ids, all_passages = zip(*data)
    nb_passages = len(all_passages)

    llm = load_model()
    titres = []
    start = time.time()

    for i in tqdm(range(0, nb_passages, batch_size)):
        passages = all_passages[i:i+batch_size]
        ids = all_ids[i:i+batch_size]
        responses = generate_titles(llm, passages)
        for id_, response in zip(ids, responses):
            titres.append([id_, response])

    if "most 6 words" in SYSTEM_PROMPT :
        with open("MS/data/full/MS_short_titres_vllm.tsv", "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter='\t')
            for id_, title in titres:
                writer.writerow([id_, title])
    else :
        with open("MS/data/full/MS_titres_vllm.tsv", "w", newline='', encoding='utf-8') as file:
            writer = csv.writer(file, delimiter='\t')
            for id_, title in titres:
                writer.writerow([id_, title])

    print(f"Titres générés et enregistrés en {time.time() - start:.2f} secondes")
