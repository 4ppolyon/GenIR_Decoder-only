# eval_full.py

import os
import csv
import time
from tqdm import tqdm
from llama3.generation_modified_Beam import *

# === PARAMÈTRES ===
QUERIES_FILE = "MS/data/queries_short.dev.tsv"
QRELS_FILE = "MS/data/qrels.dev.tsv"
OUTPUT_FILE = "MS/eval/full/generated_responses.tsv"
MODEL_PATH = "Llama-3.2-3B-Instruct/original"
TRIE_PATH = "MS_titres_vllm_corrected"
MODE = "chat"  # ou "text"
NB_TITRES = 10

os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

def load_qrels(qrels_file):
    qrels_qids = set()
    with open(qrels_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 3:
                qid = row[0]
                qrels_qids.add(qid)
    return qrels_qids

def load_queries(queries_file, qid_filter=None):
    queries = []
    with open(queries_file, newline='', encoding='utf-8') as f:
        reader = csv.reader(f, delimiter='\t')
        for row in reader:
            if len(row) >= 2:
                qid, text = row[0], row[1]
                if qid_filter is None or qid in qid_filter:
                    queries.append((qid, text))
    return queries

def Take_model(name: str, trie: str):
    return Llama.build(ckpt_dir=name, trie_path=trie)

def Send_model(prompt: str, model: Llama, mode: str = "text", nb_titres: int = 10):
    if mode.lower() == "chat":
        formated_prompt = [{"role": "user", "content": prompt.strip()}]
        res = model.chat_completion(dialog=formated_prompt, beam_width=2, nb_titres=nb_titres)
    elif mode.lower() == "text":
        formated_prompt = prompt.strip() + ":\n"
        res = model.text_completion(prompt=formated_prompt, beam_width=6, nb_titres=nb_titres)
    else:
        raise ValueError("Mode must be 'chat' or 'text'.")
    return res

def generate_predictions(model, queries, output_file):
    with open(output_file, "w", encoding="utf-8") as f_out:
        start = time.time()
        for qid, query_text in tqdm(queries, desc="Generating responses"):
            try:
                response = Send_model(query_text, model, mode=MODE, nb_titres=NB_TITRES)
                passage_ids = [doc_id for (_1, _2, doc_id) in response]
                f_out.write(f"{str(qid)}\t" + "\t".join(map(str, passage_ids)) + "\n")
            except Exception as e:
                print(f"Erreur sur la requête {qid}: {e}")
        print(f"\n⏱ Temps total de génération: {time.time() - start:.2f} secondes")

if __name__ == "__main__":
    print("📥 Chargement des qrels...")
    qrels_qids = load_qrels(QRELS_FILE)

    print("📥 Chargement des requêtes...")
    queries = load_queries(QUERIES_FILE, qid_filter=qrels_qids)
    print(f"🔎 {len(queries)} requêtes chargées.")

    print("🚀 Chargement du modèle...")
    model = Take_model(MODEL_PATH, TRIE_PATH)

    print(f"🧠 Début de la génération des réponses...")
    generate_predictions(model, queries, OUTPUT_FILE)

    print(f"✅ Terminé. Résultats disponibles dans : {OUTPUT_FILE}")
