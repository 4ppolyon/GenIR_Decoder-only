import pandas as pd
import json
from tqdm import tqdm

dir_path = './MS/data/'
TITLES_PATH = dir_path + "full/corrected/MS_titres_vllm_corrected.tsv"

queries = pd.read_csv(
    dir_path + "queries.train.tsv",
    sep="\t", names=["qid", "query_text"],
    dtype={"qid": int}
)
qrels = pd.read_csv(
    dir_path + "qrels.train.tsv",
    sep="\t", names=["qid", "pid"],
    usecols=[0, 2],
    dtype={"qid": int, "pid": int}
)
titles = pd.read_csv(
    TITLES_PATH,
    sep="\t", names=["pid", "titre"],
    usecols=[0, 1],
    dtype={"pid": int}
)


queries["query_text"] = queries["query_text"].astype(str).str.strip()
titles["titre"] = titles["titre"].astype(str).str.strip()


qid_to_query = dict(zip(queries.qid, queries.query_text))
pid_to_title = dict(zip(titles.pid, titles.titre))


print(f"Nombre de requêtes : {len(qid_to_query)}")
print(f"Nombre de titres   : {len(pid_to_title)}")


examples = []
for _, row in tqdm(qrels.iterrows(), total=qrels.shape[0], desc="Génération d'exemples"):
    query = qid_to_query.get(row.qid)
    title = pid_to_title.get(row.pid)
    if query and title:
        examples.append({
            "input": query,
            "output": title
        })

print(f"{len(examples)} exemples générés.")


output_file = dir_path + "llama3_train_dataset_titlegen.jsonl"
with open(output_file, "w", encoding="utf-8") as f:
    for ex in tqdm(examples, desc="Sauvegarde des exemples", total=len(examples)):
        f.write(json.dumps(ex, ensure_ascii=False) + "\n")

print(f"Fichier sauvegardé : {output_file}")
