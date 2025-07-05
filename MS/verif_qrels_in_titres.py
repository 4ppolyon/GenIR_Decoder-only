import csv
from tqdm import tqdm

qrels_file = "MS/data/qrels.dev.tsv"
titres_file = "MS/data/full/corrected/MS_titres_vllm_corrected.tsv"


with open(titres_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    titre_ids = {row[0] for row in reader}


missing_ids = set()
with open(qrels_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    next(reader)
    for row in tqdm(reader):
        if len(row) < 3:
            print(f"Ligne incomplète dans {qrels_file}: {row}")
            continue
        doc_id = row[2]
        if doc_id not in titre_ids:
            missing_ids.add(doc_id)


if missing_ids:
    print(f"{len(missing_ids)} identifiants manquants dans {titres_file} :")
    for doc_id in sorted(missing_ids):
        print(doc_id)
else:
    print(f"Tous les identifiants de qrels.train.tsv sont présents dans {titres_file}.")
