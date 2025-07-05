import csv
from collections import defaultdict
import numpy as np
from sklearn.metrics import ndcg_score


qrels_file = "MS/data/qrels.dev.tsv"
# responses_file = "MS/eval/generated_responses_ft_chat.tsv"

# responses_file = "MS/eval/generated_responses_zs2.tsv" # 0.0545 0.0446
# responses_file = "MS/eval/generated_responses_zs6.tsv" # 0.0557 0.0495

# responses_file = "MS/eval/generated_responses_zs1_6k.tsv" # 0.0261
responses_file = "MS/eval/generated_responses_zs6_6k.tsv" # 0.0521 0.0465

k = 10  # profondeur pour nDCG@k


qrels = defaultdict(set)
with open(qrels_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) >= 3:
            qid = row[0]
            docid = row[2]
            qrels[qid].add(docid)


preds = {}
with open(responses_file, newline='', encoding='utf-8') as f:
    reader = csv.reader(f, delimiter='\t')
    for row in reader:
        if len(row) >= 2:
            qid = row[0]
            preds[qid] = row[1:]


if "zs1" not in responses_file:
    ndcg_scores = []
mrr_scores = []
per_query_scores = {}

for qid in qrels:
    rel_docs = qrels[qid]
    predicted = preds.get(qid, [])
    if not predicted:
        continue


    y_score = [1.0 / (i + 1) for i in range(len(predicted))]
    y_true = [1 if docid in rel_docs else 0 for docid in predicted]


    y_true_np = np.asarray([y_true])
    y_score_np = np.asarray([y_score])


    if "zs1" not in responses_file:
        ndcg = ndcg_score(y_true_np, y_score_np, k=k)
        ndcg_scores.append(ndcg)


    try:
        rank = next(i + 1 for i, docid in enumerate(predicted) if docid in rel_docs)
        mrr = 1.0 / rank
    except StopIteration:
        mrr = 0.0
    mrr_scores.append(mrr)

    if "zs1" not in responses_file:

        per_query_scores[qid] = {
            "nDCG": ndcg,
            "MRR": mrr,
            "Rel_Found": [docid for docid in predicted if docid in rel_docs]
        }
    else:
        per_query_scores[qid] = {
            "MRR": mrr,
            "Rel_Found": [docid for docid in predicted if docid in rel_docs]
        }


with open("per_query_scores.tsv", mode="w", encoding="utf-8", newline='') as f_out:
    writer = csv.writer(f_out, delimiter='\t')
    if "zs1" in responses_file:
        writer.writerow(["QID", "MRR", "Rel_Found"])
    else:
        writer.writerow(["QID", f"nDCG@{k}", "MRR", "Rel_Found"])
    for qid, scores in per_query_scores.items():
        rel_found_str = ";".join(scores["Rel_Found"])
        if "zs1" in responses_file:
            writer.writerow([qid, f"{scores['MRR']:.4f}", rel_found_str])
        else:
            writer.writerow([qid, f"{scores['nDCG']:.4f}", f"{scores['MRR']:.4f}", rel_found_str])


if "zs1" not in responses_file:
    average_ndcg = np.mean(ndcg_scores)
average_mrr = np.mean(mrr_scores)

if "zs1" not in responses_file:
    print(f"Average nDCG@{k}: {average_ndcg:.4f}")
print(f"MRR: {average_mrr:.4f}")
