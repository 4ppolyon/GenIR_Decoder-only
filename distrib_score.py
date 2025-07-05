import csv
from collections import Counter
import matplotlib.pyplot as plt


input_file = "per_query_scores.tsv"


ndcg_bins = Counter()
mrr_bins = Counter()


with open(input_file, newline='', encoding='utf-8') as f:
    reader = csv.DictReader(f, delimiter='\t')
    for row in reader:
        try:
            ndcg = float(row[f"nDCG@10"])
            mrr = float(row["MRR"])
        except ValueError:
            continue

        ndcg_bin = round(ndcg, 2)
        mrr_bin = round(mrr, 2)

        ndcg_bins[ndcg_bin] += 1
        mrr_bins[mrr_bin] += 1


print("Distribution des nDCG@10 :")
for bin_value in sorted(ndcg_bins):
    print(f"{bin_value:.2f} : {ndcg_bins[bin_value]}")

print("\nDistribution des MRR :")
for bin_value in sorted(mrr_bins):
    print(f"{bin_value:.2f} : {mrr_bins[bin_value]}")


plt.figure(figsize=(12, 5))

# nDCG
plt.subplot(1, 2, 1)
plt.bar(ndcg_bins.keys(), ndcg_bins.values(), width=0.008, align='center')
plt.title("nDCG@10 Distribution")
plt.xlabel("nDCG@10")
plt.ylabel("Number of Queries")
plt.xticks([round(i, 2) for i in list(ndcg_bins.keys())], rotation=90)

# MRR
plt.subplot(1, 2, 2)
plt.bar(mrr_bins.keys(), mrr_bins.values(), width=0.008, align='center', color='orange')
plt.title("MRR Distribution")
plt.xlabel("MRR")
plt.ylabel("Number of Queries")
plt.xticks([round(i, 2) for i in list(mrr_bins.keys())], rotation=90)

plt.tight_layout()
plt.show()
