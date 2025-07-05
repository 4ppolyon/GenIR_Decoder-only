import time
import pandas as pd
import csv
from statistics import median
import matplotlib.pyplot as plt
import seaborn as sns

# file_name = 'rd/MS_rd_titres'
# file_name = 'rd/MS_rd_titres_vllm'
# file_name = 'rd/MS_tk_rd_titres_vllm'

# file_name = '1000st/MS_1000st_titres'
# file_name = '1000st/MS_1000st_titres_vllm'

file_name = 'full/MS_titres_vllm'
# file_name = 'full/MS_short_titres_vllm'

# file_name = 'full/corrected/MS_titres_vllm_corrected'
# file_name = 'full/corrected/MS_short_titres_vllm_corrected'

base_path = './MS/data/'
output_base = './res/gen_titres/'

file_path = f'{base_path}{file_name}.tsv'
stat_file = f'{output_base}{file_name}_stats.tsv'
graph_file = f'{output_base}{file_name}_graph.png'
export_path = f'{output_base}{file_name}_titres.tsv'
export_path_pg = f'{output_base}{file_name}_problem_gen.tsv'
export_path_pi = f'{output_base}{file_name}_problem_inst.tsv'
export_path_l = f'{output_base}{file_name}_autres.tsv'

start = time.time()


df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1, 3, 4], names=[
    'passage_id', 'passage', 'size', 'category_refined'
])


if df.shape[1] < 4:
    raise ValueError(f"Le fichier semble incomplet {df.shape[1]} colonnes. Il faut d'abord exécuter 'add_data'.")


df = df.dropna(subset=['size', 'category_refined'])
df['size'] = df['size'].astype(int)


size_series = df['size']
avg_size = round(size_series.mean(),2)
var_size = round(size_series.var(),2)
median_size = median(size_series)
min_size = size_series.min()
max_size = size_series.max()
min_id = df.loc[size_series.idxmin(), 'passage_id']
max_id = df.loc[size_series.idxmax(), 'passage_id']


category_counts = df['category_refined'].value_counts()
total = len(df)


rows = [
    ["Total", total],
    ["Max", f"{max_size} (id: {max_id})"],
    ["Min", f"{min_size} (id: {min_id})"],
    ["Average", avg_size],
    ["Variance", var_size],
    ["Median", median_size]
]


for cat in ['ok', 'autre', 'problèmes de génération', 'problèmes instruction']:
    count = category_counts.get(cat, 0)
    perc = (count / total) * 100
    rows.append([f"NB {cat}", count])
    rows.append([f"PRC {cat}", f"{perc:.2f}%"])


unknown = df[~df['category_refined'].isin(['ok', 'autre', 'problèmes de génération', 'problèmes instruction'])]
if not unknown.empty:
    print(f"ALED: {len(unknown)} lignes avec catégories inconnues :")
    print(unknown[['passage_id', 'category_refined']])
    rows.append(["NB Aled", len(unknown)])


with open(stat_file, 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f, delimiter='\t')
    writer.writerow(["Metric", "Value"])
    writer.writerows(rows)


df[df['category_refined'] == 'ok'].to_csv(export_path, sep='\t', index=False)
df[df['category_refined'] == 'problèmes de génération'].to_csv(export_path_pg, sep='\t', index=False)
df[df['category_refined'] == 'problèmes instruction'].to_csv(export_path_pi, sep='\t', index=False)
df[df['category_refined'] == 'autre'].to_csv(export_path_l, sep='\t', index=False)


plt.figure(figsize=(10, 6))
sns.histplot(size_series, bins=range(min_size, max_size + 1), color='blue', stat='count', alpha=0.7)
sns.kdeplot(size_series, color='red', linewidth=2)
plt.title('Distribution of Summary Lengths')
plt.xlabel('Summary Length (tokens)')
plt.ylabel('Number of Summaries')
plt.grid(axis='y', alpha=0.75)
plt.savefig(graph_file, dpi=300, bbox_inches='tight')
plt.show()


print(f"Total summaries: {total}")
print(f"Average size: {avg_size}")
print(f"Variance: {var_size}")
print(f"Median size: {median_size}")
print(f"Min size: {min_size} (id: {min_id})")
print(f"Max size: {max_size} (id: {max_id})")
print("\nCategory counts:")
for cat in category_counts.index:
    count = category_counts[cat]
    perc = (count / total) * 100
    print(f"{cat}: {count} ({perc:.2f}%)")
if not unknown.empty:
    print("\nUnknown categories:")
    print(unknown[['passage_id', 'category_refined']])

print()


print(f"Exported statistics to {stat_file}")
print(f"Exported graph to {graph_file}")
print(f"Exported OK summaries to {export_path}")
print(f"Exported generation problems to {export_path_pg}")
print(f"Exported instruction problems to {export_path_pi}")
print(f"Exported 'autre' to {export_path_l}")
print(f"Script executed in {time.time() - start:.2f} seconds.")
