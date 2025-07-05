import time
import pandas as pd
from llama3.tokenizer import Tokenizer

# file_name = 'rd/MS_rd_titres'
# file_name = 'rd/MS_rd_titres_vllm'
# file_name = 'rd/MS_tk_rd_titres_vllm'

# file_name = '1000st/MS_1000st_titres'
# file_name = '1000st/MS_1000st_titres_vllm'

file_name = 'full/MS_titres_vllm'
# file_name = 'full/MS_short_titres_vllm'

# file_name = 'full/corrected/MS_titres_vllm_corrected'
# file_name = 'full/corrected/MS_short_titres_vllm_corrected'

min_tk_nb = 9
max_tk_nb = 21
if 'short' in file_name:
    min_tk_nb = 4
    max_tk_nb = 16

file_path = './MS/data/' + file_name + '.tsv'


tokenizer = Tokenizer(model_path="./Llama-3.2-3B-Instruct/original/tokenizer.model")

df = pd.read_csv(file_path, sep='\t', header=None, usecols=[0, 1], names=['passage_id', 'summary'])
start=time.time()
df['tokenized_summary'] = df['summary'].apply(lambda x: tokenizer.encode(x, bos=False, eos=False) if isinstance(x, str) else [])


df['size'] = df['tokenized_summary'].apply(lambda x: len(x) if x else 0)


def classify_summary_refined_token(summary, size):
    if not isinstance(summary, str) or summary.strip() == '':
        return 'problèmes de génération'

    summary_lower = summary.lower()


    instruction_phrases = [
        "i cannot provide",
        "i can't tell",
        "i am not able",
        "i'm not able",
        "unable to provide",
        "i cannot create explicit content",
        "i cannot fulfill your request",
        "i can't provide information",
        "i can't assist with",
        "i can't create content",
        "i'm happy to help",
        "i'm ready to assist",
        "i'm ready to help",
        "i don't see a paragraph provided",
        "i couldn't find any",
        "i didn't receive a paragraph",
        "i was trained on a vast amount",
        "i'm a large language model",
        "i'm a text-based ai",
        "i'm an ai",
        "i'm an artificial intelligence language model",
        "i'm sorry, but i can't fulfill your request",
        "i can't provide a summary",
        "i can't provide guidance",
        "i can't provide you with",
        "i can't provide assistance",
        "i'm here to help, but i can't provide",
        "i'll choose headache",
        "i see what you did there",
    ]
    ending_phrases = [
        "Is there anything else I can help you with?"
        "Is there anything else I can assist you with?",
    ]

    if any(phrase in summary_lower for phrase in instruction_phrases) or \
         any(phrase in summary_lower for phrase in ending_phrases):
        return 'problèmes instruction'


    generation_issues = ['"', '\n', 'here is a nominal phrase', 'this is a nominal phrase', 'here are the summaries', 'here is a summary', 'here\'s a nominal phrase', 'here\'s a summary', 'here are the nominal phrases']
    if any(issue in summary_lower for issue in generation_issues):
        return 'problèmes de génération'


    if min_tk_nb <= size <= max_tk_nb:
        return 'ok'
    else:
        return 'autre'

df['category_refined'] = df.apply(
    lambda row: classify_summary_refined_token(row['summary'], row['size']),
    axis=1
)


df.to_csv(file_path, sep='\t', index=False, header=False)
print(f"Fichier {file_path} mis à jour avec succès en {time.time()-start:.2f} secondes.")
