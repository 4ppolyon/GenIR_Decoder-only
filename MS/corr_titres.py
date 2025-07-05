import time

import pandas as pd
from tqdm import tqdm

from llama3.tokenizer import Tokenizer
import os

def corriger_resume(row):
    resume = row['summary']
    resume_tokens = row['tokenized_summary']
    taille = row['size']
    categorie = row['category_refined']

    printing = False


    if categorie == "ok":
        return resume


    if categorie == "problèmes instruction":
        passage = dp.loc[dp['passage_id'] == row['passage_id'], 'passage'].values[0]
        passage_token = tokenizer.encode(passage, bos=False, eos=False)
        taille_passage = len(passage_token)
        if taille_passage < min_tk_nb :
            if printing:
                print(f"[Erreur d'instruction] id={row['passage_id']} : passage trop court ({taille_passage}).")
            return None
        elif taille_passage > max_tk_nb:
            if printing:
                print(f"[Erreur d'instruction] id={row['passage_id']} : passage trop long ({taille_passage}). On coupe à max_tk_nb tokens.")
            return tokenizer.decode(passage_token[:max_tk_nb])
        else:
            if printing:
                print(f"[Erreur d'instruction] id={row['passage_id']} : passage correct ({taille_passage}). On le garde.")
            return passage



    elif categorie == "autre":
        if taille < min_tk_nb:
            passage = dp.loc[dp['passage_id'] == row['passage_id'], 'passage'].values[0]
            passage_token = tokenizer.encode(passage, bos=False, eos=False)
            taille_passage = len(passage_token)
            if taille_passage < min_tk_nb:
                if printing:
                    print(f"[Autre] id={row['passage_id']} : résumé trop court ({taille}).")
                    print(f"\t en plus passage trop court ({taille_passage}).")
                return None
            elif taille_passage > max_tk_nb:
                if printing:
                    print(f"[Autre] id={row['passage_id']} : résumé trop court ({taille}).")
                    print(f"\t passage trop long ({taille_passage}). On coupe à max_tk_nb tokens.")
                return tokenizer.decode(passage_token[:max_tk_nb])
            else:
                if printing:
                    print(f"[Autre] id={row['passage_id']} : résumé trop court ({taille}).")
                    print(f"\t passage correct ({taille_passage}). On le garde.")
                return passage
        elif taille > max_tk_nb:
            if printing:
                print(f"[Autre] id={row['passage_id']} : résumé trop long ({taille}). On coupe à max_tk_nb tokens.")
            return tokenizer.decode(resume_tokens[:max_tk_nb])
        else:
            if printing:
                print(f"[Autre] id={row['passage_id']} : résumé bon ({taille}). On garde le résumé.")
            return resume


    elif categorie == "problèmes de génération":

        lines = resume.split('\n')

        filtered_lines = []
        skip_next_line = False

        for i, line in enumerate(lines):
            if skip_next_line:
                skip_next_line = False
                continue

            line_lower = line.lower()
            matched = False

            for pat in bad_generation_patterns:
                pat_lower = pat.lower()


                if pat_lower in line_lower and ": " in line:
                    before_colon, after_colon = line.split(": ", 1)
                    if pat_lower in before_colon.lower():
                        filtered_lines.append(after_colon.strip())
                        matched = True
                        break


                elif pat_lower in line_lower:
                    if i + 1 < len(lines):
                        next_line = lines[i + 1].strip()
                        if next_line:
                            filtered_lines.append(next_line)
                            skip_next_line = True
                    matched = True
                    break

            if not matched:
                filtered_lines.append(line.strip())

        if not filtered_lines:

            passage = dp.loc[dp['passage_id'] == row['passage_id'], 'passage'].values[0]
            passage_token = tokenizer.encode(passage, bos=False, eos=False)
            taille_passage = len(passage_token)
            if taille_passage < min_tk_nb:
                if printing:
                    print(f"[problèmes de génération] id={row['passage_id']} : aucune ligne utile après filtrage.")
                    print(f"\t en plus passage trop court ({taille_passage}).")
                return None
            elif taille_passage > max_tk_nb:
                if printing:
                    print(f"[problèmes de génération] id={row['passage_id']} : aucune ligne utile après filtrage.")
                    print(f"\t passage trop long ({taille_passage}). On coupe à max_tk_nb tokens.")
                return tokenizer.decode(passage_token[:max_tk_nb])
            else:
                if printing:
                    print(f"[problèmes de génération] id={row['passage_id']} : aucune ligne utile après filtrage.")
                    print(f"\t passage correct ({taille_passage}). On le garde.")
                return passage


        resume_one_line = " ".join(filtered_lines)


        resume_one_line = resume_one_line.replace('"', '').strip()


        resume_one_line = tokenizer.encode(resume_one_line, bos=False, eos=False)
        taille = len(resume_one_line)

        if taille < min_tk_nb:
            passage = dp.loc[dp['passage_id'] == row['passage_id'], 'passage'].values[0]
            passage_token = tokenizer.encode(passage, bos=False, eos=False)
            taille_passage = len(passage_token)
            if taille_passage < min_tk_nb:
                if printing:
                    print(f"[problèmes de génération] id={row['passage_id']} : résumé trop court ({taille}).")
                    print(f"\t en plus passage trop court ({taille_passage}).")
                return None
            elif taille_passage > max_tk_nb:
                if printing:
                    print(f"[problèmes de génération] id={row['passage_id']} : résumé trop court ({taille}).")
                    print(f"\t passage trop long ({taille_passage}). On coupe à max_tk_nb tokens.")
                return tokenizer.decode(passage_token[:max_tk_nb])
            else:
                if printing:
                    print(f"[problèmes de génération] id={row['passage_id']} : résumé trop court ({taille}).")
                    print(f"\t passage correct ({taille_passage}). On le garde.")
                return passage
        elif taille > max_tk_nb:
            if printing:
                print(f"[problèmes de génération] id={row['passage_id']} : résumé trop long ({taille}). On coupe à max_tk_nb tokens.")
            return tokenizer.decode(resume_one_line[:max_tk_nb])
        else:
            if printing:
                print(f"[problème de génération] id={row['passage_id']} : résumé bon ({taille}). On garde le résumé.")
            return tokenizer.decode(resume_one_line)


    else:
        print(f"GROS PROBLEME : catégorie inconnue {categorie} pour l'id {row['passage_id']}. On garde le résumé tel quel.")
        return resume


passage_file = './MS/data/MS_collection.tsv'

file_name = 'MS_titres_vllm'
# file_name = 'MS_short_titres_vllm'

min_tk_nb = 9
max_tk_nb = 20
if 'short' in file_name:
    min_tk_nb = 4
    max_tk_nb = 15

base_path = './MS/data/full/'
corrected = '_corrected'

file_path = f'{base_path}{file_name}.tsv'


df = pd.read_csv(file_path, sep='\t', header=None, names=[
    'passage_id', 'summary', 'tokenized_summary', 'size', 'category_refined'
])


if df.shape[1] < 5:
    raise ValueError(f"Le fichier semble incomplet {df.shape[1]} colonnes. Il faut d'abord exécuter 'add_data'.")


dp = pd.read_csv(passage_file, sep='\t', header=None, names=[
    'passage_id', 'passage'
])


tokenizer = Tokenizer(model_path="./Llama-3.2-3B-Instruct/original/tokenizer.model")


bad_generation_patterns = [
    "here is a nominal phrase",
    "this is a nominal phrase",
    "here are the summaries",
    "here is a summary",
    "here's a nominal phrase",
    "here's a summary",
    "here are the nominal phrases"
]

start = time.time()

corr_file = f'{base_path}{file_name}{corrected}.tsv'


if os.path.exists(corr_file):
    os.remove(corr_file)
else:
    os.makedirs(base_path, exist_ok=True)


df_corr = df['passage_id'].copy()
df_corr = df_corr.to_frame()


import ast
df['tokenized_summary'] = df['tokenized_summary'].apply(ast.literal_eval)
df_corr['resume_corrige'] = df.apply(corriger_resume, axis=1)
df_corr = df_corr[df_corr['resume_corrige'].notnull()]


df_corr.to_csv(corr_file, sep='\t', index=False, header=False)
print(f"Fichier {file_path} traité et sauvegardé sous {corr_file} en {round(time.time() - start, 2)} secondes.")
