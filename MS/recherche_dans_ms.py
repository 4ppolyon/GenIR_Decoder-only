import pandas as pd
from tokenizer import Tokenizer

try:
    input_id = int(input("Entrez l'ID du passage à afficher : ").strip())
except ValueError:
    print("Erreur : l'ID doit être un entier.")
    exit(1)

# un bon exemple de passage/titre id est : 7715968

search_a_query = False

if search_a_query:
    query_file_1 = './MS/data/queries.dev.tsv'
    query_file_2 = './MS/data/queries.train.tsv'

    try:
        df = pd.read_csv(query_file_1, sep='\t', header=None, usecols=[0, 1], names=['query_id', 'query'])
    except FileNotFoundError:
        print(f"Erreur : fichier {query_file_1} introuvable.")
        exit(1)

    try:
        df2 = pd.read_csv(query_file_2, sep='\t', header=None, usecols=[0, 1], names=['query_id', 'query'])
    except FileNotFoundError:
        print(f"Erreur : fichier {query_file_2} introuvable.")
        exit(1)

    row = df[df['query_id'] == input_id]
    row2 = df2[df2['query_id'] == input_id]

    if row.empty:
        print(f"Aucune query trouvée pour l'ID : {input_id} dans {query_file_1}")
    else:
        print(f"ID : {row['query_id'].values[0]}")
        print(f"query : {row['query'].values[0]}")
    if row2.empty:
        print(f"Aucune query trouvée pour l'ID : {input_id} dans {query_file_2}")
    else:
        print(f"ID : {row2['query_id'].values[0]}")
        print(f"query : {row2['query'].values[0]}")

else:
    passage_file = './MS/data/MS_collection.tsv'
    # title_file = './MS/data/full/corrected/MS_short_titres_vllm_corrected.tsv'
    title_file = './MS/data/full/corrected/MS_titres_vllm_corrected.tsv'

    try:
        df = pd.read_csv(passage_file, sep='\t', header=None, usecols=[0, 1], names=['passage_id', 'passage'])
    except FileNotFoundError:
        print(f"Erreur : fichier {passage_file} introuvable.")
        exit(1)


    row = df[df['passage_id'] == input_id]
    tokenizer = Tokenizer(model_path="./Llama-3.2-3B-Instruct/original/tokenizer.model")

    if row.empty:
        print(f"Aucun passage trouvé pour l'ID : {input_id}, dans {passage_file}")
    else:
        for index, data in row.iterrows():
            print()
            print(f"Passage : {data['passage']}")
            print(f"Passage tokenized : {tokenizer.encode(data['passage'], bos=False, eos=False)}")
            print(f"Taille du passage tokenized : {len(tokenizer.encode(tokenizer.decode(tokenizer.encode(data['passage'], bos=False, eos=False)[:16]), bos=False, eos=False))}")

    try:
        df = pd.read_csv(title_file, sep='\t', header=None, usecols=[0, 1], names=['passage_id', 'passage'])
    except FileNotFoundError:
        print(f"Erreur : fichier {title_file} introuvable.")
        exit(1)

    row = df[df['passage_id'] == input_id]
    tokenizer = Tokenizer(model_path="./Llama-3.2-3B-Instruct/original/tokenizer.model")

    if row.empty:
        print(f"Aucun passage trouvé pour l'ID : {input_id}, dans {title_file}")
    else:
        for index, data in row.iterrows():
            print()
            print(f"Titre : {data['passage']}")
            print(f"Titre tokenized : {tokenizer.encode(data['passage'], bos=False, eos=False)}")
            print(f"Taille du titre tokenized : {len(tokenizer.encode(tokenizer.decode(tokenizer.encode(data['passage'], bos=False, eos=False)[:16]), bos=False, eos=False))}")
