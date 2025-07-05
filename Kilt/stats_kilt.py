import json
from llama3.tokenizer import Tokenizer
from tqdm import tqdm
import random


def max_long_all_passage_token(tokenizer_path : str = "./Llama-3.2-3B/original/tokenizer.model"):
    tokenizer = Tokenizer(tokenizer_path)
    lengths = {}
    max_length = 0
    total_length = 0
    nb_articles = 0
    random_lines = random.sample(range(0, 5903530), 10)  # Prendre 10 lignes aléatoires
    print("Exemples de titres :")
    with open('data/kilt_knowledgesource.json', 'r', encoding='utf-8') as f:
        for line in tqdm(f, desc='Indexing titles', unit='lines'):
            try:
                article = json.loads(line)
                title = article.get('wikipedia_title', '').strip()
                if title:
                    title_tokens = tokenizer.encode(title, bos=False, eos=False)
                    length = len(title_tokens)
                    total_length += length
                    nb_articles += 1
                    if nb_articles in random_lines:
                        print(f"\t- {title}")
                        random_lines.remove(nb_articles)
                    max_length = max(max_length, length)
                    if length not in lengths.keys():
                        lengths[length] = 1
                    else:
                        lengths[length] += 1
            except json.JSONDecodeError:
                exit(-1)
    print(f"il y a {nb_articles} articles")
    print(f"Longueur maximale des passages: {max_length} tokens")
    print(f"Longueur moyenne des passages: {round(total_length/nb_articles,2)} tokens")
    print("Distribution des longueurs des passages :")
    for i in range (1, max_length + 1):
        if i in lengths.keys():
            print(f"{i} tokens : {lengths[i]} passages")
        else:
            print(f"{i} tokens : 0 passages")

if __name__ == "__main__":
    max_long_all_passage_token()
