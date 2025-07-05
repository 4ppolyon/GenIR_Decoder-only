import csv
from llama3.tokenizer import Tokenizer
from tqdm import tqdm


def read_tsv(file_path):
    with open(file_path, newline='', encoding='utf-8') as file:
        reader = csv.reader(file, delimiter='\t')
        data = [row for row in reader]
    return data


def creat_grouped_good(file_good : str = "./MS/data/qrels.train.tsv"):

    good_content = read_tsv(file_good)


    good_dict = {}
    for row in good_content:
        key = row[0]
        value = row[2]
        if key not in good_dict:
            good_dict[key] = []
        good_dict[key].append(value)


    grouped_good = {}
    for key, value_list in good_dict.items():
        size = len(value_list)
        if size not in grouped_good:
            grouped_good[size] = []
        grouped_good[size].append((key, value_list))


    for size, items in sorted(grouped_good.items()):
        print(f"Taille {size}: {len(items)} clés")
        for key, values in items[:10]:
            print(f"  - {key} {values}")


    with open("grouped_good.tsv", "w", newline='', encoding='utf-8') as file:
        writer = csv.writer(file, delimiter='\t')
        for size, items in sorted(grouped_good.items()):
            for key, values in items:
                writer.writerow([key, values, size])

def max_long_all_passage(all_content):

    max_length = 0
    max_passage = ""
    for row in tqdm(all_content):
        passage = row[1]
        length = len(passage.split())
        if length > max_length:
            max_length = length
            max_passage = passage

    print(f"Longueur maximale des passages: {max_length} mots\n{max_passage}")

def max_long_all_passage_token(all_content, tokenizer_path : str = "./Llama-3.2-3B/original/tokenizer.model"):
    tokenizer = Tokenizer(model_path=tokenizer_path)

    max_length = 0
    max_passage = ""
    lengths = {}
    for row in tqdm(all_content):
        passage = row[1]
        length = len(tokenizer.encode(passage, bos=False, eos=False))
        if length not in lengths:
            lengths[length] = 1
        else:
            lengths[length] += 1
        if length > max_length:
            max_length = length
            max_passage = passage
    print(f"Longueur maximale des passages: {max_length} tokens")
    print(max_passage)
    print(f"lengths des longueurs des passages :{lengths}")
    for i in range (1, max_length + 1):
        if i in lengths.keys():
            print(f"{i} tokens : {lengths[i]} passages")
        else:
            print(f"{i} tokens : 0 passages")


if __name__ == "__main__":
    file_all = "./MS/data/MS_collection.tsv"
    all_content = read_tsv(file_all)

    max_long_all_passage_token(all_content)
