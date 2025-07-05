import json
import time
from tqdm import tqdm
import pandas as pd


from llama3.tokenizer import Tokenizer


class TrieNode:
    def __init__(self):
        self.children = {}  
        self.is_end_of_word = False  
        self.id = -1  

    def to_dict(self):
        """
        Serialize the TrieNode into a dictionary for JSON saving.
        """
        return {
            'children': {str(key): child.to_dict() for key, child in self.children.items()},
            'is_end_of_word': self.is_end_of_word,
            
        }

    @staticmethod
    def from_dict(data):
        """
        Deserialize a dictionary back into a TrieNode.
        """
        node = TrieNode()
        node.is_end_of_word = data['is_end_of_word']
        
        node.children = {int(key): TrieNode.from_dict(child_data) for key, child_data in data['children'].items()}
        return node


class Tokenized_Trie:
    def __init__(self, filename="./Llama-3.2-3B-Instruct/original/tokenizer.model"):
        self.root = TrieNode()
        print("Utilisation du tokenizer "+ filename)
        self.tokenizer = Tokenizer(model_path=filename)

    def insert(self, tokenized_title, ID = None):
        node = self.root
        for token in tokenized_title:
            if token not in node.children:
                node.children[token] = TrieNode()
            node = node.children[token]
        if ID is not None:
            node.id = ID
        node.is_end_of_word = True

    def after_this(self, prefix_tokens=None):
        node = self.root
        if prefix_tokens is None:
            prefix_tokens = []
        for token in prefix_tokens:
            assert token in node.children, f"Le token '{token}' n'est pas dans le Trie : Préfix '{prefix_tokens}'"
            node = node.children[token]
        next_token = list(node.children.keys())
        if node.is_end_of_word:
            next_token.append(self.tokenizer.eos_id)
        return next_token, node.id

    def load(self, name):
        if '_rd_' in name:
            dir = 'MS/data/rd/'
        elif '_1000st_' in name:
            dir = 'MS/data/1000st/'
        elif '_corrected' in name:
            dir = 'MS/data/full/corrected/'
        else:
            dir = 'MS/data/full/'
        end = '.tsv'
        filename = dir + name + end
        start = time.time()
        df = pd.read_csv(filename, sep='\t', usecols=[0, 1], header=None, names=['ID','summary'])
        print(f"Création de l'arbre de préfixe à partir de : {filename}")

        df.apply(
            lambda x: self.insert(
                self.tokenizer.encode(x["summary"], bos=False, eos=False),
                x["ID"]
            ),
            axis=1  
        )

        print("Trie créé avec succès. Temps écoulé : {:.2f} secondes".format(time.time() - start))

    def max_depth(self):
        def _max_depth(node):
            if not node.children:
                return 1
            return 1 + max(_max_depth(child) for child in node.children.values())
        return _max_depth(self.root)

    def min_depth(self):
        def _min_depth(node):
            if node.is_end_of_word:
                return 1
            if not node.children:
                return float('inf')
            return 1 + min(_min_depth(child) for child in node.children.values())
        return _min_depth(self.root)
