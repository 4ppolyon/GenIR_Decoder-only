


import os
import sys
from typing import (
    List,
    Optional,
    Tuple,
    TypedDict
)
from pathlib import Path
from fairscale.nn.model_parallel.initialize import (
    get_model_parallel_rank,
    initialize_model_parallel,
    model_parallel_is_initialized,
)


import torch
import torch.nn.functional as F
from torch.distributed import destroy_process_group


from llama3.model import ModelArgs, Transformer
from llama3.tokenizer import ChatFormat, Message, Dialog
from llama3.tokenized_trie import *


class CompletionPrediction(TypedDict, total=False):
    generation: str  
    tokens: List[str]  
    logprobs: List[float]  

class ChatPrediction(TypedDict, total=False):
    generation: Message  
    tokens: List[str]  
    logprobs: List[float]  


class Llama:
    @staticmethod
    def build(
            ckpt_dir: str,
            trie_path: str,
            seed: int = 1
    ) -> "Llama":

        
        tokenizer_path = "Llama-3.2-3B-Instruct/original/tokenizer.model"

        assert os.path.isdir(ckpt_dir), f"Checkpoint directory '{ckpt_dir}' does not exist."
        
        assert os.path.isfile(tokenizer_path), f"Tokenizer file '{tokenizer_path}' does not exist."

        trie = Tokenized_Trie(tokenizer_path)

        print(f"Chargement du Trie depuis {trie_path}")
        trie.load(trie_path)

        max_seq_len = trie.max_depth()
        assert 1 <= max_seq_len <= 8192, f"max_seq_len must be between 1 and 8192, got {max_seq_len}."

        
        num_gpus = torch.cuda.device_count()
        print("Total GPUs available:", num_gpus)
        if not torch.distributed.is_initialized():
            torch.distributed.init_process_group("nccl")
        if not model_parallel_is_initialized():
            model_parallel_size = int(os.environ.get("WORLD_SIZE", 1))
            initialize_model_parallel(model_parallel_size)

        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)

        
        torch.manual_seed(seed)

        if local_rank > 0:
            sys.stdout = open(os.devnull, "w")

        start_time = time.time()
        checkpoints = sorted(Path(ckpt_dir).glob("*.pth"))
        assert len(checkpoints) > 0, f"no checkpoint files found in {ckpt_dir}"
        assert model_parallel_size == len(
            checkpoints
        ), f"Loading a checkpoint for MP={len(checkpoints)} but world size is {model_parallel_size}"
        ckpt_path = checkpoints[get_model_parallel_rank()]
        checkpoint = torch.load(ckpt_path, map_location="cpu")
        with open("Llama-3.2-3B-Instruct/original/params.json", "r") as f:
            params = json.loads(f.read())

        model_args: ModelArgs = ModelArgs(
            max_batch_size=1,
            **params,
        )
        tokenizer = Tokenizer(model_path=tokenizer_path)
        assert model_args.vocab_size == tokenizer.n_words
        if torch.cuda.is_bf16_supported():
            torch.set_default_tensor_type(torch.cuda.BFloat16Tensor)
        else:
            torch.set_default_tensor_type(torch.cuda.HalfTensor)

        
        model = Transformer(model_args)
        model.load_state_dict(checkpoint, strict=False)
        print(f"Modèle chargé en {time.time() - start_time:.2f}s.")

        return Llama(model, tokenizer, trie, max_seq_len)

    def __init__(self, model: Transformer, tokenizer: Tokenizer, trie: Tokenized_Trie, depth: int):
        self.model = model
        self.tokenizer = tokenizer
        self.formatter = ChatFormat(tokenizer)
        self.trie = trie
        self.depth = depth

    def find_next(self, liste, prev, nb, prompt, outlist=None):
        if outlist is None:
            outlist = []

        full_prompt = prompt + prev[0]

        input_ids = torch.tensor([full_prompt], dtype=torch.long, device="cuda")
        with torch.no_grad():
            logits = self.model(input_ids, 0)  
            logits = logits[:, -1, :]  
        logits = logits[0] 

        allowed_logits = logits[liste] 
        probs = F.softmax(allowed_logits, dim=0).tolist() 
        probs_conj = []
        
        for i in range(len(liste)):
            if len(prev[0]) == 0:
                probs_conj.append(probs[i])
            else:
                probs_conj.append(probs[i] * prev[1])

        
        results = [(prev[0] + [liste[i]], probs_conj[i]) for i in range(len(liste))]
        
        results.sort(key=lambda x: x[1], reverse=True)
        
        results = results[:nb]

        
        for i in range(len(results)):
            outlist.append(results[i])

        return len(results)

    def find_best(self, liste, nb):
        liste.sort(key=lambda x: x[1], reverse=True)
        return liste[:nb]

    def beam_search(self, x, y, prompt):
        
        beam = [([], 0.0, -1)]

        
        results = []

        profondeur = 0
        
        while len(beam) > 0 and profondeur < self.depth:
            print(f"Profondeur {profondeur}")
            
            selected_prefix = []
            for current_prefix in beam:
                print(f"Préfixe courant : {current_prefix}")
                next_tokens, id = self.trie.after_this(
                    current_prefix[0])  
                if self.tokenizer.eos_id in next_tokens:
                    assert id != -1, "L'ID ne doit pas être -1."
                    results.append((current_prefix[0], current_prefix[1], id))
                    next_tokens.remove(self.tokenizer.eos_id)
                nb = self.find_next(next_tokens, current_prefix, y, prompt, selected_prefix)
                if id != -1:
                    print(f"{nb}\tFor: {current_prefix}\ton ajoute : {id}")
                else :
                    print(f"{nb}\tFor: {current_prefix}")

            
            selected_prefix.sort(key=lambda x: x[1], reverse=True)

            
            beam = selected_prefix

            print(f"\t{len(beam)} préfixes")
            profondeur += 1
        results = self.find_best(results, x)
        return results

    @torch.inference_mode()
    def generate(
            self,
            prompt_tokens: List[int],
            beam_width: int = 4,
            nb_titres: int = 5,
    ) -> List[Tuple[List[int], float, int]]:
        
        start_time = time.time()
        results = self.beam_search(nb_titres, beam_width, prompt_tokens)

        print(f"Génération terminée en {time.time() - start_time:.2f}s avec faisceau de largeur {beam_width}")
        return results

    def text_completion(
            self,
            prompt: str,
            beam_width: int = 4,
            nb_titres: int = 5,
    ) -> List[Tuple[str, float, int]]:

        prompt_tokens = self.tokenizer.encode(prompt, bos=False, eos=False)
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            beam_width=beam_width,
            nb_titres=nb_titres,
        )
        output = []
        for i in range(len(generation_tokens)):
            output.append((self.tokenizer.decode(generation_tokens[i][0]), generation_tokens[i][1], generation_tokens[i][2]))
        return output

    def chat_completion(
        self,
        dialog: Dialog,
        beam_width: int = 4,
        nb_titres: int = 5,
    ) -> List[Tuple[str, float, int]]:

        prompt_tokens = self.formatter.encode_dialog_prompt(dialog)
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            beam_width=beam_width,
            nb_titres=nb_titres,
        )
        output = []
        for i in range(len(generation_tokens)):
            output.append((self.tokenizer.decode(generation_tokens[i][0]), generation_tokens[i][1], generation_tokens[i][2]))
        return output
