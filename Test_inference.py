from llama3.generation_modified_Beam import *

# export RANK=0
# export WORLD_SIZE=1
# export LOCAL_RANK=0
# export MASTER_ADDR=localhost
# export MASTER_PORT=29500

# Contexte système : rôle clair
SYSTEM_PROMPT = (
    "You are an assistant that writes concise and informative titles"
    # "The assistant is an expert in the field(s) raised by the user. It must respond in a precise, concise and useful way, providing relevant information and following the instructions given. There is only one request from the user and the assistant must respond to this request without asking further questions."
    # "You provide summary of documents which are related to the user request."
)

def Take_model(name: str, trie: str):
    """
    Charge le modèle Llama à partir du répertoire spécifié.
    """
    return Llama.build(
        ckpt_dir=name,
        trie_path=trie
    )

def Send_model(prompt: str, model: Llama, mode: str = "text", beam_width: int = 2, nb_titres: int = 10):
    """
    Envoie une liste d'entrées au modèle et récupère les réponses.
    """
    if mode.lower() == "chat":
        formated_prompt = [
            {
                "role": "user",
                "content": prompt.strip()
            }
        ]
        res = model.chat_completion(
            dialog=formated_prompt,
            beam_width=beam_width,
            nb_titres=nb_titres,
        )
    elif mode.lower() == "text":
        formated_prompt = prompt.strip() + ":\n"
        res = model.text_completion(
            prompt=formated_prompt,
            beam_width=beam_width,
            nb_titres=nb_titres,
        )
    else:
        raise ValueError("Mode must be 'chat' or 'text'.")
    return res

def print_answer(response : List[Tuple[str, float, int]], prompt: str):
    """
    Affiche la réponse brute du modèle.
    """
    print(f"\n\tInput: {prompt}\n")
    for i in range(len(response)):
        print(f"\tResponse {i + 1}: ID = {response[i][2]} : {response[i][0]}\t(Score: {response[i][1]*100:.4f}%)")

if __name__ == "__main__":
    name = "Llama-3.2-3B-Instruct/original"
    trie = "MS_titres_vllm_corrected"
    # trie = "MS_rd_titres"
    mode = "chat"
    prompts = ["Everything about cancer",
               "what was the immediate impact of the success of the manhattan project?",
               "what are bed bugs caused from"
    ]

    print(f"Parameters:\n\tModel: {name}\n\tTrie: {trie}\n\tMode: {mode}\n")
    model = Take_model(name, trie)
    for prompt in prompts:
        print(f"\nTesting prompt: {prompt}")
        for i in [1, 2, 6]:
            response = Send_model(prompt, model, mode, nb_titres=10, beam_width=i)
            print_answer(response, prompt)