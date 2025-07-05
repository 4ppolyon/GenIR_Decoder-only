Obtaining file:///lustre/fswork/projects/rech/dsv/ufy16sp/Stage_M2/llama3
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
-----------------------------------------
Mémoire RAM disponible avant exécution :
               total        used        free      shared  buff/cache   available
Mem:           502Gi        45Gi       388Gi       172Mi        88Gi       457Gi
Swap:             0B          0B          0B
-----------------------------------------
Exécution de Test_Generation_Tree_Beam2_chat.py sans argument d'entrée avec torchrun...
ModelArgs with use_scaled_rope loaded!
Parameters:
	Model: Llama-3.2-3B/original
	Trie: Trie/MS_tokenized_trie.json
	Mode: chat
	Prompt: Everything about cancer

Utilisation du tokenizer Llama-3.2-3B/original/tokenizer.model

Loading Tokenized_Trie from file
Tokenized_Trie loaded successfully. Time elapsed: 926.06 seconds

Total GPUs available: 1
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Modèle chargé en 9.48s.
Starting generation (beam search)...
Profondeur 0
1	For: ([], 0.0)
	1 préfixes
Profondeur 1
1	For: ([36064], 0.2353515625)
	1 préfixes
Profondeur 2
1	For: ([36064, 499], 0.15720748901367188)
	1 préfixes
Profondeur 3
1	For: ([36064, 499, 1390], 0.054347120225429535)
	1 préfixes
Profondeur 4
1	For: ([36064, 499, 1390, 311], 0.054347120225429535)
	1 préfixes
Profondeur 5
1	For: ([36064, 499, 1390, 311, 1440], 0.054347120225429535)
	1 préfixes
Profondeur 6
1	For: ([36064, 499, 1390, 311, 1440, 922], 0.054347120225429535)
	1 préfixes
Profondeur 7
1	For: ([36064, 499, 1390, 311, 1440, 922, 9641], 0.054347120225429535)
	1 préfixes
Profondeur 8
1	For: ([36064, 499, 1390, 311, 1440, 922, 9641, 3420], 0.054347120225429535)
	1 préfixes
Profondeur 9
1	For: ([36064, 499, 1390, 311, 1440, 922, 9641, 3420, 596], 0.054347120225429535)
	1 préfixes
Profondeur 10
1	For: ([36064, 499, 1390, 311, 1440, 922, 9641, 3420, 596, 29320], 0.054347120225429535)
	1 préfixes
Profondeur 11
1	For: ([36064, 499, 1390, 311, 1440, 922, 9641, 3420, 596, 29320, 70107], 0.054347120225429535)
	1 préfixes
Profondeur 12
1	For: ([36064, 499, 1390, 311, 1440, 922, 9641, 3420, 596, 29320, 70107, 13], 0.054347120225429535)
	1 préfixes
Profondeur 13
0	For: ([36064, 499, 1390, 311, 1440, 922, 9641, 3420, 596, 29320, 70107, 13, 9641], 0.054347120225429535)
	0 préfixes
Génération terminée en 0.66s

	Input: Everything about cancer
	Response 1: Everything you want to know about Donald Trump's bankruptcies. Donald	(Score: 5.4347%)
