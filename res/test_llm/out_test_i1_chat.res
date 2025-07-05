Obtaining file:///lustre/fswork/projects/rech/dsv/ufy16sp/Stage_M2/llama3
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
-----------------------------------------
Mémoire RAM disponible avant exécution :
               total        used        free      shared  buff/cache   available
Mem:           502Gi        37Gi       453Gi       167Mi        29Gi       464Gi
Swap:             0B          0B          0B
-----------------------------------------
Exécution de Test_Generation_Tree_Beam2_chat.py sans argument d'entrée avec torchrun...
ModelArgs with use_scaled_rope loaded!
Parameters:
	Model: Llama-3.2-3B-Instruct/original
	Trie: Trie/MS_tokenized_trie.json
	Mode: chat
	Prompt: Everything about cancer

Utilisation du tokenizer Llama-3.2-3B-Instruct/original/tokenizer.model

Loading Tokenized_Trie from file
Tokenized_Trie loaded successfully. Time elapsed: 921.70 seconds

Total GPUs available: 1
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Modèle chargé en 12.24s.
Starting generation (beam search)...
Profondeur 0
1	For: ([], 0.0)
	1 préfixes
Profondeur 1
1	For: ([34], 0.5234375)
	1 préfixes
Profondeur 2
1	For: ([34, 11967], 0.5234375)
	1 préfixes
Profondeur 3
1	For: ([34, 11967, 374], 0.5234375)
	1 préfixes
Profondeur 4
1	For: ([34, 11967, 374, 264], 0.5234375)
	1 préfixes
Profondeur 5
1	For: ([34, 11967, 374, 264, 6485], 0.492767333984375)
	1 préfixes
Profondeur 6
1	For: ([34, 11967, 374, 264, 6485, 1912], 0.3599511384963989)
	1 préfixes
Profondeur 7
1	For: ([34, 11967, 374, 264, 6485, 1912, 315], 0.3599511384963989)
	1 préfixes
Profondeur 8
1	For: ([34, 11967, 374, 264, 6485, 1912, 315, 19338], 0.3599511384963989)
	1 préfixes
Profondeur 9
1	For: ([34, 11967, 374, 264, 6485, 1912, 315, 19338, 28987], 0.3599511384963989)
	1 préfixes
Profondeur 10
1	For: ([34, 11967, 374, 264, 6485, 1912, 315, 19338, 28987, 5370], 0.3599511384963989)
	1 préfixes
Profondeur 11
1	For: ([34, 11967, 374, 264, 6485, 1912, 315, 19338, 28987, 5370, 36853], 0.3599511384963989)
	1 préfixes
Profondeur 12
0	For: ([34, 11967, 374, 264, 6485, 1912, 315, 19338, 28987, 5370, 36853, 13], 0.3599511384963989)
	0 préfixes
Génération terminée en 7.29s

	Input: Everything about cancer
	Response 1: Cancer is a complex group of diseases affecting various organs.	(Score: 35.9951%)
