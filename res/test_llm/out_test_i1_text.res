Obtaining file:///lustre/fswork/projects/rech/dsv/ufy16sp/Stage_M2/llama3
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
-----------------------------------------
Mémoire RAM disponible avant exécution :
               total        used        free      shared  buff/cache   available
Mem:           502Gi        51Gi       398Gi       1.9Gi        75Gi       451Gi
Swap:             0B          0B          0B
-----------------------------------------
Exécution de Test_Generation_Tree_Beam2.py sans argument d'entrée avec torchrun...
ModelArgs with use_scaled_rope loaded!
Parameters:
	Model: Llama-3.2-3B-Instruct/original
	Trie: Trie/MS_tokenized_trie.json
	Mode: text
	Prompt: Everything about cancer

Utilisation du tokenizer Llama-3.2-3B-Instruct/original/tokenizer.model

Loading Tokenized_Trie from file
Tokenized_Trie loaded successfully. Time elapsed: 971.14 seconds

Total GPUs available: 1
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Modèle chargé en 8.79s.
Starting generation (beam search)...
Profondeur 0
1	For: ([], 0.0)
	1 préfixes
Profondeur 1
1	For: ([9], 0.16796875)
	1 préfixes
Profondeur 2
1	For: ([9, 220], 0.04691314697265625)
	1 préfixes
Profondeur 3
1	For: ([9, 220, 18], 0.04416432976722717)
	1 préfixes
Profondeur 4
1	For: ([9, 220, 18, 12], 0.04416432976722717)
	1 préfixes
Profondeur 5
1	For: ([9, 220, 18, 12, 20], 0.04416432976722717)
	1 préfixes
Profondeur 6
1	For: ([9, 220, 18, 12, 20, 5672], 0.04416432976722717)
	1 préfixes
Profondeur 7
1	For: ([9, 220, 18, 12, 20, 5672, 19813], 0.04416432976722717)
	1 préfixes
Profondeur 8
1	For: ([9, 220, 18, 12, 20, 5672, 19813, 311], 0.04416432976722717)
	1 préfixes
Profondeur 9
1	For: ([9, 220, 18, 12, 20, 5672, 19813, 311, 220], 0.04416432976722717)
	1 préfixes
Profondeur 10
1	For: ([9, 220, 18, 12, 20, 5672, 19813, 311, 220, 18], 0.04416432976722717)
	1 préfixes
Profondeur 11
1	For: ([9, 220, 18, 12, 20, 5672, 19813, 311, 220, 18, 12], 0.04416432976722717)
	1 préfixes
Profondeur 12
1	For: ([9, 220, 18, 12, 20, 5672, 19813, 311, 220, 18, 12, 20], 0.04416432976722717)
	1 préfixes
Profondeur 13
0	For: ([9, 220, 18, 12, 20, 5672, 19813, 311, 220, 18, 12, 20, 5672], 0.04416432976722717)
	0 préfixes
Génération terminée en 0.69s

	Input: Everything about cancer
	Response 1: * 3-5 weeks refers to 3-5 weeks	(Score: 4.4164%)
