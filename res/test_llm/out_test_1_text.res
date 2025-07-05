Obtaining file:///lustre/fswork/projects/rech/dsv/ufy16sp/Stage_M2/llama3
  Preparing metadata (setup.py): started
  Preparing metadata (setup.py): finished with status 'done'
-----------------------------------------
Mémoire RAM disponible avant exécution :
               total        used        free      shared  buff/cache   available
Mem:           502Gi        47Gi       254Gi       1.7Gi       215Gi       454Gi
Swap:             0B          0B          0B
-----------------------------------------
Exécution de Test_Generation_Tree_Beam2.py sans argument d'entrée avec torchrun...
ModelArgs with use_scaled_rope loaded!
Parameters:
	Model: Llama-3.2-3B/original
	Trie: Trie/MS_tokenized_trie.json
	Mode: text
	Prompt: Everything about cancer

Utilisation du tokenizer Llama-3.2-3B/original/tokenizer.model

Loading Tokenized_Trie from file
Tokenized_Trie loaded successfully. Time elapsed: 992.39 seconds

Total GPUs available: 1
> initializing model parallel with size 1
> initializing ddp with size 1
> initializing pipeline with size 1
Modèle chargé en 8.81s.
Starting generation (beam search)...
Profondeur 0
1	For: ([], 0.0)
	1 préfixes
Profondeur 1
1	For: ([34], 0.0703125)
	1 préfixes
Profondeur 2
1	For: ([34, 11967], 0.06097412109375)
	1 préfixes
Profondeur 3
1	For: ([34, 11967, 374], 0.03263068199157715)
	1 préfixes
Profondeur 4
1	For: ([34, 11967, 374, 264], 0.023708229884505272)
	1 préfixes
Profondeur 5
1	For: ([34, 11967, 374, 264, 8624], 0.007130991019948851)
	1 préfixes
Profondeur 6
1	For: ([34, 11967, 374, 264, 8624, 315], 0.003983327015049554)
	1 préfixes
Profondeur 7
1	For: ([34, 11967, 374, 264, 8624, 315, 653], 0.0024584596421008964)
	1 préfixes
Profondeur 8
1	For: ([34, 11967, 374, 264, 8624, 315, 653, 59707], 0.0024584596421008964)
	1 préfixes
Profondeur 9
1	For: ([34, 11967, 374, 264, 8624, 315, 653, 59707, 2849], 0.0024584596421008964)
	1 préfixes
Profondeur 10
1	For: ([34, 11967, 374, 264, 8624, 315, 653, 59707, 2849, 6650], 0.001978291743253065)
	1 préfixes
Profondeur 11
1	For: ([34, 11967, 374, 264, 8624, 315, 653, 59707, 2849, 6650, 304], 0.0013987140840968936)
	1 préfixes
Profondeur 12
1	For: ([34, 11967, 374, 264, 8624, 315, 653, 59707, 2849, 6650, 304, 10099], 0.0010381081092906632)
	1 préfixes
Profondeur 13
0	For: ([34, 11967, 374, 264, 8624, 315, 653, 59707, 2849, 6650, 304, 10099, 13], 0.0010381081092906632)
	0 préfixes
Génération terminée en 0.84s

	Input: Everything about cancer
	Response 1: Cancer is a disease of uncontrolled cell growth in animals.	(Score: 0.1038%)
