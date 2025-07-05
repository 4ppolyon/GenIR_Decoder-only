# Exigences

## Fichiers
- `kilt_knowledgesource.json` : le [fichier du dataset KILT](http://dl.fbaipublicfiles.com/KILT/kilt_knowledgesource.json), a placer dans le dossier `Kilt/data/` du projet.
- `MS_collection.tsv` : le [fichier du dataset MS MARCO](https://microsoft.github.io/msmarco/Datasets.html), a placer dans le dossier `MS/data/` du projet.

## Répertoire
- `Llama-3.2-3B-Instruct` : le [modèle Llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct) de Hugging Face, a placer a la racine du projet.

# Pipelines

## Pipeline Generation des titres a partir des passages
### 1. ```python -m Generation_titres_vllm.py```

Il y a plusieurs prompts disponibles dans le fichier, vous pouvez les commenter/décommenter pour choisir celui que vous souhaitez utiliser. Celui décommenté est celui qui a été utilisé pour la génération des titres dans mon memoire.

## Pipeline de post-traitement des titres
### 1. ```python -m MS/add_data_titles.py```

Ajoute aux titres générés les données supplémentaires (passages tokenisés, nombre de tokens, catégorie) pour chaque titre.

### 2. ```python -m MS/stats_titres.py```

Affiche les statistiques des titres générés (en nombre de tokens), moyenne, variance, min, max, ainsi quantité et fréquence de chaque catégorie. Mais aussi, enregistre les titres généré dans des fichiers tsv dans le répertoire `res/gen_titres/full/` en fonction de la catégorie pour faciliter le controle de la qualité des titres généré.

### 3. ```python -m MS/corr_titres.py```

Corrige les titres générés en fonction de la catégorie, en utilisant des règles de correction prédéfinies. Les titres corrigés sont enregistrés dans un fichier tsv. Il y a plusieurs règles de correction disponibles dans le dossier `MS/data/full/corrected/`.

### 4. ```python -m MS/add_data_titles.py```

Ajoute aux titres générés et corrigé les données supplémentaires (passages tokenisés, nombre de tokens, catégorie) pour chaque titre.

### 5. ```python -m MS/stats_titres.py``` 

Affiche les statistiques des titres générés (en nombre de tokens), moyenne, variance, min, max, ainsi quantité et fréquence de chaque catégorie. Mais aussi, enregistre les titres généré dans des fichiers tsv dans le répertoire `res/gen_titres/full/corrected/` en fonction de la catégorie pour faciliter le controle de la qualité des titres généré.

## Evaluation sur le dataset de développement de MS MARCO
### 1. ```python -m generate_predictions_full.py```

Génère les prédictions pour le dataset de développement de MS MARCO en utilisant les titres générés et corrigés. Les prédictions sont enregistrées dans un fichier tsv `MS/eval/full/generated_responses.tsv`.

### 2. ```python -m Eval.py``` 

Évalue les prédictions générées en utilisant le dataset de développement de MS MARCO. Affiche le MRR@10 et le NDCG@10 des prédictions. Sauvegarde les résultats par requête dans un fichier tsv `per_query_scores.tsv`.

### 3. ```python -m distrib_score.py```

Affiche la distribution des scores MRR@10 et NDCG@10 pour chaque catégorie.

## Pipeline de fine-tuning du modèle Llama-3.2-3B-Instruct
### 1. ```python -m fine-tuning/ft_in.py```

Fine-tune avec entrée et sortie attendu dans l'entrée du modèle Llama-3.2-3B-Instruct sur les titres générés et corrigés. Le modèle fine-tuné est enregistré dans le dossier `llama3_lora_finetuned_in/`. Le modèle est fine-tuné sur les titres générés et corrigés du dataset de developpement.
 - Possede un mode DEBUG pour afficher ce qui est envoyé a Lora.

### 1.bis ```python -m fine-tuning/ft_in_out.py```

Fine-tune avec entrée et sortie attendu séparé le modèle Llama-3.2-3B-Instruct sur les titres générés et corrigés. Le modèle fine-tuné est enregistré dans le dossier `llama3_lora_finetuned_in_out/`. Le modèle est fine-tuné sur les titres générés et corrigés du dataset de developpement.
 - Possede un mode DEBUG pour afficher ce qui est envoyé a Lora.

### 2. ```python -m fine-tuning/ft_test.py```

Test du modèle fine-tuné sur les requêtes du dataset de développement de MS MARCO. /!\ **En language naturel** /!\
 - A une option save pour enregistrer le modele en format `.pth` dans le dossier `llama3_lora_finetuned_in_out/` ou `llama3_lora_finetuned_in/` selon le modèle fine-tuné utilisé.

# Autres codes

## Tests et inférences
- `test_inference.py` :

un fichier de test pour tester l'inférence avec le modèle Llama-3.2-3B-Instruct modifié.

- `test_inference_ft.py` :

un fichier de test pour tester l'inférence avec le modèle Llama-3.2-3B-Instruct modifié et fine_tuné. /!\ Le modele est chargé avec un fichier `.pth` ce qui cause des problemes lors de l'inférence, les probabilités sont autour 10^-5 **_(non résolu pour le moment)_**.

- `genere_fichier_fine_tune.py` :

un fichier pour générer le fichier associant les titres générés et corrigés aux requêtes du dataset de développement de MS MARCO pour le fine-tuning du modèle Llama-3.2-3B-Instruct.

- `recherche_dans_ms.py` :

un fichier pour rechercher dans le dataset de MS MARCO (query ou passage et titre) en utilisant les IDs.

- `verif_qrels_in_titres.py` :

puisqu'il y a une légère possibilité qu'un passage soit laissé de coté, alors une verification est faite pour s'assurer que les passages des qrels sont bien présents dans les titres générés et corrigés.

- `Kilt/stats_Kilt.py` et `MS/stats_MS.py`

affichent les statistiques des datasets KILT et MS MARCO, respectivement.

# Autres

Les autres fichiers sont là pour donner des informations sur les résultats obtenus, si l'utilisateur souhaite les consulter. Ils ne sont pas nécessaires pour le bon fonctionnement du projet mais représentent des résultats intéressants obtenus lors du développement du projet.

# Auteur
- **Romain Alves**