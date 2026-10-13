S3 

B2 — La taille de chunk, et le passage au récursif
indicateurs 1 : le doc attendu est-il dans le top-k ? Binaire, par question : 1 si au moins un chunk du doc attendu apparaît dans les k résultats, 0 sinon. Tu moyennes sur tes 3 questions. C'est ta mesure de base : « est-ce que le bon document remonte, oui ou non ? »

indicateurs 2 : rang du premier bon chunk (à quelle position ?)
Plus fin que Hit@k : le bon doc est-il en position 1, ou noyé en position 4 ? Un chunking qui met la bonne réponse en [1] vaut mieux qu'un qui la met en [5], même si les deux « hit ». Tu notes le rang (1 = idéal ; ∞ ou k+1 si absent). La version chiffrée classique de ça s'appelle le MRR (mean reciprocal rank = moyenne de 1/rang) — 1.0 si toujours en tête, 0.5 si toujours en 2ᵉ, etc. Une seule ligne à calculer, et c'est un vrai vocabulaire d'évaluation à mettre dans ton README.

indicateurs 3 : précision du top-k (combien de bruit ?)
Sur les k chunks remontés, combien viennent réellement du doc attendu ? nb_chunks_du_bon_doc / k. C'est ici que la taille de chunk se voit le mieux : avec des petits chunks (150), tu peux avoir 3 chunks pertinents du bon doc dans ton top-5 ; avec des gros (500), peut-être 1 seul chunk mais qui contient tout. Ça capture le compromis « précis mais fragmenté » vs « complet mais grossier ».

                chunk=150              chunk=500
          Hit@5  rang  prec@5    Hit@5  rang  prec@5
Q1          1     1    0.40        1     1     0.20
Q2          1     2    0.40        1     1     0.20
Q3          1     1    0.20        0     -     0.00

Overlaps (15% de la taille du chunk)

Mesures : 

Deux texte dont la seul differences est le "passage: " : 0.953
