#!/bin/bash

# Mettre à jour toutes les références distantes
echo "Mise à jour des références distantes..."
git fetch --all

# Sauvegarder la branche actuelle
current_branch=$(git branch --show-current)
echo "Branche actuelle : $current_branch"

# Récupérer toutes les branches distantes
echo -e "\nRécupération de toutes les branches distantes..."
for branch in $(git branch -r | grep -v HEAD); do
    # Enlever 'origin/' du nom de la branche
    branch_name=${branch#origin/}
    
    echo -e "\nTraitement de la branche : $branch_name"
    
    # Vérifier si la branche locale existe déjà
    if git show-ref --verify --quiet refs/heads/$branch_name; then
        echo "La branche $branch_name existe déjà localement"
        git checkout $branch_name
        git pull origin $branch_name
    else
        echo "Création de la branche locale $branch_name"
        git checkout -b $branch_name $branch
    fi
done

# Retourner à la branche initiale
echo -e "\nRetour à la branche initiale : $current_branch"
git checkout $current_branch

echo -e "\nToutes les branches ont été récupérées et mises à jour !"
# Afficher toutes les branches
echo -e "\nListe des branches disponibles :"
git branch -a 