#!/bin/bash

#===============================================================================
# Script de soumission SLURM générique
# Auteur: [Votre nom]
# Date: [Date]
#===============================================================================

#===============================================================================
# PARAMÈTRES SLURM OBLIGATOIRES
#===============================================================================
#SBATCH --partition=cpu_shared        # Partition à utiliser
#SBATCH --time=24:00:00              # Temps maximum d'exécution (format: jours-hh:mm:ss)
#SBATCH --account=YourAccountProject  # Nom du projet/compte

#===============================================================================
# RESSOURCES DEMANDÉES
#===============================================================================
#SBATCH --nodes=1                     # Nombre de nœuds
#SBATCH --ntasks=4                    # Nombre total de tâches (processus MPI)
#SBATCH --cpus-per-task=4            # Nombre de threads par tâche (pour OpenMP)
#SBATCH --mem=16G                     # Mémoire totale par nœud
##SBATCH --gres=gpu:1                # Décommenter pour demander 1 GPU

#===============================================================================
# PARAMÈTRES DE SORTIE
#===============================================================================
#SBATCH --job-name=job_test          # Nom du job
#SBATCH --output=logs/%j_%x.out      # Fichier de sortie (%j = JobID, %x = nom du job)
#SBATCH --error=logs/%j_%x.err       # Fichier d'erreur
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications par email
#SBATCH --mail-user=your@email.com   # Adresse email

#===============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT
#===============================================================================
# Désactiver l'export des variables d'environnement
#SBATCH --export=NONE

# Créer le répertoire de logs s'il n'existe pas
mkdir -p logs

# Charger les modules nécessaires
module purge                          # Nettoyer l'environnement
module load python/3.9                # Charger Python 3.9
# module load cuda/11.7                # Décommenter pour CUDA si GPU

# Configuration des variables d'environnement
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

#===============================================================================
# INFORMATIONS SUR LE JOB
#===============================================================================
echo "======================================"
echo "Date de début : $(date)"
echo "Nom du job : $SLURM_JOB_NAME"
echo "ID du job : $SLURM_JOB_ID"
echo "Nœud(s) utilisé(s) : $SLURM_JOB_NODELIST"
echo "Nombre de nœuds : $SLURM_JOB_NUM_NODES"
echo "Nombre de tâches : $SLURM_NTASKS"
echo "Nombre de CPU par tâche : $SLURM_CPUS_PER_TASK"
echo "======================================"

#===============================================================================
# EXÉCUTION DU PROGRAMME
#===============================================================================
# Définir le répertoire de travail
cd $SLURM_SUBMIT_DIR

# Exécuter le programme principal
python3 main.py

# Ou pour un programme MPI
# srun python3 main.py

#===============================================================================
# FIN DU JOB
#===============================================================================
echo "======================================"
echo "Date de fin : $(date)"
echo "======================================"

# Nettoyer si nécessaire
# rm -f temporary_files*