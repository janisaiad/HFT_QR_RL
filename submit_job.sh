#!/bin/bash

#===============================================================================
# Script de soumission SLURM générique
# Auteur: [Janis AIAD]
# Date: [2024-11-20]
#===============================================================================

#===============================================================================
# PARAMÈTRES SLURM OBLIGATOIRES
#===============================================================================
#SBATCH --partition=gpu_v100         # Partition à utiliser
#SBATCH --time=00:30:00              # Temps maximum d'exécution (format: jours-hh:mm:ss)
#SBATCH --account=lobib              # Nom du projet/compte

#===============================================================================
# RESSOURCES DEMANDÉES
#===============================================================================
#SBATCH --nodes=1                     # Nombre de nœuds
#SBATCH --ntasks=1                    # Nombre total de tâches (processus MPI)
#SBATCH --cpus-per-task=1            # Nombre de threads par tâche (pour OpenMP)
#SBATCH --mem=4096                     # Mémoire totale par nœud
#SBATCH --gres=gpu:1                  # Décommenter pour demander 1 GPU


#===============================================================================
# PARAMÈTRES DE SORTIE
#===============================================================================
#SBATCH --job-name=job_test_gpu        # Nom du job
#SBATCH --output=logs/%j_%x.out      # Fichier de sortie (%j = JobID, %x = nom du job)
#SBATCH --error=logs/%j_%x.err       # Fichier d'erreur
#SBATCH --mail-type=BEGIN,END,FAIL   # Notifications par email
#SBATCH --mail-user=janis.aiad@polytechnique.edu  # Adresse email

#===============================================================================
# CONFIGURATION DE L'ENVIRONNEMENT
#===============================================================================
# Désactiver l'export des variables d'environnement
#SBATCH --export=NONE

# Créer le répertoire de logs s'il n'existe pas
mkdir -p logs

# Charger les modules nécessaires
export UV_LINK_MODE=symlink
source ~/.bashrc
echo $UV_LINK_MODE


uv --link-mode=copysync
module spider python
module spider cuda
module purge                          # Nettoyer l'environnement
module load python/3.10               # Charger Python 3.10
module load cuda/11.7                # Décommenter pour CUDA si GPU

VENV_DIR="$SLURM_SUBMIT_DIR/.venv"
curl -LsSf https://astral.sh/uv/install.sh | sh

uv venv $VENV_DIR
source $VENV_DIR/bin/activate
uv sync
uv pip install -e .
uv pip install cupy
uv cache prune



uv run tests/test_env.py
source .venv/bin/activate



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
python3 tests/gpu_cholesky.py

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