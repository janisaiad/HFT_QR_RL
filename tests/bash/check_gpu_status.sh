#!/bin/bash

#===============================================================================
# Script pour vérifier l'état des GPUs
#===============================================================================

echo "=== État des partitions GPU ==="
sinfo -p gpu,gpu_v100,gpu_a100,gpu_a100_80g

echo -e "\n=== Jobs en cours sur les GPUs ==="
squeue -p gpu,gpu_v100,gpu_a100,gpu_a100_80g

echo -e "\n=== Détails des nœuds GPU ==="
for node in $(sinfo -p gpu -h -o "%N"); do
    echo -e "\nNœud: $node"
    scontrol show node $node
done

# Pour obtenir les informations nvidia-smi, il faut se connecter à un nœud:
echo -e "\n=== Pour obtenir les détails GPU (nvidia-smi) ==="
echo "Utilisez une des commandes suivantes:"
echo "1. Pour V100:   srun --partition=gpu_v100 --gres=gpu:1 --time=00:05:00 --pty nvidia-smi"
echo "2. Pour A100:   srun --partition=gpu_a100 --gres=gpu:1 --time=00:05:00 --pty nvidia-smi"
echo "3. Pour A100 80GB: srun --partition=gpu_a100_80g --gres=gpu:1 --time=00:05:00 --pty nvidia-smi"