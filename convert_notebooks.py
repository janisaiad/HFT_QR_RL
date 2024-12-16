#!/usr/bin/env python3

import os
import jupytext
from pathlib import Path

def convert_notebooks():
    """
    Parcourt récursivement tous les fichiers .ipynb et les convertit en .py
    en utilisant jupytext, en conservant la même structure de dossiers.
    """
    # Obtenir le chemin absolu du répertoire courant
    root_dir = Path().absolute()
    
    # Parcourir récursivement tous les fichiers
    for ipynb_path in root_dir.rglob("*.ipynb"):
        # Ignorer les fichiers dans .ipynb_checkpoints
        if ".ipynb_checkpoints" in str(ipynb_path):
            continue
            
        try:
            # Créer le chemin pour le fichier py correspondant
            py_path = ipynb_path.with_name(f"{ipynb_path.stem}_jpy.py")
            
            print(f"Converting {ipynb_path} to {py_path}")
            
            # Lire le notebook
            notebook = jupytext.read(str(ipynb_path))
            
            # Convertir et sauvegarder en format py
            jupytext.write(notebook, str(py_path), fmt="py:percent")
            
            print(f"Successfully converted {ipynb_path.name}")
            
        except Exception as e:
            print(f"Error converting {ipynb_path.name}: {str(e)}")

if __name__ == "__main__":
    convert_notebooks() 