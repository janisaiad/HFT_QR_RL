# Makefile pour le projet HFT_QR_RL

# Variables
PYTHON = python3
VENV = venv
PIP = $(VENV)/bin/pip
POETRY = poetry

# Cibles par défaut
.PHONY: all
all: setup run

# Configuration de l'environnement
.PHONY: setup
setup: $(VENV)/bin/activate

$(VENV)/bin/activate: requirements.txt
	$(PYTHON) -m venv $(VENV)
	$(PIP) install -r requirements.txt

# Installation des dépendances avec Poetry
.PHONY: install
install:
	$(POETRY) install

# Exécution du script principal
.PHONY: run
run:
	$(PYTHON) main.py

# Nettoyage
.PHONY: clean
clean:
	rm -rf $(VENV)
	find . -type f -name '*.pyc' -delete
	find . -type d -name '__pycache__' -delete

# Affichage de l'aide
.PHONY: help
help:
	@echo "Utilisation du Makefile:"
	@echo "  make setup    : Crée l'environnement virtuel et installe les dépendances"
	@echo "  make install  : Installe les dépendances avec Poetry"
	@echo "  make run      : Exécute le script principal"
	@echo "  make clean    : Nettoie l'environnement virtuel et les fichiers .pyc"
	@echo "  make all      : Exécute setup et run"
	@echo "  make help     : Affiche ce message d'aide"
	@echo "  make check-license : Vérifie la présence du fichier LICENSE"
	@echo "  make show-readme   : Affiche le contenu du README.md"
	@echo "  make data     : Exécute le script de téléchargement des données"
	@echo "  make visualize: Génère les visualisations"

# Vérification de la licence
.PHONY: check-license
check-license:
	@if [ -f LICENSE ]; then \
		echo "Licence trouvée : "; \
		head -n 1 LICENSE; \
	else \
		echo "Attention : Fichier LICENSE manquant!"; \
	fi

# Affichage du README
.PHONY: show-readme
show-readme:
	@if [ -f README.md ]; then \
		echo "Contenu du README :"; \
		cat README.md; \
	else \
		echo "Attention : Fichier README.md manquant!"; \
	fi

# Exécution du script de téléchargement des données
.PHONY: data
data:
	$(PYTHON) data/script.py

# Génération des visualisations
.PHONY: visualize
visualize:
	$(PYTHON) data/vizualization.py
