# Reinforcement Learning Application Project for Optimal Control of Reactive QR Queues in High Frequency Trading

## Project Description

This project aims to apply reinforcement learning techniques to optimize the control of reactive QR (Quick Response) queues in a high-frequency trading context. It combines advanced concepts of machine learning, queueing theory, and optimization to improve resource management and performance of trading systems.

## Project Objectives

1. Model reactive QR queues in a high-frequency trading environment
2. Implement reinforcement learning algorithms for optimal control
3. Evaluate and optimize system performance in terms of execution time and resource utilization
4. Visualize results and performance metrics

## Project Structure

The project is organized into several key components:

1. `model.py`: Implementation of the QR queue model
2. `rl_agent.py`: Reinforcement learning agent
3. `environment.py`: Simulation environment for agent-model interaction
4. `visualization.py`: Results visualization module
5. `main.py`: Main script for running experiments

## Installation and Setup

1. Clone the repository:
   ```
   git clone https://github.com/janisaiad/qr-queue-rl-control-hft.git
   cd qr-queue-rl-control-hft
   ```

2. Utilisez le Makefile pour configurer l'environnement et installer les dépendances :
   ```
   make setup
   ```

3. Pour installer les dépendances avec Poetry :
   ```
   make install
   ```

## Usage

1. Pour exécuter le script principal :
   ```
   make run
   ```

2. Pour télécharger les données nécessaires :
   ```
   make data
   ```

3. Pour générer les visualisations :
   ```
   make visualize
   ```

4. Pour nettoyer l'environnement et les fichiers générés :
   ```
   make clean
   ```

5. Pour afficher l'aide du Makefile :
   ```
   make help
   ```

6. Pour vérifier la présence du fichier LICENSE :
   ```
   make check-license
   ```

7. Pour afficher le contenu du README :
   ```
   make show-readme
   ```

Les résultats seront générés dans le dossier `databento/`.

## Main Components

### model.py

This file contains the implementation of the reactive QR queue model. It simulates the behavior of trading orders and their processing in a multi-queue system.

### rl_agent.py

Implements the reinforcement learning agent using algorithms such as Q-Learning or Deep Q-Network (DQN) to learn the optimal control policy.

### environment.py

Defines the simulation environment in which the agent interacts with the queue model. It manages states, actions, and rewards.

### visualization.py

Module responsible for generating graphs and visualizations to analyze system performance and learning results.

### main.py

Main script that orchestrates the entire process, from initialization to running experiments and generating results.

## Results and Analysis

Experiment results will be stored in the `databento/` folder. They will include:

- Learning convergence graphs
- System performance metrics (average response time, resource utilization, etc.)
- Visualizations of learned policies

## Contribution

Contributions to this project are welcome. Please follow these steps to contribute:

1. Fork the project
2. Create a branch for your feature (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

Project Link: [https://github.com/janisaiad/qr-queue-rl-control-hft](https://github.com/janisaiad/qr-queue-rl-control-hft)
