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
   git clone https://github.com/janisaiad/HFT_QR_RL.git
   cd HFT_QR_RL
   ```

2. Install Poetry (if not already installed):
   ```
   curl -sSL https://install.python-poetry.org | python3 -
   ```

3. Use Poetry to set up the environment and install dependencies:
   ```
   poetry install
   ```

4. Activate the Poetry shell:
   ```
   poetry shell
   ```

## Usage

1. To run the main script:
   ```
   poetry run python main.py
   ```

2. To download necessary data:
   ```
   poetry run python data/script.py
   ```

3. To generate visualizations:
   ```
   poetry run python data/visualization.py
   ```

4. To clean the environment and generated files:
   ```
   poetry run python clean.py
   ```

5. To display project information:
   ```
   poetry show
   ```

6. To update dependencies:
   ```
   poetry update
   ```

7. To add a new dependency:
   ```
   poetry add <package_name>
   ```

Results will be generated in the `databento/` folder.

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
2. Create a development branch for your feature (`git checkout -b dev/NewFeature`)
3. Make your changes and commit them (`git commit -m 'Add NewFeature'`)
4. Push to the branch (`git push origin dev/NewFeature`)
5. Open a Pull Request to the `dev` branch

### Development Practices

- Always use the `dev` branch for ongoing development.
- Ensure your commits are clear and descriptive.
- Before submitting a Pull Request, make sure your code is well-commented and follows the project's style conventions.
- For technical discussions or questions, contact janis.aiad@polytechnique.edu.

## License

This project is licensed under the MIT License. See the `LICENSE` file for more details.

## Contact

Project Link: [https://github.com/janisaiad/HFT_QR_RL](https://github.com/janisaiad/HFT_QR_RL)  

For any questions or suggestions, please contact: janis.aiad@polytechnique.edu
