# Reinforcement Learning Application Project for Optimal Control of Reactive QR Queues in High Frequency Trading

## Project Description

This project aims to apply reinforcement learning techniques to optimize the control of reactive QR (Quick Response) queues in a high-frequency trading context. It combines advanced concepts of machine learning, queueing theory, and optimization to improve resource management and performance of trading systems.


### GPU Support

This project uses the `polars` library, which supports GPU acceleration. Install cudf manually to use GPU.
// For RTX4060  with cudf/cuml :: 
Successfully installed MarkupSafe-3.0.2 cachetools-5.5.0 click-8.1.7 cloudpickle-3.1.0 cuda-python-12.6.2.post1 cudf-cu12-24.10.1 cuml-cu12-24.10.0 cupy-cuda12x-13.3.0 cuvs-cu12-24.10.0 dask-2024.9.0 dask-cuda-24.10.0 dask-cudf-cu12-24.10.1 dask-expr-1.1.14 distributed-2024.9.0 distributed-ucxx-cu12-0.40.0 fastrlock-0.8.2 fsspec-2024.10.0 jinja2-3.1.4 joblib-1.4.2 libcudf-cu12-24.10.1 libucx-cu12-1.17.0 libucxx-cu12-0.40.0 llvmlite-0.43.0 locket-1.0.0 markdown-it-py-3.0.0 mdurl-0.1.2 msgpack-1.1.0 numba-0.60.0 numpy-2.0.2 nvidia-cublas-cu12-12.6.3.3 nvidia-cufft-cu12-11.3.0.4 nvidia-curand-cu12-10.3.7.77 nvidia-cusolver-cu12-11.7.1.2 nvidia-cusparse-cu12-12.5.4.2 nvidia-nvjitlink-cu12-12.6.77 nvtx-0.2.10 pandas-2.2.2 partd-1.4.2 psutil-6.1.0 pyarrow-17.0.0 pygments-2.18.0 pylibcudf-cu12-24.10.1 pylibraft-cu12-24.10.0 pynvjitlink-cu12-0.4.0 pynvml-11.4.1 python-dateutil-2.9.0.post0 pytz-2024.2 pyyaml-6.0.2 raft-dask-cu12-24.10.0 rapids-dask-dependency-24.10.0 rich-13.9.4 rmm-cu12-24.10.0 scipy-1.14.1 six-1.16.0 sortedcontainers-2.4.0 tblib-3.0.0 toolz-1.0.0 tornado-6.4.1 treelite-4.3.0 typing_extensions-4.12.2 tzdata-2024.2 ucx-py-cu12-0.40.0 ucxx-cu12-0.40.0 zict-3.0.0

worked :
uv pip install --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match cudf-cu12
and for timeout 
UV_HTTP_TIMEOUT=300 uv pip install --index-url https://pypi.org/simple --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match cudf-cu12


### New Libraries

- **NumPy**: Utilisé pour le calcul numérique et le traitement des données.
- **Scikit-learn**: Utilisé pour les algorithmes d'apprentissage automatique et l'évaluation des modèles.

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

2. Grant execution rights to the launch script:
   ```
   chmod +x launch.sh
   ```

3. Launch the environment:
   ```
   ./launch.sh
   ```

## Usage with Makefile

1. To set up the environment and install dependencies:
   ```
   @make setup
   ```

2. To run the main script:
   ```
   @make run
   ```

3. To download necessary data:
   ```
   @make data
   ```

4. To generate visualizations:
   ```
   @make visualize
   ```

5. To clean the environment and generated files:
   ```
   @make clean
   ```

6. To display help information:
   ```
   @make help
   ```

7. To check the license:
   ```
   @make check-license
   ```

8. To show the README content:
   ```
   @make show-readme
   ```

## Usage with Poetry

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

4. To display project information:
   ```
   poetry show
   ```

5. To update dependencies:
   ```
   poetry update
   ```

6. To add a new dependency:
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
