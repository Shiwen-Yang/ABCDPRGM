## 📘 About the Model

**ABCDPRGM** (*Attractor-Based Coevolving Dot Product Random Graph Model*) is a dynamic latent space model designed to capture evolving network behaviors such as **flocking** or **polarization**. It extends the classic **Random Dot Product Graph (RDPG)** framework by introducing group-based attractors that guide the movement of node latent positions over time.

## 🧪 How to Reproduce the Results

To reproduce the main synthetic experiments and figures from the paper, use the Jupyter notebook:

**[`Gen_Sim_Data.ipynb`](https://github.com/Shiwen-Yang/ABCDPRGM/blob/main/Gen_Sim_Data.ipynb)**

This notebook will:
- Generate synthetic dynamic networks based on the ABCDPRGM model
- Simulate polarizing or flocking behavior under different parameter settings
- Estimate the parameters $(\beta_1, \beta_2, \beta_3, \beta_4)$ from the generated graphs
- Output trajectory plots and other visualizations used in the paper

You can modify values such as:
- Number of nodes
- Number of time steps
- Initial Dirichlet parameters
- Influence coefficients (β)

No additional configuration files are required. All dependencies are standard Python scientific packages (see requirements.txt).

### 📊 Real-World Data Analysis (Age of Empires IV)

To replicate the real-world data analysis presented in the paper, use the notebook:

**[`aoe.ipynb`](https://github.com/Shiwen-Yang/ABCDPRGM/blob/main/aoe.ipynb)**

This notebook performs the following:

- **Loads and preprocesses** ranked match data from Age of Empires IV
- **Constructs time-series networks** based on player interactions
- **Estimates model parameters** $(\beta_1, \beta_2, \beta_3, \beta_4)$ using the ABCDPRGM framework
- **Visualizes** the evolution of latent positions to identify flocking or polarization behaviors

**Note:** The dataset is sourced from [aoe4world.com/dumps](https://aoe4world.com/dumps). Please ensure you download the appropriate data corresponding to the time frame specified in the paper.

*Disclaimer:* The data is provided by aoe4world under Microsoft's "Game Content Usage Rules" using assets from Age of Empires IV, and it is not endorsed by or affiliated with Microsoft.
