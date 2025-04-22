## 📘 About the Model

**ABCDPRGM** (*Attractor-Based Coevolving Dot Product Random Graph Model*) is a dynamic latent space model designed to capture evolving network behaviors such as **flocking** or **polarization**. It extends the classic **Random Dot Product Graph (RDPG)** framework by introducing group-based attractors that guide the movement of node latent positions over time.

## 🧪 How to Reproduce the Results

To reproduce the main experiments and figures from the paper, start with the Jupyter notebook:

**`Gen_Sim_Data.ipynb`**

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
