## 📘 About the Model

**ABCDPRGM** (*Attractor-Based Coevolving Dot Product Random Graph Model*) is a dynamic latent space model designed to capture evolving network behaviors such as **flocking** or **polarization**. It extends the classic **Random Dot Product Graph (RDPG)** framework by introducing group-based attractors that guide the movement of node latent positions over time.

### Core Structure and Attractors🧲

Each node $i$ is assigned a latent position $Z_{i,t} \in \mathbb{R}^p$, constrained so that inner products \( Z_{i,t}^T Z_{j,t} \in [0, 1] \). At each time point \( t \), a symmetric adjacency matrix \( Y_t \) is generated such that:
\[
Y_{ij,t} \sim \text{Bernoulli}(Z_{i,t}^T Z_{j,t})
\]
This defines an **RDPG at each time step**, where edges are conditionally independent given the latent positions.

In addition, each node belongs to one of \( K \) groups, and at each time step, node \( i \)'s neighbors influence its position via two attractors:
- **Intra-group attractor** \( A^w_{i,t} \): mean latent position of same-group neighbors
- **Inter-group attractor** \( A^b_{i,t} \): mean latent position of different-group neighbors

These attractors are calculated using the observed adjacency matrix at time \( t \) and the latent positions of neighboring nodes.

### Latent Position Evolution

The model assumes latent positions evolve via a **Dirichlet Generalized Linear Model (GLM)** with a log-link function. Specifically, the lifted latent position \( Z^*_{i,t} \in \mathbb{H}^{p+1} \) evolves as:
\[
Z^*_{i,t+1} \sim \text{Dirichlet}\left( \exp\left[\beta_1 Z^*_{i,t} + \beta_2 A^{w*}_{i,t} + \beta_3 A^{b*}_{i,t} + \beta_4 \right] \right)
\]
Here:
- \( $\beta_1$ \): self-inertia (past influence)
- \( \beta_2 \): same-group attractor strength
- \( \beta_3 \): different-group attractor strength
- \( \beta_4 \): intercept term (controls variance)

The inclusion of the log-link allows for both positive and negative influences, while the Dirichlet distribution ensures latent positions remain on the simplex.