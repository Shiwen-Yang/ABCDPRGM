import torch
import matplotlib.pyplot as plt
from torch.distributions import Normal, Dirichlet, Uniform
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import os
import sys
from matplotlib.widgets import Slider

class ReLUPenalty2D(nn.Module):
    def __init__(self, sigma, device='cuda'):
        super().__init__()
        self.sigma = sigma
        self.device = device
        self.theta = nn.Parameter(torch.tensor(0.0, device=device, dtype=torch.float64))

    def rotation_matrix(self):
        c = torch.cos(self.theta)
        s = torch.sin(self.theta)
        return torch.stack([
            torch.stack([c, -s]),
            torch.stack([s,  c])
        ])

    def forward(self, Z):
        relu = nn.ReLU()
        
        Z = Z.to(self.device)
        V = self.rotation_matrix()
        scaled = Z @ V.T / self.sigma
        
        normal = Normal(0, 1)
        phi = normal.log_prob(-scaled).exp()
        Phi = normal.cdf(-scaled)
        
        loss_per_row_0 = torch.sum(phi - scaled * Phi, dim=1) * self.sigma
        loss_per_row_1 = torch.sum(relu(-Z @ V.T), dim=1)
        
        total_loss_0 = loss_per_row_0.sum()
        total_loss_1 = loss_per_row_1.sum()
        
        return total_loss_0, total_loss_1

def shortest_distance_to_y_eq_x(point):
    point = np.squeeze(np.array(point))
    x0, y0 = point
    return abs(x0 - y0) / np.sqrt(2)

def plot_loss_landscape_with_slider(Y, sigma=0.02, num_points=1000, theta_min = -torch.pi, theta_max = torch.pi):
    thetas = torch.linspace(theta_min, theta_max, num_points, dtype=torch.float64)
    losses_0 = []
    losses_1 = []

    model = ReLUPenalty2D(sigma=sigma)

    for theta_val in thetas:
        model.theta.data = theta_val.clone().to(model.device)
        loss_val_0, loss_val_1 = model.forward(Y)
        losses_0.append(loss_val_0.item())
        losses_1.append(loss_val_1.item())

    # Setup figure with 1x2 layout
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plt.subplots_adjust(bottom=0.25)

    # Plot both losses on the same subplot
    axes[0].plot(thetas.numpy(), losses_0, label='Gaussian_Conditional', color='blue')
    axes[0].plot(thetas.numpy(), losses_1 , label='ReLU_Raw', color='green')
    axes[0].axhline(y=0, color='black', linestyle='--', linewidth=1)
    current_theta_line = axes[0].axvline(x=0.0, color='red', linestyle='-', linewidth=1.5, label='Current θ')
    axes[0].set_xlabel("θ (radians)")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Loss Landscape of ReLUPenalty2D")
    axes[0].legend()
    axes[0].grid(True)

    # Scatter plot of rotated input
    scatter = axes[1].scatter(Y[:, 0].numpy(), Y[:, 1].numpy(), alpha=1, s=20)
    scatter_mean = axes[1].scatter(*Y.mean(dim=0).numpy(), color='red', s=60)


    axes[1].axhline(y=0, color='black', linestyle='--', linewidth=1)
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].set_title("Input Data (Rotated by θ)")
    axes[1].set_xlabel("Y[:, 0]")
    axes[1].set_ylabel("Y[:, 1]")
    axes[1].axis('equal')
    axes[1].set_xlim(-0.5, 1.5)
    axes[1].set_ylim(-0.5, 1.5)
    axes[1].legend(title=f'Dist(mean(Y), y=x) ≈')
    axes[1].grid(True)

    # Slider
    ax_slider = plt.axes([0.25, 0.08, 0.5, 0.03])
    theta_slider = Slider(ax_slider, "θ (radians)", valmin=thetas.min().item(), valmax=thetas.max().item(), valinit=0.0)

    def rotate_points(Y, theta):
        c = torch.cos(theta)
        s = torch.sin(theta)
        rotation_matrix = torch.tensor([[c, -s], [s, c]], dtype=torch.float64)
        return Y @ rotation_matrix.T

    def update(val):
        theta = torch.tensor(theta_slider.val, dtype=torch.float64)
        Y_rotated = rotate_points(Y, theta)
        Y_mean_rotated = rotate_points(Y.mean(dim=0, keepdim=True), theta)
        
        scatter.set_offsets(Y_rotated.numpy())
        scatter_mean.set_offsets(Y_mean_rotated.numpy())
        current_theta_line.set_xdata([theta.item(), theta.item()])
        
        distance = shortest_distance_to_y_eq_x(Y_mean_rotated)
        new_label = f'Dist(mean(Y), y=x) ≈ {distance:.4f}'
        axes[1].legend(title=new_label)
        fig.canvas.draw_idle()

    theta_slider.on_changed(update)

    plt.show()

    best_theta = thetas[torch.argmin(torch.tensor(losses_0))]
    return min(losses_0), best_theta



if __name__ == "__main__":
    alpha = torch.tensor([[10, 1, 1], [1, 10, 1]], dtype= torch.float64)
    # alpha = torch.tensor([[5, 1]], dtype= torch.float64)
    n= 2
    K, p = alpha.shape
    torch.manual_seed(5)
    dir = Dirichlet(alpha)
    X = dir.sample((n // K,)).transpose(0, 1).reshape(n, p)[:, :2]

    sigma = torch.tensor(0.05, dtype = torch.float64)
    normal = Normal(0, sigma)
    noise = normal.sample((n, p - 1))
    Y = X + noise
    plot_loss_landscape_with_slider(Y, sigma, 5000, -torch.pi/4, torch.pi/4)