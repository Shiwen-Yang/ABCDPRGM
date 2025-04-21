# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# import matplotlib.widgets as widgets

# # Define the mollified ReLU function psi_tau
# def psi_tau(x, tau):
#     return norm.pdf(x / tau) * tau + x * norm.cdf(x / tau)

# # Generate x values
# x = np.linspace(-2, 2, 500)

# # Initial tau value
# initial_tau = 0.3
# y = psi_tau(x, initial_tau)

# # Create the plot
# fig, ax = plt.subplots(figsize=(8, 5))
# plt.subplots_adjust(left=0.1, bottom=0.25)
# line, = ax.plot(x, y, label=f'τ = {initial_tau}')
# ax.set_title('Mollified ReLU function $\\psi_\\tau(x)$')
# ax.set_xlabel('x')
# ax.set_ylabel('$\\psi_\\tau(x)$')
# ax.grid(True)
# ax.set_ylim(-0.5, 2.5)
# legend = ax.legend()

# # Add a slider for tau
# ax_tau = plt.axes([0.1, 0.1, 0.8, 0.05])
# tau_slider = widgets.Slider(ax_tau, 'τ', 0.01, 1.0, valinit=initial_tau, valstep=0.01)

# # Update function for slider
# def update(val):
#     tau = tau_slider.val
#     y = psi_tau(x, tau)
#     line.set_ydata(y)
#     legend.texts[0].set_text(f'τ = {tau:.2f}')
#     fig.canvas.draw_idle()

# tau_slider.on_changed(update)

# plt.show()







# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# from matplotlib.widgets import Slider

# # Define the mollified ReLU function psi_tau
# def psi_tau(x, tau):
#     return norm.pdf(x / tau) * tau + x * norm.cdf(x / tau)

# # Define the loss function for a 2D input
# def loss_2d(x1, x2, sigma):
#     relu_penalty = psi_tau(-x1, sigma) + psi_tau(-x2, sigma)
#     sum_penalty = psi_tau(x1 + x2 - 1, np.sqrt(2) * sigma)
#     return relu_penalty + sum_penalty

# # Create grid
# x1_vals = np.linspace(-0.2, 1.2, 300)
# x2_vals = np.linspace(-0.2, 1.2, 300)
# X1, X2 = np.meshgrid(x1_vals, x2_vals)

# # Initial sigma
# initial_sigma = 0.2
# Z = loss_2d(X1, X2, initial_sigma)

# # Plot setup
# fig, ax = plt.subplots(figsize=(7, 6))
# plt.subplots_adjust(left=0.1, bottom=0.25)
# cmap = ax.pcolormesh(X1, X2, Z, shading='auto', cmap='viridis')
# fig.colorbar(cmap, ax=ax)
# ax.set_title('Loss Landscape in $\\mathbb{R}^2$')
# ax.set_xlabel('$x_1$')
# ax.set_ylabel('$x_2$')

# # Slider axis
# ax_sigma = plt.axes([0.1, 0.1, 0.8, 0.05])
# sigma_slider = Slider(ax_sigma, 'σ', 0.01, 0.5, valinit=initial_sigma, valstep=0.01)

# # Update function
# def update(val):
#     sigma = sigma_slider.val
#     Z = loss_2d(X1, X2, sigma)
#     cmap.set_array(Z.ravel())
#     cmap.set_clim(vmin=Z.min(), vmax=Z.max())
#     fig.canvas.draw_idle()

# sigma_slider.on_changed(update)
# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# from matplotlib.widgets import Slider

# # Mollified ReLU function
# def psi_tau(x, tau):
#     return norm.pdf(x / tau) * tau + x * norm.cdf(x / tau)

# # 2D loss function
# def loss_2d(x1, x2, sigma):
#     relu_penalty = psi_tau(-x1, sigma) + psi_tau(-x2, sigma)
#     sum_penalty = psi_tau(x1 + x2 - 1, np.sqrt(2) * sigma)
#     # sum_penalty = 0
#     return relu_penalty + sum_penalty

# # 1D slice orthogonal to the all-ones vector
# def orthogonal_slice(t, offset, sigma):
#     # direction vector orthogonal to [1, 1]: e.g., [1, -1]
#     dir_vec = np.array([1, -1])
#     dir_vec = dir_vec / np.linalg.norm(dir_vec)
#     # slice: offset * 1 + t * dir_vec
#     x = offset + t * dir_vec
#     return loss_2d(x[0], x[1], sigma)

# # Create 1D t values for the slice
# t_vals = np.linspace(-1.0, 1.0, 500)
# initial_sigma = 0.2
# initial_offset = 0

# # Initial loss values
# loss_vals = np.array([orthogonal_slice(t, initial_offset, initial_sigma) for t in t_vals])

# # Plot setup
# fig, ax = plt.subplots(figsize=(8, 5))
# plt.subplots_adjust(left=0.1, bottom=0.3)
# line, = ax.plot(t_vals, loss_vals, label='Loss along orthogonal slice')
# ax.set_title('1D Slice of Loss Landscape (Orthogonal to $\\mathbf{1}$)')
# ax.set_xlabel('Slice parameter $t$')
# ax.set_ylabel('Loss')
# ax.grid(True)
# ax.legend()

# # Sliders
# ax_sigma = plt.axes([0.1, 0.18, 0.8, 0.05])
# sigma_slider = Slider(ax_sigma, 'σ', 0.01, 1.0, valinit=initial_sigma, valstep=0.01)

# ax_offset = plt.axes([0.1, 0.1, 0.8, 0.05])
# offset_slider = Slider(ax_offset, 'Offset', -0.5, 1.2, valinit=initial_offset, valstep=0.01)

# # Update function
# def update(val):
#     sigma = sigma_slider.val
#     offset = offset_slider.val
#     new_loss_vals = np.array([orthogonal_slice(t, offset, sigma) for t in t_vals])
#     line.set_ydata(new_loss_vals)
#     ax.set_ylim(new_loss_vals.min() - 0.1, new_loss_vals.max() + 0.1)
#     fig.canvas.draw_idle()

# sigma_slider.on_changed(update)
# offset_slider.on_changed(update)

# plt.show()


# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# from matplotlib.widgets import Slider

# # Define the mollified ReLU function psi_tau
# def psi_tau(x, tau):
#     return norm.pdf(x / tau) * tau + x * norm.cdf(x / tau)

# # Define the 1D loss function
# def loss_1d(c, p, sigma):
#     relu_term = p * psi_tau(-c, sigma)
#     sum_term = psi_tau(p * c - 1, np.sqrt(p) * sigma)
#     return relu_term + sum_term

# # Domain of c values
# c_vals = np.linspace(-0.5, 1.5, 500)
# initial_p = 2
# initial_sigma = 0.2
# loss_vals = [loss_1d(c, initial_p, initial_sigma) for c in c_vals]

# # Plot setup
# fig, ax = plt.subplots(figsize=(8, 5))
# plt.subplots_adjust(left=0.1, bottom=0.3)
# line, = ax.plot(c_vals, loss_vals, label='1D loss function')
# ax.set_title('1D Optimization of Reparametrized Loss')
# ax.set_xlabel('$c$')
# ax.set_ylabel('Loss')
# ax.grid(True)
# ax.legend()

# # Sliders
# ax_sigma = plt.axes([0.1, 0.18, 0.8, 0.05])
# sigma_slider = Slider(ax_sigma, 'σ', 0.01, 1.0, valinit=initial_sigma, valstep=0.01)

# ax_p = plt.axes([0.1, 0.1, 0.8, 0.05])
# p_slider = Slider(ax_p, 'p', 2, 20, valinit=initial_p, valstep=1)

# # Redefine the update function to include annotation of the minimum loss and its location
# def update(val):
#     sigma = sigma_slider.val
#     p = int(p_slider.val)
#     new_loss_vals = [loss_1d(c, p, sigma) for c in c_vals]
#     line.set_ydata(new_loss_vals)
    
#     # Find the minimum loss and corresponding c
#     min_index = np.argmin(new_loss_vals)
#     min_c = c_vals[min_index]
#     min_loss = new_loss_vals[min_index]
    
#     # Update plot limits and title
#     ax.set_ylim(min(new_loss_vals) - 0.1, max(new_loss_vals) + 0.1)
    
#     # Remove old text if it exists
#     [child.remove() for child in ax.get_children() if isinstance(child, plt.Annotation)]
    
#     # Add annotation for minimum point
#     ax.annotate(f'Min at c={min_c:.3f}, loss={min_loss:.3f}', 
#                 xy=(min_c, min_loss), 
#                 xytext=(min_c, min_loss + 0.1),
#                 # arrowprops=dict(facecolor='black', shrink=0.05),
#                 fontsize=10, ha='center')
    
#     fig.canvas.draw_idle()

# # Reconnect the updated function to the sliders
# sigma_slider.on_changed(update)
# p_slider.on_changed(update)

# # Trigger initial update to display the annotation
# update(None)

# plt.show()




# import numpy as np
# import matplotlib.pyplot as plt
# from scipy.stats import norm
# from matplotlib.widgets import Slider

# def tighter_lambda(y, p, sigma):
#     phi_i = norm.pdf(-y / sigma)
#     phi_s = norm.pdf((p * y - 1) / (np.sqrt(p) * sigma))
#     avg_phi_i = phi_i  # since y is scalar here
#     return (avg_phi_i / sigma) + (np.sqrt(p) / sigma) * phi_s

# # Define the function
# def f(y, p, sigma):
#     Phi_i = norm.cdf(-y / sigma)
#     Phi_s = norm.cdf((p * y - 1) / (np.sqrt(p) * sigma))
#     lambda_y = tighter_lambda(y, p, sigma)
#     return lambda_y * y**2 + (Phi_i - Phi_s) * y





# # Initial values
# p0 = 3.0
# sigma0 = 1.0
# y_vals = np.linspace(-5, 5, 500)

# # Plot setup
# fig, ax = plt.subplots()
# plt.subplots_adjust(bottom=0.3)
# curve, = plt.plot(y_vals, f(y_vals, p0, sigma0), lw=2)
# ax.set_title("Plot of $f(y) = \\lambda y^2 + (\\Phi_i - \\Phi_s) y$")
# ax.set_xlabel("y")
# ax.set_ylabel("f(y)")
# ax.grid(True)

# # Slider axes
# ax_p = plt.axes([0.2, 0.15, 0.65, 0.03])
# ax_sigma = plt.axes([0.2, 0.1, 0.65, 0.03])

# # Sliders
# slider_p = Slider(ax_p, r"$p$", valmin=1.0, valmax=10.0, valinit=p0)
# slider_sigma = Slider(ax_sigma, r"$\sigma$", valmin=0.01, valmax=5.0, valinit=sigma0)

# # Update function
# def update(val):
#     p = slider_p.val
#     sigma = slider_sigma.val
#     curve.set_ydata(f(y_vals, p, sigma))
#     fig.canvas.draw_idle()

# # Connect update to sliders
# slider_p.on_changed(update)
# slider_sigma.on_changed(update)

# plt.show()
