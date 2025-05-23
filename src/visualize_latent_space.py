import seaborn as sns
import matplotlib.pyplot as plt
from src import Simulation as sim

def lat_vis(df, LB):
    pd_df = sim.ABC_Monte_Carlo.lat_pos(df.unsqueeze(dim = 0), 3)
    # pd_df = sim.ABC_Monte_Carlo.lat_pos(df, 3)
    sns.scatterplot(pd_df, x = "dim_1", y = "dim_2", s = 2, hue = "group", legend = False)

    plt.xlim(LB, 1)
    plt.ylim(LB, 1)
    plt.axvline(x=0, alpha = 0.5)
    plt.axhline(y=0, alpha = 0.5)
    plt.plot([0,1], [1,0], alpha = 0.5)
    plt.gca().set_aspect('equal')
    # plt.savefig(fr"C:\Users\yangs\Desktop\making gif\scatterplot_{idx}.png")
    # plt.close()


def lat_vis_3d(df1, df2, LB):
    import plotly.graph_objects as go

    # Sample data (replace this with your torch tensor)
    data_1 = df1
    data_2 = df2


    vertices = [(0,0,0),(1, 0, 0), (0, 1, 0), (0, 0, 1)]

    # Define the four triangular surfaces
    surfaces = [
        [vertices[1], vertices[2], vertices[3]],
        [vertices[0], vertices[1], vertices[2]],
        [vertices[0], vertices[1], vertices[3]],
        [vertices[0], vertices[2], vertices[3]]
    ]

    # Create figure
    fig = go.Figure()

    for surface in surfaces:
        x, y, z = surface[0], surface[1], surface[2]
        fig.add_trace(go.Mesh3d(x=x, y=y, z=z, opacity=0.5, color = "green"))



    # Add 3D scatter plot
    fig.add_trace(go.Scatter3d(
        x=data_1[:, 0],
        y=data_1[:, 1],
        z=data_1[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color="red",
            opacity=0.4
        )
    ))

    fig.add_trace(go.Scatter3d(
        x=data_2[:, 0],
        y=data_2[:, 1],
        z=data_2[:, 2],
        mode='markers',
        marker=dict(
            size=1,
            color="blue",
            opacity=0.4
        )
    ))

    # Set axes range
    fig.update_layout(
        scene=dict(
            xaxis=dict(range=[LB, 1]),
            yaxis=dict(range=[LB, 1]),
            zaxis=dict(range=[LB, 1]),
            aspectmode='cube' 
        )
    )

    # Show plot
    fig.show()
