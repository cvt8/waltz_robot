import matplotlib.pyplot as plt
import pandas as pd
import matplotlib.cm as cm
import numpy as np


def plot_element_positions(element, start_frame=35, cutoff_at_end=50):
    # Read element positions from file
    file = 'element_positions.csv'
    data = pd.read_csv(file)
    positions = data[element].values.tolist()
    
    # Convert strings to arrays of floats
    positions = [
        [float(coord) for coord in position.strip('[]').split()] 
        for position in positions
    ]
    positions = positions[start_frame: len(positions) - cutoff_at_end]
    print(positions)

    # Extract x and y positions
    x_positions = [position[0] for position in positions]
    y_positions = [position[1] for position in positions]


    # Create a color scale based on the sequence of points
    colors = cm.viridis(np.linspace(0, 1, len(x_positions)))

    # Plot the trajectory with a color gradient
    plt.scatter(x_positions, y_positions, c=colors, cmap='viridis', s=10)
    plt.plot(x_positions, y_positions, color='gray', alpha=0.5)
    plt.colorbar(label='Progression (time)')
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Positions of {element}')
    plt.savefig(f'{element}_positions_with_color_scale.png')

    # Plot the trajectory with arrows
    plt.figure(figsize=(8, 6))
    for i in range(len(x_positions) - 1):
        plt.arrow(
            x_positions[i],
            y_positions[i],
            x_positions[i + 1] - x_positions[i],
            y_positions[i + 1] - y_positions[i],
            color='blue',
            head_width=0.01,
            length_includes_head=True,
            alpha=0.7
        )
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(f'Positions of {element}')
    plt.savefig(f'{element}_positions_with_arrows.png')
    plt.show()

if __name__ == '__main__':
    plot_element_positions('Left Hand')