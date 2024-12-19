'''This script simulates the trajectory of the center of gravity of a dancer in a ballroom. 
You can enter the parameter of the music speed and of the room size.'''


import matplotlib.pyplot as plt
import numpy as np

def plot_waltz_trajectory(BPM, a_ellipse, b_ellipse) :

    # Paramètres de la musique
    small_circle_duration = 6*60/BPM
    print(f"Durée d'un petit cercle : {small_circle_duration:.2f} secondes")

    # Paramètres pour le mouvement circulaire (petit cercle)
    r_circle = 0.5 # rayon du petit cercle (en mètres)
    omega_circle = -2 * np.pi / small_circle_duration  # vitesse angulaire (rad/s)
    # omega_circle = -2 * np.pi * BPM / (6*60)  # vitesse angulaire (rad/s)

    # Paramètres pour le mouvement elliptique (grand cercle de bal)
    omega_ellipse = 2 * np.pi / 60  # vitesse pseudo-angulaire (rad/s)

    # Temps
    t = np.linspace(0, 120, 1000)  # 60 secondes, 1000 points

    # Mouvement circulaire
    x_circle = r_circle * np.cos(omega_circle * t)
    y_circle = r_circle * np.sin(omega_circle * t)

    # Mouvement elliptique
    x_ellipse = a_ellipse * np.cos(omega_ellipse * t)
    y_ellipse = b_ellipse * np.sin(omega_ellipse * t)

    # Composition des mouvements
    x_total = x_circle + x_ellipse
    y_total = y_circle + y_ellipse

    # Tracé de la trajectoire
    plt.figure(figsize=(10, 6))
    plt.plot(x_total, y_total, label='Trajectory of the center of gravity')
    plt.xlabel('x (m)')
    plt.ylabel('y (m)')
    plt.legend()
    plt.grid(True)
    plt.axis('equal')
    plt.savefig('waltz_trajectory.png')
    plt.close()


plot_waltz_trajectory(187, 5., 3.)