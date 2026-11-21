#python time
from ursina import *
import numpy as np
from atom import Atom

app = Ursina()
rng = np.random.default_rng()

SCALE = 10**-12
N_ATOMS = 5
DT = 10**-15
K = 1

#generate N_ATOMS number of atoms at random positions in a 10x10x10 box
atoms = [Atom(rng.uniform(-5.0, 5.0, size=3) * SCALE) for _ in range(N_ATOMS)]
masses = np.array([atom.mass for atom in atoms])

atom_entities = [Entity(model="sphere", color=color.red) for _ in atoms]

EditorCamera()

def calculate_forces(positions):
    forces = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            #springs between every atom, for now
            force = K * (positions[j] - positions[i])
            forces[i] += force
            forces[j] -= force

    return forces

forces = calculate_forces(np.array([atom.position for atom in atoms]))

def update():
    global forces

    positions = np.array([atom.position for atom in atoms])
    velocities = np.array([atom.velocity for atom in atoms])

    #verlet integration
    old_accelerations = forces / masses[:, np.newaxis]

    positions += velocities * DT + 0.5 * old_accelerations * DT**2

    forces = calculate_forces(positions)

    new_accelerations = forces / masses[:, np.newaxis]
    velocities += 0.5 * (old_accelerations + new_accelerations) * DT

    for atom, entity, position, velocity in zip(atoms, atom_entities, positions, velocities):
        atom.position = position.copy()
        atom.velocity = velocity.copy()
        entity.position = tuple(position / SCALE)

if __name__ == "__main__":
    app.run()
