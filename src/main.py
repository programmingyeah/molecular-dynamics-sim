#python time
from ursina import *
from ursina.shaders import lit_with_shadows_shader
import numpy as np
from atom import *

app = Ursina()
rng = np.random.default_rng()

SCALE = 1e-10  #one angstrom
CONTAINER_SIZE = 10*SCALE
SIGMA = 3e-10
LJ_CUTOFF = 2.5 * SIGMA
K_b = 1.380649e-23
EPSILON = 1e-21
N_ATOMS = 100
DT = 5*1e-15
K = 1

#generate N_ATOMS number of atoms at random positions within CONTAINER_SIZE
limit = CONTAINER_SIZE-max(ELEMENT_RADIUS_M.values())
atoms = []

while len(atoms) < N_ATOMS:
    position = rng.uniform(-limit, limit, size=3)

    if all(
        np.linalg.norm(position - atom.position) >= 0.9 * SIGMA
        for atom in atoms
    ):
        atoms.append(Atom(position))

#ursina setup
Entity.default_shader = lit_with_shadows_shader

sun = DirectionalLight(shadows=True)
sun.look_at(Vec3(1, -1, -1))

atom_entities = [Entity(model="sphere", color=color.red, scale=2 * atom.radius/SCALE) for atom in atoms]
container_entity = Entity(
    model="cube",
    scale=2 * CONTAINER_SIZE / SCALE,
    color=color.rgba(180, 180, 180, 100),
    wireframe=True,
    double_sided=True,
)
temperature_text = Text(
    text="Temperature: 0K",
    position=window.top_left + Vec2(0.02, -0.02),
    origin=(-0.5, 0.5),
    scale=1.2,
)

EditorCamera()

#functions
def calculate_forces(positions):
    forces = np.zeros((len(atoms), 3))
    for i in range(len(atoms)):
        for j in range(i+1, len(atoms)):
            #lennard jones potential U(r)=4eps * ((sigma/r)^12-(sigma/r)^6)
            r_vec = positions[i] - positions[j]
            r = np.linalg.norm(r_vec)
            if r==0 or r>=LJ_CUTOFF: continue
            force = 24*EPSILON/r**2 * (2*(SIGMA/r)**12 - (SIGMA/r)**6) * r_vec
            forces[i] += force
            forces[j] -= force

    return forces

forces = calculate_forces(np.array([atom.position for atom in atoms]))

def update():
    global forces

    masses = np.array([atom.mass for atom in atoms])
    positions = np.array([atom.position for atom in atoms])
    velocities = np.array([atom.velocity for atom in atoms])

    #verlet integration
    old_accelerations = forces / masses[:, np.newaxis]

    positions += velocities * DT + 0.5 * old_accelerations * DT**2

    #wall collisions
    for i, atom in enumerate(atoms):
        limit = CONTAINER_SIZE - atom.radius
        outside = np.abs(positions[i]) > limit

        velocities[i, outside] *= -1
        positions[i] = np.clip(positions[i], -limit, limit)

    forces = calculate_forces(positions)

    new_accelerations = forces / masses[:, np.newaxis]
    velocities += 0.5 * (old_accelerations + new_accelerations) * DT

    for atom, entity, position, velocity in zip(atoms, atom_entities, positions, velocities):
        atom.position = position.copy()
        atom.velocity = velocity.copy()
        entity.position = tuple(position / SCALE)

    #calculate the temperature
    kinetic_energy = sum(
        0.5 * atom.mass * np.linalg.norm(atom.velocity)**2 
        for atom in atoms
    )

    #in the future we'll be adding atoms in runtime so dont use N_ATOMS 
    temperature = (2 * kinetic_energy / (3 * len(atoms) * K_b))

    temperature_text.text = f"Temperature: {temperature:.1f}K"
            

if __name__ == "__main__":
    app.run()
