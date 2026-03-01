from yade import pack, qt
from yade import *
from math import radians

# --------------------------------------------------
# MATERIAL
# --------------------------------------------------
mat = FrictMat(
    young=1e5,
    poisson=0.3,
    frictionAngle=radians(10),
    density=2600
)
O.materials.append(mat)

# --------------------------------------------------
# GEOMETRY
# --------------------------------------------------
Li = 0.04
Hi = 0.10
width = 0.05
Lf = 0.60
Hwall = Hi * 1.6     # <<< shorter walls

# --------------------------------------------------
# BASE
# --------------------------------------------------
O.bodies.append(
    box(center=(Lf/2, width/2, -0.005),
        extents=(Lf/2, width/2+0.01, 0.005),
        fixed=True, material=mat)
)

# --------------------------------------------------
# LEFT WALL
# --------------------------------------------------
O.bodies.append(
    box(center=(-0.005, width/2, Hwall/2),
        extents=(0.005, width/2+0.01, Hwall/2),
        fixed=True, material=mat)
)

# --------------------------------------------------
# GATE
# --------------------------------------------------
gate_id = O.bodies.append(
    box(center=(Li, width/2, Hwall/2),
        extents=(0.005, width/2+0.01, Hwall/2),
        fixed=True, material=mat, color=(1, 0, 0))
)

# --------------------------------------------------
# SIDE WALLS
# --------------------------------------------------
O.bodies.append([
    box(center=(Lf/2, 0, Hwall/2),
        extents=(Lf/2, 0.003, Hwall/2),
        fixed=True, material=mat, wire=True),
    box(center=(Lf/2, width, Hwall/2),
        extents=(Lf/2, 0.003, Hwall/2),
        fixed=True, material=mat, wire=True)
])

# --------------------------------------------------
# PARTICLES (250, tall column)
# --------------------------------------------------
sp = pack.SpherePack()
sp.makeCloud(
    minCorner=(0.004, 0.004, 0.002),
    maxCorner=(Li-0.004, width-0.004, Hi*3.0),
    rMean=0.003,
    rRelFuzz=0.3,
    num=250,
    seed=1
)
sp.toSimulation(material=mat)

print("Particles:",
      len([b for b in O.bodies if isinstance(b.shape, Sphere)]))

# --------------------------------------------------
# GATE MOTION
# --------------------------------------------------
gate_released = False

def liftGate():
    global gate_released

    if O.iter == 2000 and not gate_released:
        gate_released = True
        print(">>> GATE LIFTING <<<")

    if gate_released:
        g = O.bodies[gate_id]
        g.state.pos += Vector3(0, 0, 0.0005)

        if g.state.pos[2] > Hwall:
            O.bodies.erase(gate_id)
            gateLifter.dead = True
            print(">>> FLOW <<<")

# --------------------------------------------------
# ENGINES
# --------------------------------------------------
O.engines = [
    ForceResetter(),
    InsertionSortCollider(
        [Bo1_Sphere_Aabb(), Bo1_Box_Aabb()],
        verletDist=0.002
    ),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom(), Ig2_Box_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_ScGeom_FrictPhys_CundallStrack()]
    ),
    NewtonIntegrator(gravity=(0, 0, -9.81), damping=0.1),
    PyRunner(command='liftGate()', iterPeriod=10, label='gateLifter')
]

O.dt = 0.5 * PWaveTimeStep()

qt.View()
O.run()
