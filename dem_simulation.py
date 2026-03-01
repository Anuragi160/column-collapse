from yade import pack
from yade import *
from math import radians
import numpy as np
import os
import sys

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 1
SPLIT = sys.argv[2] if len(sys.argv) > 2 else 'train'
OUTPUT_DIR = sys.argv[3] if len(sys.argv) > 3 else '/home/vision/Desktop/DEM-ML-250'
TARGET_FRAMES = 2000
COLLECT_EVERY = 5

mat = FrictMat(young=1e5, poisson=0.3, frictionAngle=radians(10), density=2600)
O.materials.append(mat)

Li = 0.04
Hi = 0.10
width = 0.05
Lf = 0.60
Hwall = Hi * 1.6

O.bodies.append(box(center=(Lf/2, width/2, -0.005),
    extents=(Lf/2, width/2+0.01, 0.005), fixed=True, material=mat, color=(0.5,0.5,0.5)))
O.bodies.append(box(center=(-0.005, width/2, Hwall/2),
    extents=(0.005, width/2+0.01, Hwall/2), fixed=True, material=mat, color=(0.4,0.4,0.4)))
gate_id = O.bodies.append(box(center=(Li, width/2, Hwall/2),
    extents=(0.005, width/2+0.01, Hwall/2),
    fixed=True, material=mat, color=(0.9,0.3,0.3)))
wall_y1 = box(center=(Lf/2, 0, Hwall/2), extents=(Lf/2, 0.003, Hwall/2),
    fixed=True, material=mat, color=(0.7,0.7,0.9), wire=True)
wall_y2 = box(center=(Lf/2, width, Hwall/2), extents=(Lf/2, 0.003, Hwall/2),
    fixed=True, material=mat, color=(0.7,0.7,0.9), wire=True)
O.bodies.append([wall_y1, wall_y2])

sp = pack.SpherePack()
sp.makeCloud(minCorner=(0.004, 0.004, 0.002),
             maxCorner=(Li-0.004, width-0.004, Hi*3.0),
             rMean=0.003, rRelFuzz=0.3, num=250, periodic=False, seed=SEED)
sp.toSimulation(material=mat, color=(0.65,0.55,0.45))

sphere_ids = [b.id for b in O.bodies if isinstance(b.shape, Sphere)]
n_particles = len(sphere_ids)
collected_positions = []
gate_released = False

def collectOnly():
    pos_frame = []
    for sid in sphere_ids:
        b = O.bodies[sid]
        if b is None:
            pos_frame.append([float('nan'), float('nan'), float('nan')])
        else:
            pos_frame.append([b.state.pos[0], b.state.pos[1], b.state.pos[2]])
    collected_positions.append(pos_frame)
    if len(collected_positions) % 200 == 0:
        print(f"  [SEED={SEED}] Frames: {len(collected_positions)}/{TARGET_FRAMES}")
    if len(collected_positions) >= TARGET_FRAMES:
        print(f"  [SEED={SEED}] >>> TARGET REACHED <<<")
        O.pause()

def liftGate():
    global gate_released
    if O.iter == 2000 and not gate_released:
        gate_released = True
    if gate_released:
        current_pos = O.bodies[gate_id].state.pos
        O.bodies[gate_id].state.pos = current_pos + Vector3(0, 0, 0.0005)
        if O.bodies[gate_id].state.pos[2] > Hwall:
            O.bodies.erase(gate_id)
            gateLifter.command = 'collectOnly()'
    pos_frame = []
    for sid in sphere_ids:
        b = O.bodies[sid]
        if b is None:
            pos_frame.append([float('nan'), float('nan'), float('nan')])
        else:
            pos_frame.append([b.state.pos[0], b.state.pos[1], b.state.pos[2]])
    collected_positions.append(pos_frame)
    if len(collected_positions) % 200 == 0:
        print(f"  [SEED={SEED}] Frames: {len(collected_positions)}/{TARGET_FRAMES}")
    if len(collected_positions) >= TARGET_FRAMES:
        print(f"  [SEED={SEED}] >>> TARGET REACHED <<<")
        O.pause()

def saveNPZ():
    if len(collected_positions) == 0:
        return
    position = np.array(collected_positions, dtype=np.float32)
    particle_type = 6
    out_path = os.path.join(OUTPUT_DIR, f"{SPLIT}.npz")
    if os.path.exists(out_path):
        existing_data = list(np.load(out_path, allow_pickle=True)['gns_data'])
    else:
        existing_data = []
    existing_data.append((position, particle_type))
    data = np.empty(len(existing_data), dtype=object)
    for i, traj in enumerate(existing_data):
        data[i] = traj
    np.savez_compressed(out_path, gns_data=data)
    print(f"  [SEED={SEED}] SAVED → {SPLIT}.npz | total traj: {len(existing_data)} | shape: {position.shape}")

import atexit
atexit.register(saveNPZ)

O.engines = [
    ForceResetter(),
    InsertionSortCollider([Bo1_Sphere_Aabb(), Bo1_Box_Aabb()]),
    InteractionLoop(
        [Ig2_Sphere_Sphere_ScGeom(), Ig2_Box_Sphere_ScGeom()],
        [Ip2_FrictMat_FrictMat_FrictPhys()],
        [Law2_ScGeom_FrictPhys_CundallStrack()]
    ),
    NewtonIntegrator(gravity=(0, 0, -9.81), damping=0.1),
    PyRunner(command='liftGate()', iterPeriod=COLLECT_EVERY, label='gateLifter')
]

O.dt = 0.5 * PWaveTimeStep()
O.run(TARGET_FRAMES * COLLECT_EVERY * 3, True)
saveNPZ()
