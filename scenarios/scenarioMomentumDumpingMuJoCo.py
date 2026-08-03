import os
import matplotlib.pyplot as plt
import numpy as np

from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, RigidBodyKinematics as rbk, simHelpers, simIncludeRW, simIncludeThruster
from Basilisk.simulation import NBodyGravity, mujoco, pointMassGravityModel, simpleNav
from Basilisk.fswAlgorithms import mrpFeedback, inertial3D, attTrackingError
from Basilisk.architecture import messaging, sysModel

from Basilisk import __path__

fileName = os.path.basename(os.path.splitext(__file__)[0])


def plot_attitude_error(timeData, dataSigmaBR):
    """Plot the attitude errors."""
    plt.figure(1)
    for idx in range(3):
        plt.plot(timeData, dataSigmaBR[:, idx],
                 color = simHelpers.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + r'$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')


def plot_rate_error(timeData, dataOmegaBR):
    """Plot the body angular velocity rate tracking errors."""
    plt.figure(2)
    for idx in range(3):
        plt.plot(timeData, dataOmegaBR[:, idx],
                 color = simHelpers.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error (rad/s) ')


def plot_rw_momenta(timeData, dataOmegaRw, RW, numRW):
    """Plot the RW momenta."""
    totMomentumNorm = []
    for j in range(len(timeData)):
        totMomentum = np.array([0,0,0])
        for idx in range(numRW):
            for k in range(3):
                totMomentum[k] = totMomentum[k] + dataOmegaRw[j, idx] * RW[idx].Js * RW[idx].gsHat_B[k][0]
        totMomentumNorm.append(np.linalg.norm(totMomentum))
    plt.figure(3)
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRw[:, idx] * RW[idx].Js,
                 color = simHelpers.getLineColor(idx, numRW),
                 label=r'$H_{' + str(idx+1) + r'}$')
    plt.plot(timeData, totMomentumNorm, '--',
             label=r'$\|H\|$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Momentum (Nms)')


def plot_DH(timeData, dataDH):
    """Plot the body angular velocity rate tracking errors."""
    plt.figure(4)
    for idx in range(3):
        plt.plot(timeData, dataDH[:, idx],
                 color = simHelpers.getLineColor(idx, 3),
                 label=r'$\Delta H_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Dumped momentum (Nms) ')


def plot_rw_speeds(timeData, dataOmegaRW, numRW):
    """Plot the RW spin rates."""
    plt.figure(5)
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRW[:, idx] / macros.RPM,
                 color = simHelpers.getLineColor(idx, numRW),
                 label=r'$\Omega_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Speed (RPM) ')


def plot_thrImpulse(timeDataFSW, dataMap, numTh):
    """Plot the Thruster force values."""
    plt.figure(5)
    for idx in range(numTh):
        plt.plot(timeDataFSW, dataMap[:, idx],
                 color = simHelpers.getLineColor(idx, numTh),
                 label=r'$thrImpulse_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Impulse requested [Ns]')


def plot_OnTimeRequest(timeData, dataOnTime, numTh):
    """Plot the thruster on time requests."""
    plt.figure(6)
    for idx in range(numTh):
        plt.plot(timeData, dataOnTime[:, idx],
                 color = simHelpers.getLineColor(idx, numTh),
                 label=r'$OnTimeRequest_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('OnTimeRequest [sec]')


def plot_thrForce(timeDataFSW, dataThr, numTh):
    """Plot the Thruster force values."""
    plt.figure(7)
    for idx in range(numTh):
        plt.plot(timeDataFSW, dataThr[idx],
                 color = simHelpers.getLineColor(idx, numTh),
                 label=r'$thrForce_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Thruster force [N]')


def quatAlignment(axis: list):
    ax = np.array(axis, dtype = float)
    ax = ax / np.linalg.norm(ax)
    z = np.array([0.0, 0.0, 1.0])
 
    w = 1.0 + np.dot(z, ax)
    if w < 1e-8:
        perp = np.array([1.0, 0.0, 0.0]) if abs(ax[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        perp = perp - ax * np.dot(perp, ax)
        perp = perp / np.linalg.norm(perp)
        return f"0 {perp[0]:.8f} {perp[1]:.8f} {perp[2]:.8f}"
 
    v = np.cross(z, ax)
    q = np.array([w, v[0], v[1], v[2]])
    q = q / np.linalg.norm(q)
    return f"{q[0]:.8f} {q[1]:.8f} {q[2]:.8f} {q[3]:.8f}"


def addRWsXML(rwPos: list, 
              rwAxes: list,
              rwFactory: simIncludeRW, 
              baseIndent: int = 2):

    numRW = len(rwPos)
    pad = "\t" * baseIndent

    rwTags = []
    for idx in range(numRW):
        n = idx + 1
        rw = rwFactory.create('Honeywell_HR16', rwAxes[idx], maxMomentum=100.)
        pos = rwPos[idx]
        quat = quatAlignment(rwAxes[idx])

        rwTags.append(
f"""{pad}<body name = "rw{n}Spin" pos = "{pos[0]} {pos[1]} {pos[2]}" quat = "{quat}">
{pad}   <joint name = "rw{n}Joint" pos = "0 0 0" axis = "0 0 1" ref = "0"/>
{pad}   <inertial pos = "0 0 0" mass = "{rw.mass}" diaginertia = "{rw.Jt} {rw.Jt} {rw.Js}"/>
{pad}   <geom name = "rw{n}Geom" type = "cylinder" contype = "0" conaffinity = "0"/>
{pad}</body>""")

    return "\n".join(rwTags)


def addThrustersXML(thrustLocs: list, 
                    thrustDirs: list,
                    baseIndent: int = 2):
    
    pad = "\t" * baseIndent
    thrustTags = []
    actTags = []
    for idx in range(len(thrustLocs)):
        n = idx + 1
        pos = thrustLocs[idx]
        quat = quatAlignment(thrustDirs[idx])
        thrustTags.append(f'{pad}<site name = "thrusterSite{n}" pos = "{pos[0]} {pos[1]} {pos[2]}" quat = "{quat}"/>')
        actTags.append(f'{pad}<name = "thruster{n}" site = "thrusterSite{n}" gear = "0 0 1 0 0 0" ctrlrange="0 5"/>')
    
    return "\n".join(thrustTags)


def makeMjXmlString():
    hubMass = 2500.0
    hubIxx, hubIyy, hubIzz = 1700.0, 1700.0, 1800.0
 
    c = 2**-0.5
    rwPos = [[0.0, 0.0, 0.0]] * 4
    rwAxes = [[c, 0, c], [0, c, c], [-c, 0, c], [0, -c, c]]

    a, b = 1.0, 1.28
    thrustLocs = [
        [-a, -a,  b], [ a, -a, -b], [ a, -a,  b], [ a,  a, -b],
        [ a,  a,  b], [-a,  a, -b], [-a,  a,  b], [-a, -a, -b],
    ]
    thrustDirs = np.array([
        [ 1,  0,  0], [-1,  0,  0], [ 0,  1,  0], [ 0, -1,  0],
        [-1,  0,  0], [ 1,  0,  0], [ 0, -1,  0], [ 0,  1,  0],
    ])
 
    rwFactory = simIncludeRW.rwFactory()
    rwChain = addRWsXML(rwPos, rwAxes, rwFactory)
    thrusterSites = addThrustersXML(thrustLocs, thrustDirs, baseIndent = 3)
 
    return f"""<mujoco model = "busWithRWsAndThrusters">
    <compiler angle = "radian" meshdir = ""/>
 
    <worldbody>
        <body name = "hub" pos = "0 0 0">
            <freejoint name = "busFree"/>
            <inertial pos = "0 0 0" mass = "{hubMass}" diaginertia = "{hubIxx} {hubIyy} {hubIzz}"/>
            <geom name = "hubVisual" type = "box" size = "1 1 1.28" rgba = "1 1 1 1"/>

{thrusterSites}

{rwChain}

        </body>
    </worldbody>
</mujoco>"""

def run(showPlots: bool = False):
    # -------------------------------------------------------------------------
    # 1) Simulation configuration and MJScene dynamics model
    # -------------------------------------------------------------------------
    fswTaskName = "fswTask"
    dynTaskName = "dynTask"
    simProcessName = "simProcess"

    simulationTime = macros.min2nano(5)
    simulationTimeStepFsw = macros.sec2nano(1)
    simulationTimeStepDyn = macros.sec2nano(0.1)

    sim = SimulationBaseClass.SimBaseClass()

    dynProcess = sim.CreateNewProcess(simProcessName)
    dynProcess.addTask(sim.CreateNewTask(dynTaskName, simulationTimeStepDyn))
    dynProcess.addTask(sim.CreateNewTask(fswTaskName, simulationTimeStepFsw))

    xmlString = makeMjXmlString()
    scene = mujoco.MJScene(xmlString)
    scene.ModelTag = "mujocoScene"
    sim.AddModelToTask(dynTaskName, scene)

    # -------------------------------------------------------------------------
    # 2) Retrieve spacecraft componenets
    # -------------------------------------------------------------------------
    busBody = scene.getBody("hub")

    numRWs = 4
    RWs = [scene.getBody(f"rw{i + 1}Spin") for i in range(numRWs)]
    RWJoints = [RWs[i].getScalarJoint(f"rw{i + 1}Joint") for i in range(numRWs)]

    numThrusters = 8
    thrusters = [scene.getSite(f"thrusterSite{i + 1}") for i in range(numThrusters)]




if __name__ == "__main__":
    # --- XML TESTING ---
    
    xmlString = makeMjXmlString()
 
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sat_momentumDump.xml"), "w") as f:
        f.write(xmlString)

    #run(True)