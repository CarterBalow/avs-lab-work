from typing import Tuple
import os

import matplotlib.pyplot as plt
import numpy as np

from Basilisk.architecture import messaging, sysModel
from Basilisk.simulation import NBodyGravity, mujoco, pointMassGravityModel
from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simHelpers, RigidBodyKinematics

fileName = os.path.basename(os.path.splitext(__file__)[0])


def plotInertialPos(timeAxis: np.ndarray, posData: np.ndarray) -> plt.Figure:
    """Plots inertial position vector componenets"""
    fig = plt.figure(num = 1, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, posData[:, idx] / 1000.,
                 color = simHelpers.getLineColor(idx, 3),
                 label = '$r_{BN,' + str(idx) + '}$')
    plt.legend(loc = 'lower right')
    plt.xlabel('Time [h]')
    plt.ylabel('Inertial Position [km]')

    return fig


def plotOrbitalMotion(timeAxis: np.ndarray, posData: np.ndarray, velData: np.ndarray, mu: float) -> plt.Figure:
    """Plots orbital motion of spacecraft"""
    fig = plt.figure(num = 2, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    rData = []
    for idx in range(0, len(posData)):
        rVec = np.array(posData[idx]).flatten()
        vVec = np.array(velData[idx]).flatten()
        
        oeData = orbitalMotion.rv2elem(mu, rVec, vVec)
        rData.append(oeData.rmag / 1000.)
        
    plt.plot(timeAxis * macros.NANO2MIN, rData, color='#aa0000')
    plt.xlabel('Time [min]')
    plt.ylabel('Radius [km]')

    return fig


def plotAngDisp(timeAxis: np.ndarray, panel1thetaLog: np.ndarray, panel2thetaLog: np.ndarray) -> plt.Figure:
    """Plots angular displacements of panels"""
    fig, (ax1, ax2) = plt.subplots(2, 1, sharex = True, num = 3, clear = True)

    ax1.plot(timeAxis * macros.NANO2MIN, panel1thetaLog)
    ax1.set_xlabel("Time [min]")
    ax1.set_ylabel('Panel 1 Angular Displacement [r]')

    ax2.plot(timeAxis * macros.NANO2MIN, panel2thetaLog)
    ax2.set_xlabel("Time [min]")
    ax2.set_ylabel('Panel 2 Angular Displacement [r]')

    fig.tight_layout()

    return fig


def makeMjXmlString(hubMass: float = 800.0, busIDiag: Tuple[float, float, float] = (900.0, 800.0, 600.0),):
    
    ixx, iyy, izz = busIDiag
    
    return f"""
    <mujoco model = "busWith2Panels">
        <compiler angle = "radian" meshdir = ""/>
        
        <default>
            <default class = "panel_geom">
                <geom type = "box" size = "1 1 0.01"
                contype = "0" conaffinity = "1"/>
            </default>
        </default>

        <worldbody>
            <body name = "hub" pos = "0 0 0">
                <freejoint name = "busFree"/>
                
                <inertial pos = "0 0 0" mass = "{hubMass}" diaginertia = "{ixx} {iyy} {izz}"/>
                <geom name = "hubVisual" type = "box" size = "1 1 1" rgba = "1 1 1 1"/>

                <site name = "thrustSite" pos = "0 0 0"/>

                <body name = "panel1" pos = "0.5 0.0 1.0">
                    <joint name = "hinge1" pos = "0 0 0" axis = "0 -1 0" ref = "0"/>
                    <inertial pos = "1.5 0 0" mass = "100.0" diaginertia = "100.0 50.0 50.0"/>
                    <geom name = "panel1_geom" class = "panel_geom" pos = "1.5 0 0"/>
                </body>

                <body name = "panel2" pos = "-0.5 0.0 1.0">
                    <joint name = "hinge2" pos = "0 0 0" axis = "0 1 0" ref = "0"/>
                    <inertial pos = "-1.5 0 0" mass = "100.0" diaginertia = "100.0 50.0 50.0"/>
                    <geom name = "panel2_geom" class = "panel_geom" pos = "-1.5 0 0"/>
                </body>
            </body>
        </worldbody>
    </mujoco>
    """
    

def run(showPlots: bool = False):
    # -------------------------------------------------------------------------
    # 1) Simulation configuration and MJScene dynamics model
    # -------------------------------------------------------------------------
    simTaskName = "simTask"
    simProcessName = "simProcess"

    simulationTime = macros.min2nano(10.0)
    timeStep = macros.sec2nano(0.1)

    sim = SimulationBaseClass.SimBaseClass()
    dynProcess = sim.CreateNewProcess(simProcessName)
    dynProcess.addTask(sim.CreateNewTask(simTaskName, timeStep))

    xmlString = makeMjXmlString()
    scene = mujoco.MJScene(xmlString)
    scene.ModelTag = "mujocoScene"
    sim.AddModelToTask(simTaskName, scene)

    thrustActuator = scene.addForceActuator("thrustForce", "thrustSite")

    # -------------------------------------------------------------------------
    # 2) Retrieve spacecraft componenets
    # -------------------------------------------------------------------------
    busBody = scene.getBody("hub")
    panelBodies = [scene.getBody(name) for name in ("panel1", "panel2")]
    numPanels = len(panelBodies)

    hinge1 = panelBodies[0].getScalarJoint("hinge1")
    hinge2 = panelBodies[1].getScalarJoint("hinge2")

    # -------------------------------------------------------------------------
    # 3) Adding damping/stiffness to panel hinges
    # ------------------------------------------------------------------------- 

    k = 1000.0
    c = 0.0

    springDampers = []
    for jointName, body in [("hinge1", panelBodies[0]), ("hinge2", panelBodies[1])]:
        actuator = scene.addJointSingleActuator(f"{jointName}Actuator", jointName)
        joint = body.getScalarJoint(jointName)

        sd = JointSpringDamper(k = k, c = c, thetaRef = 0.0)
        sd.ModelTag = f"{jointName}SpringDamper"
        sd.jointPosInMsg.subscribeTo(joint.stateOutMsg)
        sd.jointVelInMsg.subscribeTo(joint.stateDotOutMsg)
        actuator.actuatorInMsg.subscribeTo(sd.actuatorOutMsg)

        scene.AddModelToDynamicsTask(sd)
        springDampers.append(sd) 

    # -------------------------------------------------------------------------
    # 4) Add gravity
    # -------------------------------------------------------------------------
    oe = orbitalMotion.ClassicElements()
    rLEO = 7000. * 1000  # meters
    oe.a = rLEO
    oe.e = 0.0001
    oe.i = 0.0 * macros.D2R
    oe.Omega = 48.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 85.3 * macros.D2R
    muEarth = 0.3986004415e15  # [m^3/s^2]
    rN, vN = orbitalMotion.elem2rv(muEarth, oe)

    gravity = NBodyGravity.NBodyGravity()
    gravity.ModelTag = "gravity"
    scene.AddModelToDynamicsTask(gravity)

    earthPm = pointMassGravityModel.PointMassGravityModel()
    earthPm.muBody = muEarth
    gravity.addGravitySource("earth", earthPm, isCentralBody = True)

    gravity.addGravityTarget("hub", busBody)
    for i in range(numPanels):
        gravity.addGravityTarget(f"panel{i + 1}", panelBodies[i])

    # -------------------------------------------------------------------------
    # 5) Applying thrust
    # -------------------------------------------------------------------------
    n = np.sqrt(muEarth / oe.a / oe.a / oe.a)
    P = 2. * np.pi / n
    simulationTimeFactor = 0.01
    simulationTime = macros.sec2nano(simulationTimeFactor * P)

    T2 = macros.sec2nano(935.)
    burnStart = simulationTime
    burnEnd = simulationTime + T2

    forceConverter = InertialForceToSiteActuator()
    forceConverter.scStateInMsg.subscribeTo(busBody.getCenterOfMass().stateOutMsg)
    forceConverter.setForce_N([-2050.0, -1430.0, -0.00076])
    forceConverter.setBurnWindow(burnStartNanos = burnStart, burnEndNanos = burnEnd)
    forceConverter.ModelTag = "forceConverter"
    scene.AddModelToDynamicsTask(forceConverter)

    thrustActuator.forceInMsg.subscribeTo(forceConverter.forceOutMsg)

    # -------------------------------------------------------------------------
    # 6) Setup data recording
    # -------------------------------------------------------------------------
    numDataPoints = 100
    samplingTime = simHelpers.samplingTime(simulationTime, timeStep, numDataPoints)

    dataLog = busBody.getCenterOfMass().stateOutMsg.recorder(samplingTime)
    pl1Log = hinge1.stateOutMsg.recorder(samplingTime)
    pl2Log = hinge2.stateOutMsg.recorder(samplingTime)

    sim.AddModelToTask(simTaskName, dataLog)
    sim.AddModelToTask(simTaskName, pl1Log)
    sim.AddModelToTask(simTaskName, pl2Log)


    # -------------------------------------------------------------------------
    # 7) Setup orbit / initialize spacecraft state
    # -------------------------------------------------------------------------
    sim.InitializeSimulation()

    busFree = busBody.getFreeJoint()
    busBody.setPosition(rN)
    busFree.setVelocity(vN)

    thetaInit = 5.0 * np.pi / 180.0
    hinge1.setPosition(thetaInit)
    hinge2.setPosition(thetaInit)

    # -------------------------------------------------------------------------
    # 8) Execute simulation (passive orbit + burn)
    # -------------------------------------------------------------------------
    sim.ConfigureStopTime(burnEnd)
    sim.ExecuteSimulation()

    # -------------------------------------------------------------------------
    # 9) Post processing and plotting
    # -------------------------------------------------------------------------
    posData = dataLog.r_BN_N
    velData = dataLog.v_BN_N
    panel1data = pl1Log.state
    panel2data = pl2Log.state
    timeAxis = dataLog.times()
    
    plt.close("all")
    figureList = {}
    figureList[fileName + "1"] = plotInertialPos(timeAxis, posData)
    figureList[fileName + "2"] = plotOrbitalMotion(timeAxis, posData, velData, muEarth)
    figureList[fileName + "3"] = plotAngDisp(timeAxis, panel1data, panel2data)

    if showPlots:
        plt.show()

    return figureList


class InertialForceToSiteActuator(sysModel.SysModel):
    def __init__(self):
        super().__init__()
        self.force_N = np.zeros(3)
        self.burnStartNanos = 0
        self.burnEndNanos = 0

        self.scStateInMsg = messaging.SCStatesMsgReader()
        self.forceOutMsg = messaging.ForceAtSiteMsg()

    def setForce_N(self, force_N):
        self.force_N = np.array(force_N)

    def setBurnWindow(self, burnStartNanos, burnEndNanos):
        self.burnStartNanos = burnStartNanos
        self.burnEndNanos = burnEndNanos

    def UpdateState(self, CurrentSimNanos):
        if self.burnStartNanos <= CurrentSimNanos <= self.burnEndNanos:
            state = self.scStateInMsg()
            dcm_BN = RigidBodyKinematics.MRP2C(np.array(state.sigma_BN))
            force_B = dcm_BN @ self.force_N
        else:
            force_B = np.zeros(3)

        # NOTE: frames S and B in this scenario are identical, thus force_S = force_B
        self.forceOutMsg.write(messaging.ForceAtSiteMsgPayload(force_S = force_B), self.moduleID, CurrentSimNanos)


class JointSpringDamper(sysModel.SysModel):
    def __init__(self, k: float, c: float, thetaRef: float):
        super().__init__()
        self.k = k
        self.c = c
        self.thetaRef = thetaRef

        self.jointPosInMsg = messaging.ScalarJointStateMsgReader()
        self.jointVelInMsg = messaging.ScalarJointStateMsgReader()
        self.actuatorOutMsg = messaging.SingleActuatorMsg()

    def UpdateState(self, CurrentSimNanos):
        theta = self.jointPosInMsg().state
        thetaDot = self.jointVelInMsg().state

        torque = -self.k * (theta - self.thetaRef) - self.c * thetaDot

        payload = messaging.SingleActuatorMsgPayload()
        payload.input = torque
        self.actuatorOutMsg.write(payload, self.moduleID, CurrentSimNanos)


if __name__ == "__main__":
    run(showPlots = True)