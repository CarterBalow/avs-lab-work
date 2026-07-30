import os
import matplotlib.pyplot as plt
import numpy as np

from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, RigidBodyKinematics as rbk, simHelpers
from Basilisk.simulation import NBodyGravity, mujoco, pointMassGravityModel, simpleNav
from Basilisk.fswAlgorithms import mrpFeedback, inertial3D, attTrackingError
from Basilisk.architecture import messaging, sysModel

from Basilisk import __path__

fileName = os.path.basename(os.path.splitext(__file__)[0])


def plotBendingAngles(timeAxis: np.ndarray, theta: list, numberOfSegments: int) -> plt.Figure:
    """Plots bending angles for each segment"""
    fig = plt.figure(num = 1, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(numberOfSegments):
        plt.plot(timeAxis * macros.NANO2MIN, macros.R2D * theta[idx],
                 color = simHelpers.getLineColor(idx, numberOfSegments),
                 label = r'$\theta_' + str(idx + 1) + '$')
    plt.legend(loc = 'lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'$\theta$ [deg]')

    return fig


def plotTorsionalAngles(timeAxis: np.ndarray, beta: list, numberOfSegments: int) -> plt.Figure:
    """Plots torsional angles for each segment"""
    fig = plt.figure(num = 2, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(numberOfSegments):
        plt.plot(timeAxis * macros.NANO2MIN, macros.R2D * beta[idx],
                 color = simHelpers.getLineColor(idx, numberOfSegments),
                 label = r'$\beta_' + str(idx + 1) + '$')
    plt.legend(loc = 'lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'$\beta$ [deg]')

    return fig


def plotBendingAngleRates(timeAxis: np.ndarray, thetaDot: list, numberOfSegments: int) -> plt.Figure:
    """Plots bending angle rates for each segment"""
    fig = plt.figure(num = 3, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(numberOfSegments):
        plt.plot(timeAxis * macros.NANO2MIN, macros.R2D * thetaDot[idx],
                 color = simHelpers.getLineColor(idx, numberOfSegments),
                 label = r'$\dot{\theta}_' + str(idx + 1) + '$')
    plt.legend(loc = 'lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'$\dot{\theta}$ [deg/s]')

    return fig


def plotTorsionalAngleRates(timeAxis: np.ndarray, betaDot: list, numberOfSegments: int) -> plt.Figure:
    """Plots torsional angle rates for each segment"""
    fig = plt.figure(num = 4, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(numberOfSegments):
        plt.plot(timeAxis * macros.NANO2MIN, macros.R2D * betaDot[idx],
                 color = simHelpers.getLineColor(idx, numberOfSegments),
                 label = r'$\dot{\beta}_' + str(idx + 1) + '$')
    plt.legend(loc = 'lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'$\dot{\beta}$ [deg/s]')

    return fig


def plotAttitudeError(timeAxis: np.ndarray, sigma_BR: np.ndarray) -> plt.Figure:
    """Plots attitude error MRP components"""
    fig = plt.figure(num = 5, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, sigma_BR[:, idx],
                 color = simHelpers.getLineColor(idx, 3),
                 label = r'$\sigma_' + str(idx) + '$')
    plt.legend(loc = 'lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'$\sigma_{B/R}$')

    return fig


def plotAttitudeErrorRate(timeAxis: np.ndarray, omega_BR_B: np.ndarray) -> plt.Figure:
    """Plots attitude error rate components"""
    fig = plt.figure(num = 6, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(3):
        plt.plot(timeAxis * macros.NANO2MIN, omega_BR_B[:, idx],
                 color = simHelpers.getLineColor(idx, 3),
                 label = r'$\omega_' + str(idx) + '$')
    plt.legend(loc = 'lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'$\omega_{B/R}$')

    return fig


class geometryClass:
    massHub = 1000
    lengthHub = 3
    widthHub = 3
    heightHub = 6
    lengthPanel = 18.0
    widthPanel = 3.0
    thicknessPanel = 0.3
    massPanel = 100.0

    def __init__(self, numberOfSegments):
        self.numberOfSegments = numberOfSegments
        self.massSubPanel = self.massPanel / self.numberOfSegments
        self.lengthSubPanel = self.lengthPanel / self.numberOfSegments
        self.widthSubPanel = self.widthPanel
        self.thicknessSubPanel = self.thicknessPanel


def panelChainGen(scGeometry: geometryClass, baseIndent: int):
    openTags = []
    closeTags = []

    for idx in range(scGeometry.numberOfSegments):
        n = idx + 1
        pad = "\t" * (baseIndent + 2 * idx)

        if idx == 0:
            bendingPos = f"{scGeometry.lengthHub / 2} 0 {scGeometry.heightHub / 2 - scGeometry.thicknessSubPanel / 2}"
        else:
            bendingPos = f"{scGeometry.lengthSubPanel} 0 0"

        COM_offset = f"{scGeometry.lengthSubPanel / 2} 0 0"

        ixx = round(scGeometry.massSubPanel / 12 * (scGeometry.lengthSubPanel**2 + scGeometry.thicknessSubPanel**2), 6)
        iyy = round(scGeometry.massSubPanel / 12 * (scGeometry.widthSubPanel**2 + scGeometry.thicknessSubPanel**2), 6)
        izz = round(scGeometry.massSubPanel / 12 * (scGeometry.widthSubPanel**2 + scGeometry.lengthSubPanel**2), 6)    

        openTags.append(
f"""{pad}<body name = "subPanel{n}" pos = "{bendingPos}">
{pad}   <joint name = "bendJoint{n}" pos = "0 0 0" axis = "0 1 0" ref = "0"/>
{pad}   <joint name = "twistJoint{n}" pos = "0 0 0" axis = "1 0 0" ref = "0"/>
{pad}   <inertial pos = "{COM_offset}" mass = "{scGeometry.massSubPanel}" diaginertia = "{ixx} {iyy} {izz}"/>
{pad}   <geom name = "subPanel{n}Geom" class = "subpanel_geom" pos = "{COM_offset}"/>""")
        
        closeTags.append(f"{pad}</body>")

    return "\n".join(openTags) + "\n" + "\n".join(reversed(closeTags))


def makeMjXmlString(scGeometry: geometryClass, hubMass: float = 1000.0):
    
    ixx = scGeometry.massHub / 12 * (scGeometry.lengthHub**2 + scGeometry.heightHub**2)
    iyy = scGeometry.massHub / 12 * (scGeometry.widthHub**2 + scGeometry.heightHub**2)
    izz = scGeometry.massHub / 12 * (scGeometry.lengthHub**2 + scGeometry.widthHub**2)

    panelChain = panelChainGen(scGeometry, baseIndent = 3)
    
    return f"""<mujoco model = "busWithFlexiblePanel">
    <compiler angle = "radian" meshdir = ""/>
    <default>
        <default class = "subpanel_geom">
            <geom type = "box" size = "{scGeometry.lengthSubPanel / 2} {scGeometry.widthSubPanel / 2} {scGeometry.thicknessSubPanel / 2}"
            contype = "0" conaffinity = "1"/>
        </default>
    </default>

    <worldbody>
        <body name = "hub" pos = "0 0 0">
            <freejoint name = "busFree"/>
            <inertial pos = "0 0 0" mass = "{hubMass}" diaginertia = "{ixx} {iyy} {izz}"/>
            <geom name = "hubVisual" type = "box" size = "{scGeometry.lengthHub / 2} {scGeometry.widthHub / 2} {scGeometry.heightHub / 2}" rgba = "1 1 1 1"/>
            <site name = "hubSite" pos = "0 0 0"/>
{panelChain}
        </body>
    </worldbody>
</mujoco>"""
    

def run(showPlots: bool = False):
    # -------------------------------------------------------------------------
    # 1) Simulation configuration and MJScene dynamics model
    # -------------------------------------------------------------------------
    simTaskName = "simTask"
    simProcessName = "simProcess"
    fswTaskName = "fswTask"
    fswProcessName = "fswProcess"

    simulationTime = macros.min2nano(10.0)
    timeStep = macros.sec2nano(0.1)
    fswTimeStep = macros.sec2nano(0.5)

    sim = SimulationBaseClass.SimBaseClass()
    dynProcess = sim.CreateNewProcess(simProcessName)
    dynProcess.addTask(sim.CreateNewTask(simTaskName, timeStep))
    fswProcess = sim.CreateNewProcess(fswProcessName)
    fswProcess.addTask(sim.CreateNewTask(fswTaskName, fswTimeStep))

    numberOfSegments = 5
    scGeometry = geometryClass(numberOfSegments)
    xmlString = makeMjXmlString(scGeometry)

    scene = mujoco.MJScene(xmlString)
    scene.ModelTag = "mujocoScene"
    sim.AddModelToTask(simTaskName, scene)

    # -------------------------------------------------------------------------
    # 2) Retrieve spacecraft componenets
    # -------------------------------------------------------------------------
    busBody = scene.getBody("hub")
    subPanels = [scene.getBody(f"subPanel{i + 1}") for i in range(numberOfSegments)]

    bendJoints = [subPanels[i].getScalarJoint(f"bendJoint{i + 1}") for i in range(numberOfSegments)]
    twistJoints = [subPanels[i].getScalarJoint(f"twistJoint{i + 1}") for i in range(numberOfSegments)]

    # -------------------------------------------------------------------------
    # 3) Adding damping/stiffness to subpanels (bend & twist)
    # ------------------------------------------------------------------------- 
    kBend, cBend = 10.0, 8.0
    kTwist, cTwist = 1.0, 0.8

    springDampers = []
    for i in range(numberOfSegments):
            for jointName, k, c in [(f"bendJoint{i + 1}", kBend, cBend), (f"twistJoint{i + 1}", kTwist, cTwist)]:
                actuator = scene.addJointSingleActuator(f"{jointName}Actuator", jointName)
                joint = subPanels[i].getScalarJoint(jointName)

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
    oe.a = 8e6
    oe.e = 0.1
    oe.i = 0.0 * macros.D2R
    oe.Omega = 0.0 * macros.D2R
    oe.omega = 0.0 * macros.D2R
    oe.f = 0.0 * macros.D2R
    muEarth = 0.3986004415e15  # [m^3/s^2]
    rN, vN = orbitalMotion.elem2rv(muEarth, oe)

    gravity = NBodyGravity.NBodyGravity()
    gravity.ModelTag = "gravity"
    scene.AddModelToDynamicsTask(gravity)

    earthPm = pointMassGravityModel.PointMassGravityModel()
    earthPm.muBody = muEarth
    gravity.addGravitySource("earth", earthPm, isCentralBody = True)

    gravity.addGravityTarget("hub", busBody)
    for i in range(numberOfSegments):
        gravity.addGravityTarget(f"subPanel{i + 1}", subPanels[i])

    # -------------------------------------------------------------------------
    # 5) Navigation and attitude control
    # -------------------------------------------------------------------------
    simpleNavObj = simpleNav.SimpleNav()
    simpleNavObj.ModelTag = "simpleNav"
    simpleNavObj.scStateInMsg.subscribeTo(busBody.getCenterOfMass().stateOutMsg)
    sim.AddModelToTask(simTaskName, simpleNavObj)

    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    inertial3DObj.sigma_R0N = [0.3, 0.4, 0.5]
    sim.AddModelToTask(fswTaskName, inertial3DObj)

    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attTrackingError"
    attError.attNavInMsg.subscribeTo(simpleNavObj.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
    sim.AddModelToTask(fswTaskName, attError)

    IHubPntBc_B = np.array([[scGeometry.massHub / 12 * (scGeometry.widthHub**2 + scGeometry.heightHub**2), 0.0, 0.0],
                            [0.0, scGeometry.massHub / 12 * (scGeometry.lengthHub**2 + scGeometry.heightHub**2), 0.0],
                            [0.0, 0.0, scGeometry.massHub / 12 * (scGeometry.lengthHub**2 + scGeometry.widthHub**2)]])
    IPanelPntSc_B = np.array([[scGeometry.massPanel / 12 * (scGeometry.widthPanel**2 + scGeometry.thicknessPanel**2), 0.0, 0.0,],
                              [0.0, scGeometry.massPanel / 12 * (scGeometry.lengthPanel**2 + scGeometry.thicknessPanel**2), 0.0],
                              [0.0, 0.0, scGeometry.massPanel / 12 * (scGeometry.widthPanel**2 + scGeometry.lengthPanel**2)]])
    r_ScB_B = [scGeometry.lengthHub/2 + scGeometry.lengthPanel/2, 0.0,
           scGeometry.heightHub/2 - scGeometry.thicknessSubPanel/2]
    IHubPntB_B =  IHubPntBc_B + IPanelPntSc_B - scGeometry.massPanel * np.array(rbk.v3Tilde(r_ScB_B)) @ np.array(rbk.v3Tilde(r_ScB_B))
    
    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    decayTime = 50
    xi = 0.9
    mrpControl.P = 2 * np.max(IHubPntB_B) / decayTime
    mrpControl.K = (mrpControl.P / xi) ** 2 / np.max(IHubPntB_B)
    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)

    configData = messaging.VehicleConfigMsgPayload(ISCPntB_B = list(IHubPntB_B.flatten()))
    configDataMsg = messaging.VehicleConfigMsg()
    configDataMsg.write(configData)
    mrpControl.vehConfigInMsg.subscribeTo(configDataMsg)
    sim.AddModelToTask(fswTaskName, mrpControl)

    torqueActuator = scene.addTorqueActuator("hubTorqueAct", "hubSite")

    torqueBridge = CmdTorqueToSiteActuator()
    torqueBridge.ModelTag = "torqueBridge"
    torqueBridge.cmdTorqueInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)
    scene.AddModelToDynamicsTask(torqueBridge)

    torqueActuator.torqueInMsg.subscribeTo(torqueBridge.torqueOutMsg)

    # -------------------------------------------------------------------------
    # 6) Setup data recording
    # -------------------------------------------------------------------------
    dataLog = busBody.getCenterOfMass().stateOutMsg.recorder()
    sim.AddModelToTask(simTaskName, dataLog)

    attErrorLog = attError.attGuidOutMsg.recorder()
    sim.AddModelToTask(fswTaskName, attErrorLog)

    posData, rateData = [], []
    for i in range(numberOfSegments):
        for joint in (bendJoints[i], twistJoints[i]):
            posLog = joint.stateOutMsg.recorder()
            rateLog = joint.stateDotOutMsg.recorder()
            sim.AddModelToTask(simTaskName, posLog)
            sim.AddModelToTask(simTaskName, rateLog)
            posData.append(posLog)
            rateData.append(rateLog)

    # -------------------------------------------------------------------------
    # 7) Initialize simulation, orbit, & joint angles
    # -------------------------------------------------------------------------
    sim.InitializeSimulation()

    busFree = busBody.getFreeJoint()
    busBody.setPosition(rN)
    busFree.setVelocity(vN)

    thetaInit = 0.0
    for i in range(numberOfSegments):
        bendJoints[i].setPosition(thetaInit)
        twistJoints[i].setPosition(thetaInit)

    # -------------------------------------------------------------------------
    # 8) Execute simulation
    # -------------------------------------------------------------------------
    sim.ConfigureStopTime(simulationTime)
    sim.ExecuteSimulation()

    # -------------------------------------------------------------------------
    # 9) Post processing and plotting
    # -------------------------------------------------------------------------
    theta, thetaDot = [], []
    betas, betaDot = [], []
    for idx in range(numberOfSegments):
        theta.append(posData[2 * idx].state)
        thetaDot.append(rateData[2 * idx].state)
        betas.append(posData[2 * idx + 1].state)
        betaDot.append(rateData[2 * idx + 1].state)

    timeAxis = posData[0].times()
    timeAxisFSW = attErrorLog.times()

    plt.close("all")
    figureList = {}
    figureList[fileName + "_thetas"] = plotBendingAngles(timeAxis, theta, numberOfSegments)
    figureList[fileName + "_betas"] = plotTorsionalAngles(timeAxis, betas, numberOfSegments)
    figureList[fileName + "_thetaDots"] = plotBendingAngleRates(timeAxis, thetaDot, numberOfSegments)
    figureList[fileName + "_betaDots"] = plotTorsionalAngleRates(timeAxis, betaDot, numberOfSegments)
    figureList[fileName + "_sigma_BR"] = plotAttitudeError(timeAxisFSW, attErrorLog.sigma_BR)
    figureList[fileName + "_omega_BR_B"] = plotAttitudeErrorRate(timeAxisFSW, attErrorLog.omega_BR_B)

    if showPlots:
        plt.show()

    return


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


class CmdTorqueToSiteActuator(sysModel.SysModel):
    def __init__(self):
        super().__init__()
        self.cmdTorqueInMsg = messaging.CmdTorqueBodyMsgReader()
        self.torqueOutMsg = messaging.TorqueAtSiteMsg()

    def UpdateState(self, CurrentSimNanos):
        cmd = self.cmdTorqueInMsg()
        payload = messaging.TorqueAtSiteMsgPayload(torque_S = cmd.torqueRequestBody)
        self.torqueOutMsg.write(payload, self.moduleID, CurrentSimNanos)


if __name__ == "__main__":
    # --- XML TESTING ---
    numSegments = 5
    scGeometry = geometryClass(numSegments)
    xmlString = makeMjXmlString(scGeometry, hubMass = 1000.0)
 
    with open("spacecraft.xml", "w") as f:
        f.write(xmlString)

    run(showPlots = True)