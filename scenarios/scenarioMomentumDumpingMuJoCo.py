import os
import matplotlib.pyplot as plt
import numpy as np

from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, RigidBodyKinematics as rbk, simHelpers, simIncludeRW, simIncludeThruster, unitTestSupport
from Basilisk.simulation import NBodyGravity, mujoco, pointMassGravityModel, simpleNav
from Basilisk.fswAlgorithms import mrpFeedback, inertial3D, attTrackingError, rwMotorTorque, thrMomentumManagement, thrForceMapping, thrMomentumDumping
from Basilisk.architecture import messaging, sysModel

from Basilisk import __path__

fileName = os.path.basename(os.path.splitext(__file__)[0])


def plot_attitude_error(timeData, dataSigmaBR):
    """Plot the attitude errors."""
    fig = plt.figure(num = 1, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(3):
        plt.plot(timeData, dataSigmaBR[:, idx],
                 color = simHelpers.getLineColor(idx, 3),
                 label=r'$\sigma_' + str(idx) + r'$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel(r'Attitude Error $\sigma_{B/R}$')

    return fig


def plot_rate_error(timeData, dataOmegaBR):
    """Plot the body angular velocity rate tracking errors."""
    fig = plt.figure(num = 2, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(3):
        plt.plot(timeData, dataOmegaBR[:, idx],
                 color = simHelpers.getLineColor(idx, 3),
                 label=r'$\omega_{BR,' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Rate Tracking Error (rad/s) ')

    return fig


def plot_rw_momenta(timeData, dataOmegaRw, RW, numRW):
    """Plot the RW momenta."""
    totMomentumNorm = []
    for j in range(len(timeData)):
        totMomentum = np.array([0,0,0])
        for idx in range(numRW):
            for k in range(3):
                totMomentum[k] = totMomentum[k] + dataOmegaRw[j, idx] * RW[idx].Js * RW[idx].gsHat_B[k][0]
        totMomentumNorm.append(np.linalg.norm(totMomentum))

    fig = plt.figure(num = 3, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRw[:, idx] * RW[idx].Js,
                 color = simHelpers.getLineColor(idx, numRW),
                 label=r'$H_{' + str(idx+1) + r'}$')
    plt.plot(timeData, totMomentumNorm, '--',
             label=r'$\|H\|$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Momentum (Nms)')

    return fig


def plot_DH(timeData, dataDH):
    """Plot the body angular velocity rate tracking errors."""
    fig = plt.figure(num = 4, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(3):
        plt.plot(timeData, dataDH[:, idx],
                 color = simHelpers.getLineColor(idx, 3),
                 label=r'$\Delta H_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Dumped momentum (Nms) ')

    return fig


def plot_rw_speeds(timeData, dataOmegaRW, numRW):
    """Plot the RW spin rates."""
    fig = plt.figure(num = 5, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    plt.figure(5)
    for idx in range(numRW):
        plt.plot(timeData, dataOmegaRW[:, idx] / macros.RPM,
                 color = simHelpers.getLineColor(idx, numRW),
                 label=r'$\Omega_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('RW Speed (RPM) ')

    return fig


def plot_thrImpulse(timeDataFSW, dataMap, numTh):
    """Plot the Thruster force values."""
    fig = plt.figure(num = 6, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(numTh):
        plt.plot(timeDataFSW, dataMap[:, idx],
                 color = simHelpers.getLineColor(idx, numTh),
                 label=r'$thrImpulse_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Impulse requested [Ns]')

    return fig


def plot_OnTimeRequest(timeData, dataOnTime, numTh):
    """Plot the thruster on time requests."""
    fig = plt.figure(num = 7, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(numTh):
        plt.plot(timeData, dataOnTime[:, idx],
                 color = simHelpers.getLineColor(idx, numTh),
                 label=r'$OnTimeRequest_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('OnTimeRequest [sec]')

    return fig


def plot_thrForce(timeDataFSW, dataThr, numTh):
    """Plot the Thruster force values."""
    fig = plt.figure(num = 8, clear = True)
    ax = fig.gca()
    ax.ticklabel_format(useOffset = False, style = 'plain')
    for idx in range(numTh):
        plt.plot(timeDataFSW, dataThr[idx],
                 color = simHelpers.getLineColor(idx, numTh),
                 label=r'$thrForce_{' + str(idx+1) + r'}$')
    plt.legend(loc='lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Thruster force [N]')

    return fig


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
              baseIndent: int = 3):

    numRW = len(rwPos)
    pad = "\t" * baseIndent

    rwTags, actTags, RWs = [], [], []

    for idx in range(numRW):
        n = idx + 1
        rw = rwFactory.create('Honeywell_HR16', rwAxes[idx], maxMomentum=100.)
        RWs.append(rw)
        pos = rwPos[idx]
        quat = quatAlignment(rwAxes[idx])

        rwTags.append(
f"""{pad}<body name = "rw{n}Spin" pos = "{pos[0]} {pos[1]} {pos[2]}" quat = "{quat}">
{pad}\t<joint name = "rw{n}Joint" pos = "0 0 0" axis = "0 0 1" ref = "0"/>
{pad}\t<inertial pos = "0 0 0" mass = "{rw.mass}" diaginertia = "{rw.Jt} {rw.Jt} {rw.Js}"/>
{pad}\t<geom name = "rw{n}Geom" type = "cylinder" size = "0.2 0.05" contype = "0" conaffinity = "0"/>
{pad}</body>""")

        actTags.append(f'\t\t<motor name = "rw{n}Act" joint = "rw{n}Joint"/>')

    return "\n".join(rwTags), "\n".join(actTags), RWs


def addThrustersXML(thrustLocs: list, 
                    thrustDirs: list,
                    thrFactory: simIncludeThruster,
                    maxThrust: float = 5.0,
                    baseIndent: int = 2):

    numTHRs = len(thrustLocs)
    pad = "\t" * baseIndent

    thrustTags, actTags, THRs = [], [], []

    for idx in range(numTHRs):
        n = idx + 1
        pos = thrustLocs[idx]
        dirVec = thrustDirs[idx]

        thr = thrFactory.create('MOOG_Monarc_5', pos, dirVec, MaxThrust = maxThrust)
        THRs.append(thr)

        quat = quatAlignment(dirVec)
        thrustTags.append(f'{pad}\t<site name = "thrusterSite{n}" pos = "{pos[0]} {pos[1]} {pos[2]}" quat = "{quat}"/>')
        actTags.append(f'{pad}<motor name = "thruster{n}" site = "thrusterSite{n}" gear = "0 0 1 0 0 0" ctrlrange = "0 5"/>')
    
    return "\n".join(thrustTags), "\n".join(actTags), THRs


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
    thrFactory = simIncludeThruster.thrusterFactory()

    rwBodies, rwActs, RWs = addRWsXML(rwPos, rwAxes, rwFactory)
    thrSites, thrActs, THRs = addThrustersXML(thrustLocs, thrustDirs, thrFactory)
 
    xml = f"""<mujoco model = "busWithRWsAndThrusters">
    <compiler angle = "radian" meshdir = ""/>
 
    <worldbody>
        <body name = "hub" pos = "0 0 0">
            <freejoint name = "busFree"/>
            <inertial pos = "0 0 0" mass = "{hubMass}" diaginertia = "{hubIxx} {hubIyy} {hubIzz}"/>
            <geom name = "hubVisual" type = "box" size = "1 1 1.28" rgba = "1 1 1 1"/>

{thrSites}

{rwBodies}

        </body>
    </worldbody>

    <actuator>
{rwActs}

{thrActs}
    </actuator>

</mujoco>"""

    return xml, RWs, THRs, rwFactory, thrFactory


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

    xmlString, RWs, THRs, rwFactory, thrFactory = makeMjXmlString()
    scene = mujoco.MJScene(xmlString)
    scene.ModelTag = "mujocoScene"
    sim.AddModelToTask(dynTaskName, scene)

    # -------------------------------------------------------------------------
    # 2) Retrieve spacecraft componenets
    # -------------------------------------------------------------------------
    busBody = scene.getBody("hub")
    numRWs = len(RWs)
    numTHRs = len(THRs)

    RWBodies = [scene.getBody(f"rw{i + 1}Spin") for i in range(numRWs)]
    RWJoints = [RWBodies[i].getScalarJoint(f"rw{i + 1}Joint") for i in range(numRWs)]
    RWActuators = [scene.getSingleActuator(f"rw{i + 1}Act") for i in range(numRWs)]
    THRActuators = [scene.getSingleActuator(f"thruster{i + 1}") for i in range(numTHRs)]

    # -------------------------------------------------------------------------
    # 3) Add gravity
    # -------------------------------------------------------------------------
    gravity = NBodyGravity.NBodyGravity()
    gravity.ModelTag = "gravity"
    scene.AddModelToDynamicsTask(gravity)

    muEarth = 0.3986004415e15  # [m^3/s^2]
    earthPm = pointMassGravityModel.PointMassGravityModel()
    earthPm.muBody = muEarth
    gravity.addGravitySource("earth", earthPm, isCentralBody = True)

    gravity.addGravityTarget("hub", busBody)

    # -------------------------------------------------------------------------
    # 4) Initial conditions
    # -------------------------------------------------------------------------
    oe = orbitalMotion.ClassicElements()
    rLEO = 7000. * 1000  # meters
    oe.a = rLEO
    oe.e = 0.0001
    oe.i = 0.0 * macros.D2R
    oe.Omega = 48.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 85.3 * macros.D2R
    rN, vN = orbitalMotion.elem2rv(muEarth, oe)

    # -------------------------------------------------------------------------
    # 5) Navitation and FSW
    # -------------------------------------------------------------------------
    simpleNavObj = simpleNav.SimpleNav()
    simpleNavObj.ModelTag = "simpleNav"
    simpleNavObj.scStateInMsg.subscribeTo(busBody.getCenterOfMass().stateOutMsg)
    sim.AddModelToTask(dynTaskName, simpleNavObj)

    inertial3DObj = inertial3D.inertial3D()
    inertial3DObj.ModelTag = "inertial3D"
    inertial3DObj.sigma_R0N = [0.0, 0.0, 0.0]
    sim.AddModelToTask(fswTaskName, inertial3DObj)

    attError = attTrackingError.attTrackingError()
    attError.ModelTag = "attErrorInertial3D"
    attError.attNavInMsg.subscribeTo(simpleNavObj.attOutMsg)
    attError.attRefInMsg.subscribeTo(inertial3DObj.attRefOutMsg)
    sim.AddModelToTask(fswTaskName, attError)

    mrpControl = mrpFeedback.mrpFeedback()
    mrpControl.ModelTag = "mrpFeedback"
    sim.AddModelToTask(fswTaskName, mrpControl)
    decayTime = 10.0
    xi = 1.0
    I = np.diag([1700, 1700, 1800])
    mrpControl.Ki = -1  # make value negative to turn off integral feedback
    mrpControl.P = 3*np.max(I)/decayTime
    mrpControl.K = (mrpControl.P/xi)*(mrpControl.P/xi)/np.max(I)
    mrpControl.integralLimit = 2. / mrpControl.Ki * 0.1

    controlAxes_B = [1, 0, 0, 0, 1, 0, 0, 0, 1]

    # add module that maps the Lr control torque into the RW motor torques
    rwMotorTorqueObj = rwMotorTorque.rwMotorTorque()
    rwMotorTorqueObj.ModelTag = "rwMotorTorque"
    sim.AddModelToTask(fswTaskName, rwMotorTorqueObj)
    # Make the RW control all three body axes
    rwMotorTorqueObj.controlAxes_B = controlAxes_B

    # Momentum dumping configuration
    thrDesatControl = thrMomentumManagement.thrMomentumManagement()
    thrDesatControl.ModelTag = "thrMomentumManagement"
    sim.AddModelToTask(fswTaskName, thrDesatControl)
    thrDesatControl.hs_min = 80   # Nms  :  maximum wheel momentum

    # setup the thruster force mapping module
    thrForceMappingObj = thrForceMapping.thrForceMapping()
    thrForceMappingObj.ModelTag = "thrForceMapping"
    sim.AddModelToTask(fswTaskName, thrForceMappingObj)
    thrForceMappingObj.controlAxes_B = controlAxes_B
    thrForceMappingObj.thrForceSign = 1
    thrForceMappingObj.angErrThresh = 3.15    # this needs to be larger than pi (180 deg) for the module to work in the momentum dumping scenario

    # setup the thruster momentum dumping module
    thrDump = thrMomentumDumping.thrMomentumDumping()
    thrDump.ModelTag = "thrDump"
    sim.AddModelToTask(fswTaskName, thrDump)
    thrDump.maxCounterValue = 100          # number of control periods (simulationTimeStepFsw) to wait between two subsequent on-times
    thrDump.thrMinFireTime = 0.02       

    fswRwParamMsg = rwFactory.getConfigMessage()   
    fswThrParamMsg = thrFactory.getConfigMessage()

    # -------------------------------------------------------------------------
    # 6) Distribute torques
    # -------------------------------------------------------------------------
    rwDistributor = RWTorqueDistributor(numRWs)
    rwDistributor.rwMotorTorqueInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
    sim.AddModelToTask(fswTaskName, rwDistributor)
    for i in range(numRWs):
        RWActuators[i].actuatorInMsg.subscribeTo(rwDistributor.torqueOutMsgs[i])

    thrDistributor = ThrusterOnTimeDistributor(numTHRs, maxThrust=5.0, controlPeriod=1.0)
    thrDistributor.onTimeInMsg.subscribeTo(thrDump.thrusterOnTimeOutMsg)
    sim.AddModelToTask(fswTaskName, thrDistributor)
    for i in range(numTHRs):
        THRActuators[i].actuatorInMsg.subscribeTo(thrDistributor.forceOutMsgs[i])

    # -------------------------------------------------------------------------
    # 6) Message Linking
    # -------------------------------------------------------------------------
    vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B=[1700,0,0, 0,1700,0, 0,0,1800])
    vcMsg = messaging.VehicleConfigMsg().write(vehicleConfigOut)
    mrpControl.vehConfigInMsg.subscribeTo(vcMsg)

    mrpControl.guidInMsg.subscribeTo(attError.attGuidOutMsg)
    mrpControl.rwParamsInMsg.subscribeTo(fswRwParamMsg)

    speedCombiner = RWSpeedCombiner(RWJoints)
    sim.AddModelToTask(fswTaskName, speedCombiner, 100)
    mrpControl.rwSpeedsInMsg.subscribeTo(speedCombiner.speedOutMsg)

    rwMotorTorqueObj.rwParamsInMsg.subscribeTo(fswRwParamMsg)
    thrForceMappingObj.vehConfigInMsg.subscribeTo(vcMsg)
    rwMotorTorqueObj.vehControlInMsg.subscribeTo(mrpControl.cmdTorqueOutMsg)

    thrDesatControl.rwSpeedsInMsg.subscribeTo(speedCombiner.speedOutMsg)
    thrDesatControl.rwConfigDataInMsg.subscribeTo(fswRwParamMsg)
    thrForceMappingObj.thrConfigInMsg.subscribeTo(fswThrParamMsg)
    thrForceMappingObj.cmdTorqueInMsg.subscribeTo(thrDesatControl.deltaHOutMsg)
    thrDump.thrusterConfInMsg.subscribeTo(fswThrParamMsg)   
    thrDump.deltaHInMsg.subscribeTo(thrDesatControl.deltaHOutMsg)
    thrDump.thrusterImpulseInMsg.subscribeTo(thrForceMappingObj.thrForceCmdOutMsg)

    # -------------------------------------------------------------------------
    # 7) Set up data recording
    # -------------------------------------------------------------------------
    numDataPoints = 5000
    samplingTime = unitTestSupport.samplingTime(simulationTime, simulationTimeStepDyn, numDataPoints)

    sNavRec = simpleNavObj.attOutMsg.recorder(samplingTime)
    sim.AddModelToTask(dynTaskName, sNavRec)

    dataRec = busBody.getCenterOfMass().stateOutMsg.recorder(samplingTime)
    sim.AddModelToTask(dynTaskName, dataRec)

    rwMotorLog = rwMotorTorqueObj.rwMotorTorqueOutMsg.recorder(samplingTime)
    sim.AddModelToTask(dynTaskName, rwMotorLog)

    attErrorLog = attError.attGuidOutMsg.recorder(samplingTime)
    sim.AddModelToTask(dynTaskName, attErrorLog)

    deltaHLog  = thrDesatControl.deltaHOutMsg.recorder(samplingTime)
    sim.AddModelToTask(dynTaskName, deltaHLog)

    thrMapLog = thrForceMappingObj.thrForceCmdOutMsg.recorder(samplingTime)
    sim.AddModelToTask(dynTaskName, thrMapLog)

    onTimeLog = thrDump.thrusterOnTimeOutMsg.recorder(samplingTime)
    sim.AddModelToTask(dynTaskName, onTimeLog)

    rwSpeedLogs = []
    for i in range(numRWs):
        rwSpeedLogs.append(RWJoints[i].stateDotOutMsg.recorder(samplingTime))
        sim.AddModelToTask(dynTaskName, rwSpeedLogs[i])

    thrForceLogs = []
    for i in range(numTHRs):
        thrForceLogs.append(THRActuators[i].actuatorInMsg.recorder(samplingTime))
        sim.AddModelToTask(dynTaskName, thrForceLogs[i])

    # -------------------------------------------------------------------------
    # 8) Running simulation
    # -------------------------------------------------------------------------
    sim.InitializeSimulation()

    busFree = busBody.getFreeJoint()
    busBody.setPosition(rN)
    busFree.setVelocity(vN)

    initialOmegas = [4000., 2000., 3500., 0.]
    for i in range(numRWs):
        omega_radps = initialOmegas[i] * macros.RPM
        RWJoints[i].setVelocity(omega_radps) 

    sim.ConfigureStopTime(macros.sec2nano(10.0))
    sim.ExecuteSimulation()

    thrDesatControl.Reset(macros.sec2nano(10.0))

    sim.ConfigureStopTime(simulationTime)
    sim.ExecuteSimulation()

    # -------------------------------------------------------------------------
    # 9) Post processing and plotting
    # -------------------------------------------------------------------------
    dataSigmaBR = attErrorLog.sigma_BR
    dataOmegaBR = attErrorLog.omega_BR_B
    dataOmegaRW = np.column_stack([np.squeeze(rwSpeedLogs[i].state) for i in range(numRWs)])
    dataDH = deltaHLog.torqueRequestBody
    dataMap = thrMapLog.thrForce
    dataOnTime = onTimeLog.OnTimeRequest

    dataThr = []
    for i in range(numTHRs):
        dataThr.append(thrForceLogs[i].input)

    np.set_printoptions(precision=16)

    timeData = rwMotorLog.times() * macros.NANO2SEC

    plt.close("all")
    figureList = {}
    figureList[fileName + "_attError"] = plot_attitude_error(timeData, dataSigmaBR)
    figureList[fileName + "_rateError"] = plot_rate_error(timeData, dataOmegaBR)
    figureList[fileName + "_rwMomenta"] = plot_rw_momenta(timeData, dataOmegaRW, RWs, numRWs)
    figureList[fileName + "_DH"] = plot_DH(timeData, dataDH)
    figureList[fileName + "_thrImpulse"] = plot_thrImpulse(timeData, dataMap, numTHRs)
    figureList[fileName + "_OnTimeReq"] = plot_OnTimeRequest(timeData, dataOnTime, numTHRs)
    figureList[fileName + "_thrForce"] = plot_thrForce(timeData, dataThr, numTHRs)

    if showPlots:
        plt.show()

    return figureList


class RWTorqueDistributor(sysModel.SysModel):
    def __init__(self, numRW):
        super().__init__()
        self.ModelTag = "rwTorqueDistributor"
        self.numRW = numRW
        self.rwMotorTorqueInMsg = messaging.ArrayMotorTorqueMsgReader()
        self.torqueOutMsgs = [messaging.SingleActuatorMsg() for _ in range(numRW)]

    def UpdateState(self, CurrentSimNanos):
        if self.rwMotorTorqueInMsg.isLinked():
            payload = self.rwMotorTorqueInMsg()
            for i in range(self.numRW):
                out = messaging.SingleActuatorMsgPayload()
                out.input = payload.motorTorque[i]
                self.torqueOutMsgs[i].write(out, CurrentSimNanos, self.moduleID)

class RWSpeedCombiner(sysModel.SysModel):
    def __init__(self, joints):
        super().__init__()
        self.ModelTag = "rwSpeedCombiner"
        self.joints = joints
        self.speedOutMsg = messaging.RWSpeedMsg()

    def UpdateState(self, CurrentSimNanos):
        speeds = [joint.stateDotOutMsg.read().state for joint in self.joints]
        payload = messaging.RWSpeedMsgPayload()
        payload.wheelSpeeds = speeds
        self.speedOutMsg.write(payload, CurrentSimNanos, self.moduleID)


class ThrusterOnTimeDistributor(sysModel.SysModel):
    def __init__(self, numTHRs, maxThrust, controlPeriod):
        super().__init__()
        self.ModelTag = "thrOnTimeDistributor"
        self.numTHRs = numTHRs
        self.maxThrust = maxThrust
        self.controlPeriod = controlPeriod  # seconds
        self.onTimeInMsg = messaging.THRArrayOnTimeCmdMsgReader()
        self.forceOutMsgs = [messaging.SingleActuatorMsg() for _ in range(numTHRs)]

    def UpdateState(self, CurrentSimNanos):
        if self.onTimeInMsg.isLinked():
            payload = self.onTimeInMsg()
            for i in range(self.numTHRs):
                out = messaging.SingleActuatorMsgPayload()
                out.input = self.maxThrust if payload.OnTimeRequest[i] > 0.0 else 0.0
                self.forceOutMsgs[i].write(out, CurrentSimNanos, self.moduleID)


if __name__ == "__main__":
    # --- XML TESTING ---
    
    xmlString, *_ = makeMjXmlString()
 
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sat_momentumDump.xml"), "w") as f:
        f.write(xmlString)

    run(True)