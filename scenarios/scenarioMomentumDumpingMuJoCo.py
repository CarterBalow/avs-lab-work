import os
import matplotlib.pyplot as plt
import numpy as np

from Basilisk.utilities import SimulationBaseClass, macros, orbitalMotion, simHelpers, simIncludeRW, simIncludeThruster
from Basilisk.simulation import NBodyGravity, mujoco, pointMassGravityModel, simpleNav, thrOnTimeToForce
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
                 label = r'$\sigma_' + str(idx) + r'$')
    plt.legend(loc = 'lower right')
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
                 label = r'$\omega_{BR,' + str(idx+1) + r'}$')
    plt.legend(loc = 'lower right')
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
                 label = r'$H_{' + str(idx+1) + r'}$')
    plt.plot(timeData, totMomentumNorm, '--',
             label = r'$\|H\|$')
    plt.legend(loc = 'lower right')
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
                 label = r'$\Delta H_{' + str(idx+1) + r'}$')
    plt.legend(loc = 'lower right')
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
                 label = r'$\Omega_{' + str(idx+1) + r'}$')
    plt.legend(loc = 'lower right')
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
                 label = r'$thrImpulse_{' + str(idx+1) + r'}$')
    plt.legend(loc = 'lower right')
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
                 label = r'$OnTimeRequest_{' + str(idx+1) + r'}$')
    plt.legend(loc = 'lower right')
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
                 label = r'$thrForce_{' + str(idx+1) + r'}$')
    plt.legend(loc = 'lower right')
    plt.xlabel('Time [min]')
    plt.ylabel('Thruster force [N]')

    return fig


def addRWsXML(rwPos: list, 
              rwAxes: list,
              rwFactory: simIncludeRW, 
              maxMomentum: float = 100.,
              baseIndent: int = 3):

    numRW = len(rwPos)
    pad = "\t" * baseIndent
    varRWModel = messaging.BalancedWheels

    rwTags, actTags, RWs = [], [], []

    for idx in range(numRW):
        n = idx + 1
        pos = rwPos[idx]
        rwAxis = rwAxes[idx]

        rw = rwFactory.create('Honeywell_HR16', rwAxis, maxMomentum = maxMomentum, RWModel = varRWModel)
        RWs.append(rw)

        rwTags.append(
f"""{pad}<body name = "rw{n}Spin" pos = "{pos[0]} {pos[1]} {pos[2]}" zaxis = "{rwAxis[0]} {rwAxis[1]} {rwAxis[2]}">
{pad}\t<joint name = "rw{n}Joint" pos = "0 0 0" axis = "0 0 1" ref = "0"/>
{pad}\t<inertial pos = "0 0 0" mass = "{rw.mass}" diaginertia = "{rw.Jt} {rw.Jt} {rw.Js}"/>
{pad}\t<geom name = "rw{n}Geom" type = "cylinder" size = "0.2 0.05" contype = "0" conaffinity = "0"/>
{pad}</body>""")
        actTags.append(f'\t\t<motor name = "rw{n}Act" joint = "rw{n}Joint" ctrlrange = "{-rw.u_max} {rw.u_max}" ctrllimited = "true"/>')

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

        thrustTags.append(f'{pad}\t<site name = "thrusterSite{n}" pos = "{pos[0]} {pos[1]} {pos[2]}" zaxis = "{dirVec[0]} {dirVec[1]} {dirVec[2]}"/>')
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
    scene.extraEoMCall = True
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

    muEarth = 0.3986004415e15  # [m^3/s^2]
    earthPm = pointMassGravityModel.PointMassGravityModel()
    earthPm.muBody = muEarth
    gravity.addGravitySource("earth", earthPm, isCentralBody = True)

    gravity.addGravityTarget("hub", busBody)

    scene.AddModelToDynamicsTask(gravity)

    # -------------------------------------------------------------------------
    # 4) Initial conditions
    # -------------------------------------------------------------------------
    oe = orbitalMotion.ClassicElements()
    rLEO = 7000. * 1000  # meters
    oe.a = rLEO
    oe.e = 0.0001
    oe.i = 33.3 * macros.D2R
    oe.Omega = 148.2 * macros.D2R
    oe.omega = 347.8 * macros.D2R
    oe.f = 335 * macros.D2R
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
    mrpControl.Ki = -1
    mrpControl.P = 3 * np.max(I) / decayTime
    mrpControl.K = (mrpControl.P/xi) * (mrpControl.P/xi) / np.max(I)
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
    # 6) Distribute torques/
    # -------------------------------------------------------------------------
    rwDistributor = RWTorqueDistributor(numRWs)
    rwDistributor.rwMotorTorqueInMsg.subscribeTo(rwMotorTorqueObj.rwMotorTorqueOutMsg)
    sim.AddModelToTask(fswTaskName, rwDistributor)
    for i in range(numRWs):
        RWActuators[i].actuatorInMsg.subscribeTo(rwDistributor.torqueOutMsgs[i])

    thrForceConverter = MuJoCoJDynamicThruster(THRs, simulationTimeStepDyn * macros.NANO2SEC)
    scene.AddModelToDynamicsTask(thrForceConverter)
    thrForceConverter.onTimeInMsg.subscribeTo(thrDump.thrusterOnTimeOutMsg)

    for i in range(numTHRs):
        THRActuators[i].actuatorInMsg.subscribeTo(thrForceConverter.thrusterForceOutMsgs[i])

    # -------------------------------------------------------------------------
    # 6) Message Linking
    # -------------------------------------------------------------------------
    vehicleConfigOut = messaging.VehicleConfigMsgPayload(ISCPntB_B = [1700,0,0, 0,1700,0, 0,0,1800])
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
    samplingTime = simHelpers.samplingTime(simulationTime, simulationTimeStepDyn, numDataPoints)

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
    figureList[fileName + "_rwSpeeds"] = plot_rw_speeds(timeData, dataOmegaRW, numRWs)
    figureList[fileName + "_thrImpulse"] = plot_thrImpulse(timeData, dataMap, numTHRs)
    figureList[fileName + "_OnTimeReq"] = plot_OnTimeRequest(timeData, dataOnTime, numTHRs)
    figureList[fileName + "_thrForce"] = plot_thrForce(timeData, dataThr, numTHRs)

    np.savez(
        "new_run.npz",
        timeData=timeData,
        dataSigmaBR=dataSigmaBR,
        dataOmegaBR=dataOmegaBR,
        dataDH=dataDH,
        dataMap=dataMap,
        dataOnTime=dataOnTime,
        dataThr=np.array(dataThr),
        dataOmegaRW=dataOmegaRW,
    )

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



# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ------------------- THRUSTER DYNAMIC EFFECTOR MUJOCO --------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------

class THRRampPoint:
    def __init__(self, timeDelta: float, thrustFactor: float, ispFactor: float = 1.0):
        self.TimeDelta = timeDelta
        self.ThrustFactor = thrustFactor
        self.IspFactor = ispFactor

class THROpState:
    def __init__(self):
        self.ThrustOnCmd = 0.0
        self.ThrustFactor = 0.0
        self.IspFactor = 0.0
        self.ThrusterStartTime = 0.0
        self.PreviousIterTime = 0.0
        self.ThrustOnRampTime = 0.0
        self.ThrustOffRampTime = 0.0
        self.ThrustOnSteadyTime = 0.0
        self.totalOnTime = 0.0

def extractConfig(thr, index):
    def get(name, default):
        val = getattr(thr, name, None)
        if val is None:
            print(f"thruster {index}: {name} not found on config obj")
            return default
        return val

    maxThrust = get("MaxThrust", 1.0)
    minOnTime = get("MinOnTime", 0.0)
    steadyIsp = get("steadyIsp", 226.0)
    maxSwirlTorque = get("MaxSwirlTorque", 0.0)

    onRampRaw = getattr(thr, "ThrusterOnRamp", None) or []
    offRampRaw = getattr(thr, "ThrusterOffRamp", None) or []

    onRamp = [THRRampPoint(p.TimeDelta, p.ThrustFactor, getattr(p, "IspFactor", 1.0)) for p in onRampRaw]
    offRamp = [THRRampPoint(p.TimeDelta, p.ThrustFactor, getattr(p, "IspFactor", 1.0)) for p in offRampRaw]

    return dict(
        MaxThrust = maxThrust,
        MinOnTime = minOnTime,
        steadyIsp = steadyIsp,
        MaxSwirlTorque = maxSwirlTorque,
        ThrusterOnRamp = onRamp,
        ThrusterOffRamp = offRamp,
    )


class MuJoCoJDynamicThruster(sysModel.SysModel):
    def __init__(self, thrConfigObjs, dynTimeStep: float):
        super().__init__()
        self.ModelTag = "MuJoCoDynamicThruster"

        self.numTHR = len(thrConfigObjs)
        self.cfg = [extractConfig(t, i) for i, t in enumerate(thrConfigObjs)]
        self.ops = [THROpState() for _ in range(self.numTHR)]

        self.dt = dynTimeStep

        self.onTimeInMsg = messaging.THRArrayOnTimeCmdMsgReader()
        self.thrusterForceOutMsgs = [messaging.SingleActuatorMsg() for _ in range(self.numTHR)]

        self.newThrustCmds = [0.0] * self.numTHR
        self.prevCommandTime = None
        self.prevCallTime = None

    def ReadInputs(self):
        if not self.onTimeInMsg.isLinked():
            return False
        payload = self.onTimeInMsg()
        writtenTime = self.onTimeInMsg.timeWritten()
        dataGood = self.onTimeInMsg.isWritten()
        if not dataGood or writtenTime == self.prevCommandTime:
            return False
        self.prevCommandTime = writtenTime
        for i in range(self.numTHR):
            self.newThrustCmds[i] = payload.OnTimeRequest[i]
        return True

    def configureThrustRequests(self, currentTime):
        for i in range(self.numTHR):
            cmd = self.newThrustCmds[i]
            op = self.ops[i]
            minOnTime = self.cfg[i]["MinOnTime"]

            if cmd >= minOnTime:
                op.ThrustOnCmd = cmd
            else:
                op.ThrustOnCmd = cmd if op.ThrustFactor > 0.0 else 0.0

            op.ThrusterStartTime = currentTime
            op.PreviousIterTime = currentTime
            op.ThrustOnRampTime = 0.0
            op.ThrustOnSteadyTime = 0.0
            op.ThrustOffRampTime = 0.0

            self.newThrustCmds[i] = 0.0

    @staticmethod
    def thrFactorToTime(op, ramp):
        last = ramp[-1]
        rampTime = last.TimeDelta
        diff = last.ThrustFactor - op.ThrustFactor
        rampDirection = 1.0 if diff >= 0 else -1.0

        prevValidThrFactor = 0.0 if rampDirection > 0 else 1.0
        prevValidDelta = 0.0

        for pt in ramp:
            pointCheck = (pt.ThrustFactor <= op.ThrustFactor) if rampDirection > 0 else (pt.ThrustFactor >= op.ThrustFactor)
            if pointCheck:
                prevValidThrFactor = pt.ThrustFactor
                prevValidDelta = pt.TimeDelta
                continue
            denom = (pt.ThrustFactor - prevValidThrFactor)
            if denom == 0.0:
                rampTime = prevValidDelta
            else:
                rampTime = (pt.TimeDelta - prevValidDelta) / denom * (op.ThrustFactor - prevValidThrFactor) + prevValidDelta
            rampTime = max(rampTime, 0.0)
            break

        return rampTime

    def computeThrusterFire(self, i, currentTime):
        op = self.ops[i]
        onRamp = self.cfg[i]["ThrusterOnRamp"]

        if op.ThrustOnRampTime == 0.0 and len(onRamp) > 0:
            op.ThrustOnRampTime = self.thrFactorToTime(op, onRamp)

        localOnRamp = max((currentTime - op.PreviousIterTime) + op.ThrustOnRampTime, 0.0)

        prevTHR, prevIsp, prevDelta = 0.0, 0.0, 0.0
        for pt in onRamp:
            if localOnRamp < pt.TimeDelta:
                denomT = (pt.TimeDelta - prevDelta)
                op.ThrustFactor = (pt.ThrustFactor - prevTHR) / denomT * (localOnRamp - prevDelta) + prevTHR
                op.IspFactor = (pt.IspFactor - prevIsp) / denomT * (localOnRamp - prevDelta) + prevIsp
                op.ThrustOnRampTime = localOnRamp
                op.totalOnTime += currentTime - op.PreviousIterTime
                op.PreviousIterTime = currentTime
                return 
            prevTHR, prevIsp, prevDelta = pt.ThrustFactor, pt.IspFactor, pt.TimeDelta

        op.ThrustOnSteadyTime += currentTime - op.PreviousIterTime
        op.totalOnTime += currentTime - op.PreviousIterTime
        op.PreviousIterTime = currentTime
        op.ThrustFactor = 1.0
        op.IspFactor = 1.0
        op.ThrustOffRampTime = 0.0

    def computeThrusterShut(self, i, currentTime):
        op = self.ops[i]
        offRamp = self.cfg[i]["ThrusterOffRamp"]

        if op.ThrustOffRampTime == 0.0 and len(offRamp) > 0:
            op.ThrustOffRampTime = self.thrFactorToTime(op, offRamp)

        localOffRamp = max((currentTime - op.PreviousIterTime) + op.ThrustOffRampTime, 0.0)

        prevTHR, prevIsp, prevDelta = 1.0, 1.0, 0.0
        for pt in offRamp:
            if localOffRamp < pt.TimeDelta:
                denomT = (pt.TimeDelta - prevDelta)
                op.ThrustFactor = (pt.ThrustFactor - prevTHR) / denomT * (localOffRamp - prevDelta) + prevTHR
                op.IspFactor = (pt.IspFactor - prevIsp) / denomT * (localOffRamp - prevDelta) + prevIsp
                op.ThrustOffRampTime = localOffRamp
                op.PreviousIterTime = currentTime
                return
            prevTHR, prevIsp, prevDelta = pt.ThrustFactor, pt.IspFactor, pt.TimeDelta

        op.ThrustFactor = 0.0
        op.IspFactor = 0.0
        op.ThrustOnRampTime = 0.0

    def UpdateState(self, CurrentSimNanos):
        currentTime = CurrentSimNanos * macros.NANO2SEC

        if self.ReadInputs():
            self.configureThrustRequests(currentTime)

        if self.prevCallTime is None:
            dt = self.dt
        else:
            dt = currentTime - self.prevCallTime
        self.prevCallTime = currentTime

        for i in range(self.numTHR):
            op = self.ops[i]
            stillFiring = (op.ThrustOnCmd + op.ThrusterStartTime - currentTime) >= -dt * 10E-10 and op.ThrustOnCmd > 0.0

            if stillFiring:
                self.computeThrusterFire(i, currentTime)
            elif op.ThrustFactor > 0.0:
                self.computeThrusterShut(i, currentTime)

            force = self.cfg[i]["MaxThrust"] * op.ThrustFactor

            payload = messaging.SingleActuatorMsgPayload()
            payload.input = force
            self.thrusterForceOutMsgs[i].write(payload, CurrentSimNanos, self.moduleID)



if __name__ == "__main__":
    # --- XML TESTING ---
    
    xmlString, *_ = makeMjXmlString()
 
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)), "sat_momentumDump.xml"), "w") as f:
        f.write(xmlString)

    run(True)