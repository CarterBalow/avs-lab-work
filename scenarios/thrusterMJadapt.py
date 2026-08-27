# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# ------------------- THRUSTER DYNAMIC EFFECTOR MUJOCO --------------------
# -------------------------------------------------------------------------
# -------------------------------------------------------------------------
# The classes and helpers below reproduce, in simplified form, the on/off thrust ramping and
# minimum-on-time behavior normally handled internally by thrusterDynamicEffector, so this same
# behavior can be applied when thrusters are instead modeled as MuJoCo force actuators.

class THRRampPoint:
    """Single point (time, thrust factor, Isp factor) on a thruster's on/off ramp curve"""
    def __init__(self, timeDelta: float, thrustFactor: float, ispFactor: float = 1.0):
        self.TimeDelta = timeDelta
        self.ThrustFactor = thrustFactor
        self.IspFactor = ispFactor

class THROpState:
    """Mutable per-thruster operating state tracked across UpdateState calls"""
    def __init__(self):
        self.ThrustOnCmd = 0.0          # commanded on-time [s] for current firing
        self.ThrustFactor = 0.0         # current fraction (0-1) of MaxThrust delivered
        self.IspFactor = 0.0            # current fraction (0-1) of steady-state Isp
        self.ThrusterStartTime = 0.0    # sim time [s] the current firing command began
        self.PreviousIterTime = 0.0     # sim time [s] at the previous UpdateState call
        self.ThrustOnRampTime = 0.0     # elapsed time [s] along the on-ramp curve
        self.ThrustOffRampTime = 0.0    # elapsed time [s] along the off-ramp curve
        self.ThrustOnSteadyTime = 0.0   # cumulative time [s] at full steady thrust
        self.totalOnTime = 0.0          # cumulative time [s] this thruster has been firing

def extractConfig(thr, index):
    """Pulls fields off a thruster config object with safe defaults if a field is missing"""
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

    # Ramp tables may not be defined; default to instantaneous on/off response
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


# -------------------------------------------------------------------------
# MuJoCoJDynamicThruster: custom sys model to convert thruster on-time commands into ramped
#   force outputs for MuJoCo force actuators, reproducing thrusterDynamicEffector-style on/off
#   ramping and minimum on-time behavior
#   INPUTS:
#       - thrConfigObjs: list of thruster config objects (from thrusterFactory.create())
#       - dynTimeStep: fallback dynamics time step [s], used only before the first UpdateState call
# -------------------------------------------------------------------------
class MuJoCoDynamicThruster(sysModel.SysModel):
    def __init__(self, thrConfigObjs, dynTimeStep: float):
        super().__init__()
        self.ModelTag = "MuJoCoDynamicThruster"

        self.numTHR = len(thrConfigObjs)
        self.cfg = [extractConfig(t, i) for i, t in enumerate(thrConfigObjs)]  # per-thruster config
        self.ops = [THROpState() for _ in range(self.numTHR)]  # per-thruster operating state

        self.dt = dynTimeStep

        self.onTimeInMsg = messaging.THRArrayOnTimeCmdMsgReader()  # commanded on-times, all thrusters
        self.thrusterForceOutMsgs = [messaging.SingleActuatorMsg() for _ in range(self.numTHR)]  # per-thruster force

        self.newThrustCmds = [0.0] * self.numTHR  # staging buffer for newly received commands
        self.prevCommandTime = None  # used to detect a newly written on-time message
        self.prevCallTime = None     # used to compute dt between UpdateState calls

    def ReadInputs(self):
        """Checks for and stages a newly written on-time command message, returns True if found"""
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
        """Latches newly staged on-time commands into each thruster's operating state"""
        for i in range(self.numTHR):
            cmd = self.newThrustCmds[i]
            op = self.ops[i]
            minOnTime = self.cfg[i]["MinOnTime"]

            # Honor command if it meets MinOnTime, or if it extends an already-firing thruster
            if cmd >= minOnTime:
                op.ThrustOnCmd = cmd
            else:
                op.ThrustOnCmd = cmd if op.ThrustFactor > 0.0 else 0.0

            # Reset this thruster's firing timeline to start from "now"
            op.ThrusterStartTime = currentTime
            op.PreviousIterTime = currentTime
            op.ThrustOnRampTime = 0.0
            op.ThrustOnSteadyTime = 0.0
            op.ThrustOffRampTime = 0.0

            self.newThrustCmds[i] = 0.0

    @staticmethod
    def thrFactorToTime(op, ramp):
        """Inverts a ramp curve to find the elapsed ramp-time matching the current thrust factor"""
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
            # Found bracketing segment, linearly interpolate to find matching ramp time
            denom = (pt.ThrustFactor - prevValidThrFactor)
            if denom == 0.0:
                rampTime = prevValidDelta
            else:
                rampTime = (pt.TimeDelta - prevValidDelta) / denom * (op.ThrustFactor - prevValidThrFactor) + prevValidDelta
            rampTime = max(rampTime, 0.0)
            break

        return rampTime

    def computeThrusterFire(self, i, currentTime):
        """Advances thruster i's on-ramp state while it is commanded to fire"""
        op = self.ops[i]
        onRamp = self.cfg[i]["ThrusterOnRamp"]

        # If just starting to ramp on, find correct starting point (handles direction reversal)
        if op.ThrustOnRampTime == 0.0 and len(onRamp) > 0:
            op.ThrustOnRampTime = self.thrFactorToTime(op, onRamp)

        localOnRamp = max((currentTime - op.PreviousIterTime) + op.ThrustOnRampTime, 0.0)

        prevTHR, prevIsp, prevDelta = 0.0, 0.0, 0.0
        for pt in onRamp:
            if localOnRamp < pt.TimeDelta:
                # Within this ramp segment, linearly interpolate thrust/Isp factors
                denomT = (pt.TimeDelta - prevDelta)
                op.ThrustFactor = (pt.ThrustFactor - prevTHR) / denomT * (localOnRamp - prevDelta) + prevTHR
                op.IspFactor = (pt.IspFactor - prevIsp) / denomT * (localOnRamp - prevDelta) + prevIsp
                op.ThrustOnRampTime = localOnRamp
                op.totalOnTime += currentTime - op.PreviousIterTime
                op.PreviousIterTime = currentTime
                return 
            prevTHR, prevIsp, prevDelta = pt.ThrustFactor, pt.IspFactor, pt.TimeDelta

        # Ramp fully traversed (or empty): thruster at full steady-state thrust
        op.ThrustOnSteadyTime += currentTime - op.PreviousIterTime
        op.totalOnTime += currentTime - op.PreviousIterTime
        op.PreviousIterTime = currentTime
        op.ThrustFactor = 1.0
        op.IspFactor = 1.0
        op.ThrustOffRampTime = 0.0

    def computeThrusterShut(self, i, currentTime):
        """Advances thruster i's off-ramp state while it is ramping down after commanded shutoff"""
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

        # Off-ramp fully traversed (or empty): thruster now fully off
        op.ThrustFactor = 0.0
        op.IspFactor = 0.0
        op.ThrustOnRampTime = 0.0

    def UpdateState(self, CurrentSimNanos):
        """Advances all thrusters' ramp states by one step and publishes delivered forces"""
        currentTime = CurrentSimNanos * macros.NANO2SEC

        if self.ReadInputs():
            self.configureThrustRequests(currentTime)

        # dt falls back to nominal dynamics step on the very first call
        if self.prevCallTime is None:
            dt = self.dt
        else:
            dt = currentTime - self.prevCallTime
        self.prevCallTime = currentTime

        for i in range(self.numTHR):
            op = self.ops[i]
            # Still within commanded firing window (small numerical tolerance based on dt)
            stillFiring = (op.ThrustOnCmd + op.ThrusterStartTime - currentTime) >= -dt * 10E-10 and op.ThrustOnCmd > 0.0

            if stillFiring:
                self.computeThrusterFire(i, currentTime)
            elif op.ThrustFactor > 0.0:
                self.computeThrusterShut(i, currentTime)

            # Delivered force is rated max thrust scaled by current (possibly ramping) factor
            force = self.cfg[i]["MaxThrust"] * op.ThrustFactor

            payload = messaging.SingleActuatorMsgPayload()
            payload.input = force
            self.thrusterForceOutMsgs[i].write(payload, CurrentSimNanos, self.moduleID)