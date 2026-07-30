import os
import time
import signal
import csv
import sys
import argparse
import datetime
import matplotlib.pyplot as plt

import bota_driver
from Xeryon import *

#----------------------------------------------------------------------
#                          PARAMETER SETUP
#----------------------------------------------------------------------

# Shutdown flag + SIGINT handler to allow for full code termination
stop_flag = False
_sigint_count = 0

def signal_handler(signum, frame):
    global stop_flag, _sigint_count
    _sigint_count += 1
    stop_flag = True
    if _sigint_count >= 2:
        print("\nSecond Ctrl+C received — forcing immediate exit.")
        os._exit(1)
    else:
        print("\nCtrl+C received — stopping after the current step. Press Ctrl+C again to force-quit immediately.")

signal.signal(signal.SIGINT, signal_handler)

# Paths / config
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
config_path = "bota_binary_gen0.json"

output_dir = os.path.join(project_root, "data")
os.makedirs(output_dir, exist_ok = True)

CSV_HEADERS = [
    "timestamp",
    "voltage",
    "force_x", "force_y", "force_z",
    "torque_x", "torque_y", "torque_z",
]

# Experiment parameters
SAMPLE_DURATION = 10.0 # s
READING_FREQUENCY = 10000 # Hz
PRINTING_FREQUENCY = 1000 # Hz

# Preload/contact-detection parameters
APPROACH_DIRECTION = 1 # direction of motion (1 or -1)
APPROACH_SPEED = 0.5 # mm/s
CONTACT_FORCE_THRESHOLD = 10 # N **KEEP ABOVE NOISE FLOOR
MAX_APPROACH_TIME = 60.0 # s

# Adhesion measurement parameters
RETRACT_START_TIME = 2.0 # s
RETRACT_DISTANCE = 5 # mm
PAD_ACTIVATION_TIME = 60 # s

# Thermal-fault recovery
THERMAL_FAULT_COOLDOWN = 2.0 # s to wait before re-enabling after a thermal trip

# Voltage argument parsing (for repeatability/automation)
parser = argparse.ArgumentParser(description="Run one electroadhesion force trial.")
parser.add_argument("--voltage", type = float, default = None, help = "Electroadhesion pad voltage (kV) applied for this run.")
args = parser.parse_args()

if args.voltage is None:
    voltage = float(input("Enter electroadhesion pad voltage (kV) for this run: "))
else:
    voltage = args.voltage

# Timestamp for unique trials with similar v
run_id = f"{voltage:g}kV_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"

#----------------------------------------------------------------------
#                       THERMAL FAULT RECOVERY
#----------------------------------------------------------------------

def clear_thermal_fault(axisX, cooldown = THERMAL_FAULT_COOLDOWN):
    if axisX.isThermalProtection1() or axisX.isThermalProtection2():
        print(f"Thermal protection fault detected — stopping motion and waiting {cooldown:.1f}s before clearing...")
        try:
            axisX.stopScan()
        except Exception as e:
            print(f"Warning: failed to stop scan during fault recovery: {e}")

        time.sleep(cooldown)
        axisX.sendCommand("ENBL=1")
        time.sleep(0.3)  # give the controller a moment to report updated STAT

        if axisX.isThermalProtection1() or axisX.isThermalProtection2():
            raise RuntimeError(
                "Thermal protection fault is still active after sending "
                "ENBL=1. The drive may need a longer cooldown or a full "
                "hardware reset."
            )

        print("Thermal protection fault cleared — motor re-enabled.")
        return True
    return False

#----------------------------------------------------------------------
#                           CONTACT DETECTION
#----------------------------------------------------------------------

def wait_for_contact(axisX, 
                     bota_ft_sensor_driver, 
                     direction = APPROACH_DIRECTION, 
                     speed = APPROACH_SPEED, 
                     force_threshold = CONTACT_FORCE_THRESHOLD, 
                     timeout = MAX_APPROACH_TIME):
    global stop_flag

    axisX.setSpeed(speed)
    print("SSPD readback:", axisX.sendCommand("SSPD=?"))

    axisX.startScan(direction)

    start = time.perf_counter()
    contact_detected = False

    try:
        while time.perf_counter() - start < timeout:
            if stop_flag:
                break

            if axisX.isThermalProtection1() or axisX.isThermalProtection2():
                print("Thermal protection tripped during approach — aborting this approach.")
                break

            bota_frame = bota_ft_sensor_driver.read_frame()
            force = bota_frame.force
            print(f"Approaching... force = {force[0]:.3f}, {force[1]:.3f}, {force[2]:.3f} N")
            force_mag = force[2]

            if abs(force_mag) >= force_threshold:
                contact_detected = True
                break

            if axisX.isAtLeftEnd():
                print("Warning: reached stage travel limit before contact was detected.")
                break
    finally:
        axisX.stopScan()

    return contact_detected

#----------------------------------------------------------------------
#                           TRIAL/RECORDING
#----------------------------------------------------------------------

def run_trial(run_id, 
              axisX, 
              bota_ft_sensor_driver, 
              pad_activation_time = PAD_ACTIVATION_TIME):
    global stop_flag

    csv_path = os.path.join(output_dir, f"bota_log_{run_id}.csv")
    plot_path = os.path.join(output_dir, f"bota_plot_{run_id}.png")

    timestamps_list = []
    force_x_list, force_y_list, force_z_list = [], [], []
    torque_x_list, torque_y_list, torque_z_list = [], [], []

    # Clear out any thermal fault left over from a previous run before we
    # attempt to move at all.
    clear_thermal_fault(axisX)

    axisX.setSpeed(APPROACH_SPEED)

    # --- PRELOAD / CONTACT DETECTION ---
    print("Approaching pad until contact is detected...")
    contact_detected = wait_for_contact(axisX, bota_ft_sensor_driver)
    if not contact_detected:
        if axisX.isThermalProtection1() or axisX.isThermalProtection2():
            raise RuntimeError(
                "Failed to detect contact because a thermal protection "
                "fault tripped during the approach."
            )
        raise RuntimeError("Failed to detect contact")
    print("Contact detected — pad is preloaded.")

    # if not bota_ft_sensor_driver.deactivate():
    #     raise RuntimeError("Failed to deactivate sensor before tare")
    # if not bota_ft_sensor_driver.tare():
    #     raise RuntimeError("Failed to tare sensor")
    # if not bota_ft_sensor_driver.activate():
    #     raise RuntimeError("Failed to reactivate sensor after tare")

    if stop_flag:
        print("Stop requested — skipping recording.")
        return None

    # --- PAD ACTIVATION ---
    print(f"Waiting {pad_activation_time:.1f}s for adhesion to stabilize...")
    activation_start = time.perf_counter()
    last_activation_print = activation_start
    while time.perf_counter() - activation_start < pad_activation_time and not stop_flag:
        now = time.perf_counter()
        if now - last_activation_print >= 5.0:
            remaining = pad_activation_time - (now - activation_start)
            print(f"  ...{remaining:.0f}s remaining")
            last_activation_print = now
        time.sleep(0.05)

    if stop_flag:
        print("Stop requested — skipping recording.")
        return None

    retract_direction = -APPROACH_DIRECTION
    retract_duration = RETRACT_DISTANCE / (APPROACH_SPEED) # Time instead of distance in case of stage error

    # --- RECORDING ---
    print(f"\n=== Run {run_id} (voltage = {voltage} kV) ===")
    print(f"Logging data to: {csv_path}")
    print(f"Recording for {SAMPLE_DURATION:.1f}s")

    start_time = time.perf_counter()
    last_print_time = start_time
    next_execution_time = start_time
    rows_written = 0
    last_timestamp = None

    retract_started = False
    retract_stopped = False
    retract_start_clock = None
    thermal_fault_during_recording = False

    axisX.setSpeed(APPROACH_SPEED)

    with open(csv_path, mode = "w", newline = "") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADERS)

        while time.perf_counter() - start_time < SAMPLE_DURATION and not stop_flag:
            elapsed = time.perf_counter() - start_time

            # Retract after the specified wait time
            if not retract_started and not thermal_fault_during_recording and elapsed >= RETRACT_START_TIME:
                print(f"t={elapsed:.2f}s — starting retract...")
                axisX.startScan(retract_direction)
                retract_started = True
                retract_start_clock = time.perf_counter()

            # Watch for a thermal fault at any point during the recording
            # window (not just during the retract) — the fault can trip
            # after the retract has already finished.
            if (not thermal_fault_during_recording
                    and (axisX.isThermalProtection1() or axisX.isThermalProtection2())):
                print(f"t={elapsed:.2f}s — thermal protection tripped during recording.")
                try:
                    axisX.stopScan()
                except Exception as e:
                    print(f"Warning: failed to stop scan after thermal fault: {e}")
                retract_stopped = True  # don't let the retract-stop logic try to act further
                thermal_fault_during_recording = True

            # Stop retract after the specified duration
            if retract_started and not retract_stopped and time.perf_counter() - retract_start_clock >= retract_duration:
                axisX.stopScan()
                retract_stopped = True
                print(f"t = {elapsed:.2f}s: retract complete.")

            bota_frame = bota_ft_sensor_driver.read_frame()

            status = bota_frame.status
            force = bota_frame.force
            torque = bota_frame.torque
            timestamp = bota_frame.timestamp

            if timestamp != last_timestamp:
                csv_writer.writerow([
                    timestamp,
                    voltage,
                    force[0], force[1], force[2],
                    torque[0], torque[1], torque[2],
                ])
                rows_written += 1
                last_timestamp = timestamp

                timestamps_list.append(timestamp)
                force_x_list.append(force[0])
                force_y_list.append(force[1])
                force_z_list.append(force[2])
                torque_x_list.append(torque[0])
                torque_y_list.append(torque[1])
                torque_z_list.append(torque[2])

            current_time = time.perf_counter()
            if current_time - last_print_time >= 1.0 / PRINTING_FREQUENCY:
                print(f"{rows_written} frames logged... (t={current_time - start_time:.2f}s)")
                last_print_time = current_time

            next_execution_time += 1.0 / READING_FREQUENCY
            sleep_time = max(0, next_execution_time - time.perf_counter())
            time.sleep(sleep_time)

    # Check in case recording was terminated
    if retract_started and not retract_stopped:
        axisX.stopScan()

    if stop_flag:
        print("Stop requested during recording — data collected so far was still logged.")

    if thermal_fault_during_recording:
        print("Note: a thermal protection fault occurred mid-retract during this run. "
              "The recorded data after that point may not reflect an actual retract move.")
        clear_thermal_fault(axisX)

    # Average adhesion force calculation
    if force_x_list:
        force_magnitudes = [
            (fx ** 2 + fy ** 2 + fz ** 2) ** 0.5
            for fx, fy, fz in zip(force_x_list, force_y_list, force_z_list)
        ]
        avg_adhesion_force = sum(force_magnitudes) / len(force_magnitudes)
        print(f"Average adhesion force: {avg_adhesion_force:.4f} N "
              f"(over {len(force_magnitudes)} samples)")
    else:
        avg_adhesion_force = None
        print("No samples collected — cannot compute average adhesion force.")

    print(f"Run {run_id} complete. {rows_written} frames written to {csv_path}")

    # --- PLOTTING ---
    if len(timestamps_list) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        #axes[0].plot(timestamps_list, force_x_list, label="Force X")
        #axes[0].plot(timestamps_list, force_y_list, label="Force Y")
        axes[0].plot(timestamps_list, force_z_list, label="Force Z")
        axes[0].set_ylabel("Force (N)")
        axes[0].set_title(f"Force over Time — {voltage} kV")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(timestamps_list, torque_x_list, label="Torque X")
        axes[1].plot(timestamps_list, torque_y_list, label="Torque Y")
        axes[1].plot(timestamps_list, torque_z_list, label="Torque Z")
        axes[1].set_xlabel("Timestamp")
        axes[1].set_ylabel("Torque (Nm)")
        axes[1].set_title(f"Torque over Time — {voltage} kV")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")
        plt.close(fig)  # don't block waiting for a window to be closed —
                         # the PNG is already saved; open it whenever you like.
    else:
        print(f"Run {run_id}: not enough data collected to plot.")
    
    return avg_adhesion_force

#----------------------------------------------------------------------
#                            MAIN PROGRAM
#----------------------------------------------------------------------
controller = None
axisX = None
bota_ft_sensor_driver = None

try:
    # --- XERYON CONTROLLER SETUP ---
    controller = Xeryon("COM5", 115200)
    axisX = controller.addAxis(Stage.XLS_312, "Z")
    controller.start()
    clear_thermal_fault(axisX)  # clear any fault left over from a previous session
    #axisX.findIndex()
    if not axisX.isEncoderValid():
        print(
            "Warning: encoder index was not found. The stage may not be able to move accurately without a valid index."
        )
    axisX.setUnits(Units.mm)

    # --- BOTA DRIVER SETUP ---
    bota_ft_sensor_driver = bota_driver.BotaDriver(config_path)
    print(f" >>>>>>>>>>> BotaDriver version: {bota_ft_sensor_driver.get_driver_version_string()} <<<<<<<<<<<< ")

    if not bota_ft_sensor_driver.configure():
        raise RuntimeError("Failed to configure driver")
    if not bota_ft_sensor_driver.tare():
        raise RuntimeError("Failed to tare sensor")
    if not bota_ft_sensor_driver.activate():
        raise RuntimeError("Failed to activate driver")

    # --- TRIAL ---
    avg_adhesion_force = run_trial(run_id, axisX, bota_ft_sensor_driver)
    if avg_adhesion_force is not None:
        print(f"\nResult for run {run_id}: average adhesion force = {avg_adhesion_force:.4f} N")

    print("Completion WITHOUT errors.")

except Exception as e:
    print(f"FATAL: {e}")
    print("Completion WITH errors.")

finally:
    if axisX is not None:
        try:
            axisX.stopScan()
        except Exception as e:
            print(f"Warning: failed to stop scan: {e}")
        try:
            axisX.setSetting("ENBL", 0)  # explicitly cut motor drive power
        except Exception as e:
            print(f"Warning: failed to disable motor drive (ENBL=0): {e}")

    if controller is not None:
        try:
            controller.stop()
        except Exception as e:
            print(f"Warning: failed to cleanly stop Xeryon controller: {e}")

    if bota_ft_sensor_driver is not None:
        try:
            bota_ft_sensor_driver.deactivate()
        except Exception as e:
            print(f"Warning: failed to deactivate Bota sensor driver: {e}")
        try:
            bota_ft_sensor_driver.shutdown()  # closes the sensor's COM port
        except Exception as e:
            print(f"Warning: failed to shut down Bota sensor driver (COM port may still be open): {e}")

    print("EXITING PROGRAM")
    sys.exit(0)