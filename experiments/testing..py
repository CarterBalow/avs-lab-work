import os
import time
import signal
import csv
import bota_driver
import sys
import matplotlib.pyplot as plt

# Flag for graceful shutdown
stop_flag = False

def signal_handler(signum, frame):
    global stop_flag
    stop_flag = True

# Register signal handler for graceful shutdown
signal.signal(signal.SIGINT, signal_handler)

# Project root directory (two levels up from this script)
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Config file path
config_path = "bota_binary_gen0.json"

# Output directory for CSV and plot files
output_dir = os.path.join(project_root, "data")
os.makedirs(output_dir, exist_ok = True)
csv_path = os.path.join(output_dir, f"bota_log.csv")
plot_path = os.path.join(output_dir, f"bota_plot.png")

CSV_HEADERS = [
    "timestamp",
    "force_x", "force_y", "force_z",
    "torque_x", "torque_y", "torque_z",
]

##################
## DRIVER USAGE ##
##################

# In-memory buffer allocation
timestamps_list = []
force_x_list, force_y_list, force_z_list = [], [], []
torque_x_list, torque_y_list, torque_z_list = [], [], []

try:
    # Create driver instance
    bota_ft_sensor_driver = bota_driver.BotaDriver(config_path)

    # Get driver version information
    print(f" >>>>>>>>>>> BotaDriver version: {bota_ft_sensor_driver.get_driver_version_string()} <<<<<<<<<<<< ")

    # Transition driver from UNCONFIGURED to INACTIVE state
    if not bota_ft_sensor_driver.configure():
        raise RuntimeError("Failed to configure driver")

    # Uncomment to tare the sensor
    if not bota_ft_sensor_driver.tare():
        raise RuntimeError("Failed to tare sensor")

    # Transition driver from INACTIVE to ACTIVE state
    if not bota_ft_sensor_driver.activate():
        raise RuntimeError("Failed to activate driver")

    ########################
    ## CONTROL LOOP START ##
    ########################

    # Define the example duration
    EXAMPLE_DURATION = 10.0  # seconds

    # Define the reading frequency
    READING_FREQUENCY = 10000  # Hz

    # Define the console status-print frequency (lower than READING_FREQUENCY)
    PRINTING_FREQUENCY = 1000  # Hz

    start_time = time.perf_counter()  # High-resolution start time
    last_print_time = start_time  # Track when we last printed
    next_execution_time = start_time
    rows_written = 0
    last_timestamp = None  # track the sensor's own timestamp to detect new frames

    print(f"Logging data to: {csv_path}")

    with open(csv_path, mode = "w", newline = "") as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(CSV_HEADERS)

        while time.perf_counter() - start_time < EXAMPLE_DURATION and not stop_flag:
            # Read frame
            bota_frame = bota_ft_sensor_driver.read_frame()

            # Extract the data from the bota_frame
            status = bota_frame.status
            force = bota_frame.force
            torque = bota_frame.torque
            timestamp = bota_frame.timestamp

            # Only log if this is a genuinely new sample from the sensor
            if timestamp != last_timestamp:
                csv_writer.writerow([
                    timestamp,
                    force[0], force[1], force[2],
                    torque[0], torque[1], torque[2],
                ])
                rows_written += 1
                last_timestamp = timestamp

                # Buffer the same values for plotting after the run
                timestamps_list.append(timestamp)
                force_x_list.append(force[0])
                force_y_list.append(force[1])
                force_z_list.append(force[2])
                torque_x_list.append(torque[0])
                torque_y_list.append(torque[1])
                torque_z_list.append(torque[2])

            # Lightweight progress feedback at the throttled rate (no full data dump)
            current_time = time.perf_counter()
            if current_time - last_print_time >= 1.0 / PRINTING_FREQUENCY:
                print(f"{rows_written} frames logged... (t={current_time - start_time:.2f}s)")
                last_print_time = current_time

            #################################
            ## YOUR CONTROL LOOP CODE HERE ##
            #################################

            # Wait until next execution time
            next_execution_time += 1.0 / READING_FREQUENCY
            sleep_time = max(0, next_execution_time - time.perf_counter())
            time.sleep(sleep_time)

    # Transition driver from ACTIVE to INACTIVE state
    if not bota_ft_sensor_driver.deactivate():
        raise RuntimeError("Failed to deactivate driver")

    # Shutdown the driver
    if not bota_ft_sensor_driver.shutdown():
        raise RuntimeError("Failed to shutdown driver")

    print(f"Completition WITHOUT errors. {rows_written} frames written to {csv_path}")

except Exception as e:
    print(f"FATAL: {e}")
    print("Completition WITH errors.")

finally:
    # Plotting the data after the run
    if len(timestamps_list) > 1:
        fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

        axes[0].plot(timestamps_list, force_x_list, label="Force X")
        axes[0].plot(timestamps_list, force_y_list, label="Force Y")
        axes[0].plot(timestamps_list, force_z_list, label="Force Z")
        axes[0].set_ylabel("Force (N)")
        axes[0].set_title("Force over Time")
        axes[0].legend()
        axes[0].grid(True)

        axes[1].plot(timestamps_list, torque_x_list, label="Torque X")
        axes[1].plot(timestamps_list, torque_y_list, label="Torque Y")
        axes[1].plot(timestamps_list, torque_z_list, label="Torque Z")
        axes[1].set_xlabel("Timestamp")
        axes[1].set_ylabel("Torque (Nm)")
        axes[1].set_title("Torque over Time")
        axes[1].legend()
        axes[1].grid(True)

        plt.tight_layout()
        plt.savefig(plot_path)
        print(f"Plot saved to: {plot_path}")

        plt.show()
    else:
        print("Not enough data collected to plot.")

    print("EXITING PROGRAM")
    sys.exit(0)