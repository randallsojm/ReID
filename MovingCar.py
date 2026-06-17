import airsim
import time
import threading
import random
import math

# ── All cars ──────────────────────────────────────────────────────────────────
ALL_PARKED = [
    "Parked_01",  "Parked_01B",
    "Parked_02",  "Parked_02B",
    "Parked_03",  "Parked_03B",
    "Parked_04",  "Parked_04B",
    "Parked_05",  "Parked_05B",
    "Parked_06",  "Parked_06B",
    "Parked_07",  "Parked_07B",
    "Parked_08",  "Parked_08B",
    "Parked_09",  "Parked_09B",
    "Parked_10",  "Parked_10B",
    "Parked_11",  "Parked_11B",
    "Parked_12",  "Parked_12B",
    "Parked_13",  "Parked_13B",
    "Parked_14",  "Parked_14B",
    "Parked_15",  "Parked_15B",
    "Parked_16",  "Parked_16B",
]

MOVING_CARS = ALL_PARKED

# ── Speed cap ─────────────────────────────────────────────────────────────────
SPEED_MIN_KMH = 20.0
SPEED_MAX_KMH = 60.0

# ── Movement behaviours ───────────────────────────────────────────────────────
BEHAVIOURS = [
    ( 0.6,  0.0,  0.0,  False, "straight forward"),
    ( 0.6,  0.0,  0.0,  False, "straight forward"),
    ( 0.4,  0.0,  0.5,  False, "forward turn right"),
    ( 0.4,  0.0, -0.5,  False, "forward turn left"),
    ( 0.6,  0.0,  0.0,  True,  "reverse straight"),
    ( 0.6,  0.0,  0.4,  True,  "reverse turn right"),
    ( 0.6,  0.0, -0.4,  True,  "reverse turn left"),
]

BEHAVIOUR_MIN_S = 3.0
BEHAVIOUR_MAX_S = 8.0

# ── Thread safety ─────────────────────────────────────────────────────────────
# AirSim buffer cannot handle 32 threads calling it simultaneously.
# All API calls are serialised through this lock.
_lock = threading.Lock()


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_speed_ms(client, name):
    """Get current speed in m/s. Returns 0.0 on error."""
    try:
        with _lock:
            state = client.getCarState(vehicle_name=name)
        v = state.kinematics_estimated.linear_velocity
        return math.sqrt(v.x_val**2 + v.y_val**2 + v.z_val**2)
    except:
        return 0.0


def apply_controls(client, name, controls):
    """Apply car controls with thread lock."""
    try:
        with _lock:
            client.setCarControls(controls, vehicle_name=name)
    except Exception as e:
        print(f"[{name}] error: {e}")


# ── Moving car controller ─────────────────────────────────────────────────────

def drive_car(client, name, stop_event):
    """
    Original throttle-based behaviours with speed capping.
    Polls speed every 0.5s (not 0.1s) to reduce API call pressure.
    """
    while not stop_event.is_set():
        throttle, brake, steering, is_reverse, label = random.choice(BEHAVIOURS)
        duration   = random.uniform(BEHAVIOUR_MIN_S, BEHAVIOUR_MAX_S)
        target_kmh = random.uniform(SPEED_MIN_KMH, SPEED_MAX_KMH)
        target_ms  = target_kmh / 3.6

        print(f"[{name:14s}] {label:22s}  "
              f"target={target_kmh:.0f}km/h  "
              f"steer={steering:+.1f}  "
              f"{'REV' if is_reverse else 'FWD'}  "
              f"for {duration:.1f}s")

        deadline = time.time() + duration
        while time.time() < deadline and not stop_event.is_set():

            current_ms = get_speed_ms(client, name)

            # cut throttle if over target, apply normally if under
            applied_throttle = 0.0 if current_ms >= target_ms else throttle

            controls                = airsim.CarControls()
            controls.throttle       = applied_throttle
            controls.brake          = brake
            controls.steering       = steering
            controls.is_manual_gear = is_reverse
            controls.manual_gear    = -1 if is_reverse else 1

            apply_controls(client, name, controls)

            # poll every 0.5s — reduces simultaneous API calls from 32×10=320/s
            # down to 32×2=64/s which AirSim can handle
            time.sleep(0.5)

    # stop cleanly
    stop            = airsim.CarControls()
    stop.throttle   = 0.0
    stop.brake      = 1.0
    stop.steering   = 0.0
    apply_controls(client, name, stop)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    client = airsim.CarClient()
    client.confirmConnection()
    print("[CONNECTED] AirSim connected.")
    print(f"[SPEED]    {SPEED_MIN_KMH:.0f}–{SPEED_MAX_KMH:.0f} km/h")
    print(f"[CARS]     {len(MOVING_CARS)} cars\n")

    stop_event = threading.Event()

    # initialise all cars
    for name in ALL_PARKED:
        try:
            with _lock:
                client.enableApiControl(True, vehicle_name=name)
            print(f"[INIT]   {name}")
        except Exception as e:
            print(f"[SKIP]   {name} — {e}")

    # launch one thread per car with staggered starts
    drive_threads = []
    for name in MOVING_CARS:
        time.sleep(random.uniform(0.1, 0.5))
        t = threading.Thread(
            target=drive_car, args=(client, name, stop_event), daemon=True
        )
        t.start()
        drive_threads.append(t)
    print(f"\n[DRIVING] {len(drive_threads)} car threads running.\n")

    # release Car1 to keyboard
    with _lock:
        client.enableApiControl(False, vehicle_name="Car1")
    print("[MANUAL] Car1 released to keyboard control.")
    print("         W / S  → throttle / brake")
    print("         A / D  → steer left / right")
    print("\n[READY]  Drive Car1. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[STOPPING] Braking all cars...")
        stop_event.set()
        time.sleep(2.0)

        for name in ALL_PARKED:
            stop            = airsim.CarControls()
            stop.throttle   = 0.0
            stop.brake      = 1.0
            stop.steering   = 0.0
            apply_controls(client, name, stop)

        print("[STOPPED] All cars braked.")


if __name__ == "__main__":
    main()