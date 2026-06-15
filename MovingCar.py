import airsim
import time
import threading

STATIC_CARS = [
    "Parked_01", "Parked_01B",
    "Parked_02", "Parked_02B",
    "Parked_03", "Parked_03B",
    "Parked_04", "Parked_04B",
    "Parked_05", "Parked_05B",
    "Parked_06", "Parked_06B",
    "Parked_07", "Parked_07B",
    "Parked_08", "Parked_08B",
    "Parked_09", "Parked_09B",
    "Parked_10", "Parked_10B",
    "Parked_11", "Parked_11B",
    "Parked_12", "Parked_12B",
    "Parked_13", "Parked_13B",
    "Parked_14", "Parked_14B",
    "Parked_15", "Parked_15B",
    "Parked_16", "Parked_16B",
]

def keep_parked(client):
    """Continuously reapply brakes to all parked cars."""
    while True:
        for name in STATIC_CARS:
            try:
                parked          = airsim.CarControls()
                parked.throttle = 0.0
                parked.brake    = 1.0
                parked.steering = 0.0
                client.setCarControls(parked, vehicle_name=name)
            except:
                pass
        time.sleep(0.1)


def main():
    client = airsim.CarClient()
    client.confirmConnection()
    print("[CONNECTED] AirSim connected.")

    # ── Park all static cars ───────────────────────────────────────────────────
    for name in STATIC_CARS:
        try:
            client.enableApiControl(True, vehicle_name=name)
            init_pose = client.simGetVehiclePose(vehicle_name=name)
            client.simSetVehiclePose(init_pose, ignore_collision=False, vehicle_name=name)
            parked          = airsim.CarControls()
            parked.throttle = 0.0
            parked.brake    = 1.0
            parked.steering = 0.0
            client.setCarControls(parked, vehicle_name=name)
            print(f"[PARKED] {name}")
        except Exception as e:
            print(f"[SKIP]   {name} — {e}")

    # ── Background thread to keep them braked ─────────────────────────────────
    threading.Thread(target=keep_parked, args=(client,), daemon=True).start()
    print("[BRAKE THREAD] Running — all parked cars held stationary.")

    # ── Release Car1 to manual keyboard control ────────────────────────────────
    client.enableApiControl(False, vehicle_name="Car1")
    print("\n[MANUAL] Car1 released to keyboard control.")
    print("         W / S     → throttle / brake")
    print("         A / D     → steer left / right")
    print("         Space     → handbrake")
    print("         F1        → toggle mouse capture")
    print("\n[READY]  Drive Car1. Ctrl+C to stop.\n")

    try:
        while True:
            time.sleep(1.0)
    except KeyboardInterrupt:
        print("\n[STOPPED] Script terminated.")


if __name__ == "__main__":
    main()