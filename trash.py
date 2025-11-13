# ------------------------------------------------------------
# test_paired_devices.py
# ------------------------------------------------------------
import subprocess
import sys

def get_paired_devices():
    """Call bluetoothctl paired-devices and return a list of dicts."""
    try:
        result = subprocess.run(
            ["bluetoothctl", "paired-devices"],
            capture_output=True,
            text=True,
            timeout=10
        )
        print("=== RAW bluetoothctl OUTPUT ===")
        print(result.stdout)
        print("=== STDERR ===")
        print(result.stderr)
        print("=== RETURN CODE ===", result.returncode)
        print()

        devices = []
        if result.returncode == 0:
            for line in result.stdout.splitlines():
                line = line.strip()
                if line.startswith("Device "):
                    parts = line.split(" ", 2)      # ["Device", "MAC", "name..."]
                    mac = parts[1]
                    name = parts[2] if len(parts) > 2 else "Unknown"
                    devices.append({"mac": mac, "name": name})
        else:
            print("bluetoothctl failed!")
        return devices
    except Exception as e:
        print("Exception:", e)
        return []


if __name__ == "__main__":
    devs = get_paired_devices()
    print("\n=== PARSED DEVICES ===")
    for d in devs:
        print(f"{d['name']}  ->  {d['mac']}")
    if not devs:
        print("No paired devices found.")