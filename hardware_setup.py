import argparse
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
SKETCH_DIR = ROOT / "ino files" / "energenius_bh1750"
SECRETS_FILE = SKETCH_DIR / "secrets.h"
SECRETS_EXAMPLE = SKETCH_DIR / "secrets.example.h"
ESP32_INDEX = "https://espressif.github.io/arduino-esp32/package_esp32_index.json"
FQBN = "esp32:esp32:esp32"
PLACEHOLDERS = (
    "YOUR_WIFI_SSID",
    "YOUR_WIFI_PASSWORD",
    "YOUR_THINGSPEAK_WRITE_KEY",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Prepare, compile, and optionally upload the EnerGenius ESP32 firmware."
    )
    parser.add_argument(
        "--prepare",
        action="store_true",
        help="Install missing ESP32 components/libraries and compile the firmware.",
    )
    parser.add_argument(
        "--upload",
        action="store_true",
        help="Compile and upload. Requires configured secrets and --port.",
    )
    parser.add_argument("--port", help="ESP32 serial port, such as /dev/cu.usbserial-0001 or COM4.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    ensure_local_secrets_file()

    cli = shutil.which("arduino-cli")
    secrets_ready = secrets_are_configured()
    print(f"Project: {ROOT}", flush=True)
    print(f"Arduino CLI: {cli or 'NOT FOUND'}", flush=True)
    print(
        f"Credentials: {'configured' if secrets_ready else 'PLACEHOLDERS - teammate must update secrets.h'}",
        flush=True,
    )

    if not cli:
        print("\nInstall Arduino CLI, then rerun this command:")
        print("  macOS:   brew install arduino-cli")
        print("  Windows/Linux: https://arduino.github.io/arduino-cli/latest/installation/")
        print("\nArduino IDE can also flash ino files/energenius_bh1750/energenius_bh1750.ino.")
        return 2

    print("\nConnected serial devices:", flush=True)
    run([cli, "board", "list"], check=False)

    if not args.prepare and not args.upload:
        print("\nPreflight only. Run with --prepare to install dependencies and compile.")
        return 0

    try:
        prepare_toolchain(cli)
        compile_firmware(cli)
        if args.upload:
            if not secrets_ready:
                print("\nUpload blocked: replace all placeholders in secrets.h first.")
                return 2
            if not args.port:
                print("\nUpload blocked: pass the ESP32 port using --port.")
                return 2
            run([cli, "upload", "--fqbn", FQBN, "--port", args.port, str(SKETCH_DIR)])
            print("\nFirmware uploaded successfully.")
    except subprocess.CalledProcessError as exc:
        print(f"\nHardware preparation failed with exit code {exc.returncode}.")
        return exc.returncode or 1

    print("\nFirmware compilation succeeded.")
    if not args.upload:
        print("After connecting the ESP32, rerun with --upload --port <PORT>.")
    return 0


def ensure_local_secrets_file() -> None:
    if not SECRETS_FILE.exists():
        shutil.copy2(SECRETS_EXAMPLE, SECRETS_FILE)


def secrets_are_configured() -> bool:
    contents = SECRETS_FILE.read_text(encoding="utf-8")
    return not any(placeholder in contents for placeholder in PLACEHOLDERS)


def prepare_toolchain(cli: str) -> None:
    config = capture([cli, "config", "dump"], check=False)
    if ESP32_INDEX not in config:
        run([cli, "config", "add", "board_manager.additional_urls", ESP32_INDEX])

    run([cli, "core", "update-index"])
    installed_cores = capture([cli, "core", "list"], check=False)
    if "esp32:esp32" not in installed_cores:
        run([cli, "core", "install", "esp32:esp32"])

    installed_libraries = capture([cli, "lib", "list"], check=False).lower()
    for library in ("DHT sensor library", "BH1750"):
        if library.lower() not in installed_libraries:
            run([cli, "lib", "install", library])


def compile_firmware(cli: str) -> None:
    run([cli, "compile", "--fqbn", FQBN, str(SKETCH_DIR)])


def run(command: list[str], check: bool = True) -> subprocess.CompletedProcess:
    print(f"\n$ {' '.join(command)}", flush=True)
    return subprocess.run(command, cwd=ROOT, check=check)


def capture(command: list[str], check: bool = True) -> str:
    result = subprocess.run(
        command,
        cwd=ROOT,
        check=check,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return result.stdout


if __name__ == "__main__":
    sys.exit(main())
