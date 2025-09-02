import asyncio
import os
import time
import sys
import signal
from contextlib import asynccontextmanager
from smbus2 import SMBus
import requests
import logging

from fastapi import FastAPI

I2C_ADDRESS = 0x1A

# Build config from env vars
config = {
    "min_temp": float(os.environ.get("MIN_TEMP", "40")),
    "max_temp": float(os.environ.get("MAX_TEMP", "60")),
    "temp_unit": (
        os.environ.get("TEMP_UNIT")
        if os.environ.get("TEMP_UNIT") in ["C", "F"]
        else "C"
    ),
    "create_entity": os.environ.get("CREATE_ENTITY", "false") == "true",
    "log_temp": os.environ.get("LOG_TEMP", "false") == "true",
    "manual_mode": os.environ.get("MANUAL_MODE", "false") == "true",
    "homeassistant_url": os.environ.get(
        "HOMEASSISTANT_URL", "http://hassio/homeassistant"
    ),
    "supervisor_token": os.environ.get("SUPERVISOR_TOKEN", ""),
    "error_level": os.environ.get("ERROR_LEVEL", "info"),
}

# Setup logging
logging.basicConfig(level=config["error_level"].upper())
logger = logging.getLogger(__name__)

# Set some global variables, for state
port = None
last_status = {
    "currentTemp": None,
    "currentTempUnit": config["temp_unit"],
    "fanPercent": None,
}
command_queue = asyncio.Queue()


def safe_shutdown(signum=None, frame=None):
    logger.error(f"Failed {sys._getframe().f_lineno}: Last command failed")
    with SMBus(port) as bus:
        bus.write_byte(I2C_ADDRESS, 100)
    logger.info("Safe Mode Activated!")


# Handle signals
for sig in (signal.SIGINT, signal.SIGTERM):
    signal.signal(sig, safe_shutdown)


def find_i2c_port():
    i2c_devices = [dev for dev in os.listdir("/dev") if dev.startswith("i2c-")]

    for device in i2c_devices:
        port = int(device.split("-")[1])
        logger.info(f"checking i2c port {port} at /dev/{device}")
        with SMBus(port) as bus:
            # Check address 1A
            try:
                bus.read_byte(I2C_ADDRESS, 0x00)
                logger.info(f"Found Argon One on I2C port {port}")
                return port
            except Exception as e:
                logger.error(
                    f"Failed to communicate with Argon One on I2C port {port}: {e}"
                )

    return None


def fan_speed_report_linear(fan_percent: int, cpu_temp: int):
    url = f"{config['homeassistant_url']}/api/states/sensor.argon_one_fan_speed"
    headers = {
        "Authorization": f"Bearer {config['supervisor_token']}",
        "Content-Type": "application/json",
        "Connection": "close",
    }
    icon = "mdi:fan"
    req_body = {
        "state": str(fan_percent),
        "attributes": {
            "unit_of_measurement": "%",
            "icon": icon,
            f"Temperature {config['temp_unit']}": str(cpu_temp),
            "friendly_name": "Argon Fan Speed",
        },
    }
    try:
        requests.post(url, headers=headers, json=req_body, timeout=5)
    except Exception as e:
        logger.error(f"Failed to report fan speed: {e}")


def set_device_fan_speed(bus: SMBus, fan_percent: int):
    try:
        bus.write_byte(I2C_ADDRESS, fan_percent)
        return True
    except Exception as e:
        logger.error(f"Failed to set fan speed: {e}")
        return False


def set_fan_percent(bus: SMBus, fan_percent: int, cpu_temp: int):
    if set_device_fan_speed(bus, fan_percent):
        last_status["fanPercent"] = fan_percent
        logger.info(
            (
                f"{time.strftime('%Y-%m-%d_%H:%M:%S')}: {cpu_temp}{config['temp_unit']} - "
                f"Fan {fan_percent}% | hex:({hex(fan_percent)})"
            )
        )
        if config["create_entity"]:
            fan_speed_report_linear(fan_percent, cpu_temp)

    else:
        fan_percent = last_status["fanPercent"]


def get_temp_in_unit() -> int:
    with open("/sys/class/thermal/thermal_zone0/temp") as temp_file:
        cpu_raw_temp = int(temp_file.read().strip())
    cpu_temp = cpu_raw_temp // 1000
    if config["temp_unit"] == "F":
        cpu_temp = int(cpu_temp * 9 / 5 + 32)
    return cpu_temp


async def worker():
    logger.info('Detecting Layout of i2c, we expect to see "1a" here.')
    port = find_i2c_port()

    if port is None:
        logger.error(
            (
                "Argon One was not detected on i2c. "
                "Argon One will show a 1a on the i2c bus using the `i2cdetect` command. "
                "This add-on will not control temperature without a connection to Argon One."
            )
        )
        sys.exit(1)

    logger.info(f"Argon One Detected at I2C Port {port}. Beginning monitor...")
    logger.debug(f"Config: {config}")

    value_a = 100 / (config["max_temp"] - config["min_temp"])
    value_b = -value_a * config["min_temp"]

    try:
        with SMBus(port) as bus:
            prev_tick = time.monotonic()
            while True:
                if config["manual_mode"] and not command_queue.empty():
                    fan_percent = await command_queue.get()
                    logger.info(f"Setting manual fan speed to {fan_percent}%")
                    set_fan_percent(bus, fan_percent, last_status["currentTemp"])

                # every 30 seconds
                if time.monotonic() - prev_tick >= 30:
                    cpu_temp = get_temp_in_unit()

                    if config["log_temp"]:
                        logger.info(
                            f"Current Temperature = {cpu_temp} °{config['temp_unit']}"
                        )

                    # set the current status
                    last_status["currentTemp"] = cpu_temp

                    # if not manual mode, set fan speed based on temperature
                    if not config["manual_mode"]:
                        fan_percent = max(
                            0, min(100, int(value_a * cpu_temp + value_b))
                        )
                        if last_status["fanPercent"] != fan_percent:
                            fan_percent = set_fan_percent(bus, fan_percent, cpu_temp)

                    if config["create_entity"]:
                        fan_speed_report_linear(fan_percent, cpu_temp)

                    prev_tick = time.monotonic()

                await asyncio.sleep(1)
        pass

    except Exception as e:
        logger.error(f"Exception in main loop: {e}")
        safe_shutdown()
        # raise

    finally:
        safe_shutdown()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the ML model
    logger.info("Starting up...")
    task = asyncio.create_task(worker())
    logger.info("Worker started.")
    yield
    task.cancel()


app = FastAPI(lifespan=lifespan)


@app.get("/")
def read_root():
    return last_status


@app.put("/fan/{speed}")
async def set_fan_speed(speed: int):
    if config["manual_mode"]:
        if 0 <= speed <= 100:
            await command_queue.put(speed)
            return {"status": "success"}

        return {
            "status": "error",
            "message": "Invalid fan speed. Must be between 0 and 100.",
        }

    return {
        "status": "error",
        "message": "Setting fan speed only works in manual mode.",
    }
