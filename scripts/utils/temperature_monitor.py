# temperature_monitor.py
import time
import multiprocessing

from utils.config_logging import logger

# For temperatures (Windows & Nvidia)
import pynvml
import wmi
import sys
import clr
from pathlib import Path

class TemperatureExceededException(Exception):
    def __init__(self, cpu_temp, gpu_temp):
        message = f"Temperature threshold exceeded! CPU: {cpu_temp}°C, GPU: {gpu_temp}°C"
        super().__init__(message)
        self.cpu_temp = cpu_temp
        self.gpu_temp = gpu_temp


class TemperatureMonitor:

    def __init__(
        self,
        func,
        args=None,
        kwargs=None,
        gpu_temp_threshold=67,
        cpu_temp_threshold=95,
        monitor_interval=30,
        max_consecutive_warnings=3
    ):
        """
        Initializes the temperature monitor with parameters and the function.

        Args:
            func (callable): Function that runs.
            args (tuple): Positional arguments.
            kwargs (dict): Keyword arguments for func.
            gpu_temp_threshold (int): GPU temperature threshold in °C.
            cpu_temp_threshold (int): CPU temperature threshold in °C.
            monitor_interval (int): Monitoring interval in seconds.
            max_consecutive_warnings (int): Maximum warnings before terminating the process.
        """
        self.func = func
        self.args = args or {}
        self.kwargs = kwargs or {}
        self.gpu_temp_threshold = gpu_temp_threshold
        self.cpu_temp_threshold = cpu_temp_threshold
        self.monitor_interval = monitor_interval
        self.max_consecutive_warnings = max_consecutive_warnings


    @staticmethod
    def get_gpu_temp_nvidia():
        try:
            pynvml.nvmlInit()
            handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            temp = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
            pynvml.nvmlShutdown()
            return temp
        except Exception:
            return None


    @staticmethod
    def get_cpu_temp_lhm():
        try:
            w = wmi.WMI(namespace="root\\LibreHardwareMonitor")
            temperature_infos = w.Sensor()
            for sensor in temperature_infos:
                # Some installations return integer enums for SensorType
                sensor_type = str(sensor.SensorType).lower()
                sensor_name = str(sensor.Name).lower()
                if sensor_type == 'temperature' and 'cpu package' in sensor_name:
                    return float(sensor.Value)
            return None
        except Exception as e:
            logger.exception(f"Error retrieving CPU temperature: {e}")
            return FileNotFoundError


    def start(self):
        """
        Starts the training process and monitors CPU/GPU temperature.
        """
        process = multiprocessing.Process(
            target=self.func,
            args=self.args,
            kwargs=self.kwargs
        )
        process.start()

        consecutive_warnings = 0

        try:
            while process.is_alive():
                cpu_temp = self.get_cpu_temp_ohm()
                gpu_temp = self.get_gpu_temp_nvidia()

                overheat = (
                    (cpu_temp is not None and cpu_temp > self.cpu_temp_threshold) or
                    (gpu_temp is not None and gpu_temp > self.gpu_temp_threshold)
                )

                # print(f"CPU Temp: {cpu_temp}°C, GPU Temp: {gpu_temp}°C", end="" if overheat else "\n")
                logger.info(f"CPU Temp: {cpu_temp}°C, GPU Temp: {gpu_temp}°C")

                if overheat:
                    consecutive_warnings += 1
                    logger.warning(f"{consecutive_warnings}/{self.max_consecutive_warnings}")
                else:
                    consecutive_warnings = 0

                if consecutive_warnings >= self.max_consecutive_warnings:
                    logger.error("Maximum temperature reached. Terminating training process.")
                    process.terminate()
                    raise TemperatureExceededException(cpu_temp, gpu_temp)

                time.sleep(self.monitor_interval)

        except KeyboardInterrupt:
            # Handle Ctrl+C gracefully
            logger.warning("Ctrl+C detected. Terminating training process...")
            if process.is_alive(): 
                process.terminate()
            # Ensure main process exits cleanly
            sys.exit(0)

        except TemperatureExceededException as e:
            # Log the temperature exception
            logger.exception(e)

        finally:
            # Wait for the training process to finish
            if process.is_alive():
                process.join()
            logger.info("Training process finished successfully.")
