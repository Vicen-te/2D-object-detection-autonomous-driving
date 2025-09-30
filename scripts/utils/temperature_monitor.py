# temperature_monitor.py
import time
import multiprocessing

from utils.config_logging import setup_logging
logger = setup_logging()

# For temperatures (Windows & Nvidia)
import pynvml
import wmi


class TemperatureExceededException(Exception):
    def __init__(self, cpu_temp, gpu_temp):
        message = f"Temperature threshold exceeded! CPU: {cpu_temp}°C, GPU: {gpu_temp}°C"
        super().__init__(message)
        self.cpu_temp = cpu_temp
        self.gpu_temp = gpu_temp


class TemperatureMonitor:

    def __init__(
        self,
        train_func,
        train_args=(),
        train_kwargs=None,
        gpu_temp_threshold=67,
        cpu_temp_threshold=95,
        monitor_interval=30,
        max_consecutive_warnings=3
    ):
        """
        Initializes the temperature monitor with parameters and the training function.

        Args:
            train_func (callable): Function that runs the training.
            train_args (tuple): Positional arguments for train_func.
            train_kwargs (dict): Keyword arguments for train_func.
            gpu_temp_threshold (int): GPU temperature threshold in °C.
            cpu_temp_threshold (int): CPU temperature threshold in °C.
            monitor_interval (int): Monitoring interval in seconds.
            max_consecutive_warnings (int): Maximum warnings before terminating the process.
        """
        self.train_func = train_func
        self.train_args = train_args
        self.train_kwargs = train_kwargs or {}
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
                if sensor.SensorType == 'Temperature' and 'CPU Package' in sensor.Name:
                    return float(sensor.Value)
            return None
        except Exception as e:
            logger.exception(f"Error retrieving CPU temperature: {e}")
            return None


    def start(self):
        """
        Starts the training process and temperature monitoring.
        """
        training_process = multiprocessing.Process(
            target=self.train_func,
            args=self.train_args,
            kwargs=self.train_kwargs
        )
        training_process.start()

        consecutive_warnings = 0

        try:
            while training_process.is_alive():
                cpu_temp = self.get_cpu_temp_lhm()
                gpu_temp = self.get_gpu_temp_nvidia()

                overheat = (
                    (cpu_temp is not None and cpu_temp > self.cpu_temp_threshold) or
                    (gpu_temp is not None and gpu_temp > self.gpu_temp_threshold)
                )

                logger.info(f"CPU Temp: {cpu_temp}°C, GPU Temp: {gpu_temp}°C", end="" if overheat else "\n")

                if overheat:
                    consecutive_warnings += 1
                    logger.warning(f"{consecutive_warnings}/{self.max_consecutive_warnings}")
                else:
                    consecutive_warnings = 0

                if consecutive_warnings >= self.max_consecutive_warnings:
                    training_process.terminate()
                    raise TemperatureExceededException(cpu_temp, gpu_temp)

                time.sleep(self.monitor_interval)

        except TemperatureExceededException as e:
            logger.exception(e)
        finally:
            training_process.join()
