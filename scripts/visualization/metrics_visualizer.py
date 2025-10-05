# metrics_visualizer.py
import pandas as pd
import matplotlib.pyplot as plt
from utils.config_logging import logger


class MetricsVisualizer:
    """
    A class to load and visualize YOLO training metrics
    from a CSV file generated during training.
    """


    def __init__(self, yolo_csv: str):
        """
        Initialize the visualizer with the path to the YOLO metrics CSV.
        """
        self.yolo_csv = yolo_csv
        self.yolo_df = None


    def load_yolo_metrics(self):
        """
        Load YOLO metrics from a CSV file into a pandas DataFrame.
        """
        try:
            if self.yolo_csv:
                self.yolo_df = pd.read_csv(self.yolo_csv)
                logger.info("YOLO metrics successfully loaded.")
            else:
                logger.warning("No YOLO path provided.")

        except Exception as e:
            logger.exception(f"Failed to load YOLO metrics: {e}")


    def plot_yolo_metrics(self):
        """
        Plot YOLO training and validation metrics, including:
        - Training losses
        - Training vs validation losses
        - Validation metrics (Precision, Recall, mAP)
        - Learning rate evolution
        """
        self.load_yolo_metrics()
        logger.info(f"CSV Columns: {self.yolo_df.columns.tolist()}")

        # --- Plot 1: Training losses ---
        plt.figure(figsize=(10, 5))
        plt.plot(self.yolo_df["epoch"], self.yolo_df["train/box_loss"], label="Train Box Loss", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["train/cls_loss"], label="Train Class Loss", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["train/dfl_loss"], label="Train DFL Loss", marker="o")
        plt.title("YOLO11n - Training Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 2: Training vs Validation losses ---
        plt.figure(figsize=(10, 5))
        plt.plot(self.yolo_df["epoch"], self.yolo_df["train/box_loss"], label="Train Box Loss", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["train/cls_loss"], label="Train Class Loss", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["train/dfl_loss"], label="Train DFL Loss", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["val/box_loss"], label="Val Box Loss", marker="x")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["val/cls_loss"], label="Val Class Loss", marker="x")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["val/dfl_loss"], label="Val DFL Loss", marker="x")
        plt.title("YOLO11n - Training vs Validation Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 3: Validation metrics ---
        plt.figure(figsize=(10, 5))
        plt.plot(self.yolo_df["epoch"], self.yolo_df["metrics/precision(B)"], label="Precision", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["metrics/recall(B)"], label="Recall", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["metrics/mAP50(B)"], label="mAP@50", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["metrics/mAP50-95(B)"], label="mAP@50-95", marker="o")
        plt.title("YOLO11n - Validation Metrics")
        plt.xlabel("Epoch")
        plt.ylabel("Score")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- Plot 4: Learning rate evolution per parameter group ---
        plt.figure(figsize=(10, 5))
        plt.plot(self.yolo_df["epoch"], self.yolo_df["lr/pg0"], label="LR pg0", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["lr/pg1"], label="LR pg1", marker="o")
        plt.plot(self.yolo_df["epoch"], self.yolo_df["lr/pg2"], label="LR pg2", marker="o")
        plt.title("YOLO11n - Learning Rate Evolution (per Group)")
        plt.xlabel("Epoch")
        plt.ylabel("Learning Rate")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

        # --- logger.info last epoch summary ---
        last_epoch_summary = self.yolo_df.tail(1).T
        logger.info("Last epoch summary \n" + last_epoch_summary.to_string(header=False))
