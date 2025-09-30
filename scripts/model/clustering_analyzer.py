# clustering_analyzer.py
import numpy as np
from pathlib import Path
import cv2
from ultralytics import YOLO
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from mpl_toolkits.mplot3d import Axes3D
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
from utils.types_aliases import FeatureData

from scripts.utils.config_logging import setup_logging
logger = setup_logging()


class ClusteringAnalyzer:
    """
    Analyzes model feature space characteristics by clustering 
    the feature vectors extracted from a YOLO model.
    """


    def __init__(self, n_clusters: int = 7, random_state: int = 42):
        """
        Initializes the analyzer with clustering parameters.
        
        Args:
            n_clusters: Number of clusters (K) for K-Means.
            random_state: Seed for reproducibility in clustering and t-SNE.
        """
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.kmeans_model: KMeans | None = None
        self.feature_vectors: np.ndarray = np.array([])
        self.image_file_ids: List[str] = []
        self.cluster_labels: np.ndarray = np.array([])


    def extract_features_from_images(self, model: YOLO, dataset_path: str, layer_index: int = 10) -> FeatureData:
        """
        Processes images, performs YOLO inference, and extracts feature vectors 
        from a specified layer for each detection.

        Args:
            model: The loaded YOLO model (pre-trained or custom).
            dataset_path: Path to the folder containing the images.
            layer_index: The index of the layer to extract embeddings from (e.g., 10 for the backbone output).

        Returns:
            Tuple[np.ndarray, List[str]]: A numpy array of feature vectors and a list of image file names.
        """
        feature_vectors: List[np.ndarray] = []
        image_ids: List[str] = []
        image_paths: List[Path] = [
            p for p in Path(dataset_path).iterdir() 
            if p.suffix.lower() in ('.png', '.jpg', '.jpeg')
        ]
        
        logger.info(f"Found {len(image_paths)} images. Starting feature extraction...")

        for filepath in tqdm(
            image_paths, 
            desc='Extracting features', 
            unit='image', 
            ncols=100
        ):
            try:
                # Use YOLO's built-in predict method with 'embed' argument
                # This returns a list of results objects, typically one per image.
                # NOTE: The provided original code assumes a flat vector is returned
                # by embeddings[0].cpu().numpy().flatten(). This works if the model
                # is modified to only output a single feature vector per image or 
                # if YOLO's structure is altered. We adapt to the common use case 
                # where the embedding is extracted from the Results object.
                embeddings = model.predict(
                    str(filepath), 
                    embed=[layer_index], # Extract embeddings from the specified layer
                    verbose=False
                ) 
                
                if embeddings and len(embeddings) > 0:
                    # Assuming we take the full embedding vector related to the image
                    # The exact way to retrieve the single feature vector might depend
                    # on the ultralytics version and custom model configuration.
                    # This line is a common adaptation for getting the overall feature.
                    vec = embeddings[0].cpu().numpy().flatten()
                    
                    feature_vectors.append(vec)
                    image_ids.append(filepath.name)
                else:
                    logger.warning(f"No valid detections or embeddings for {filepath.name}")
                    pass
            except Exception as e:
                logger.exception(f"Error processing image {filepath.name}: {e}")
                
        self.feature_vectors = np.array(feature_vectors)
        self.image_file_ids = image_ids
        logger.info(f"Extracted {len(self.feature_vectors)} feature vectors.")
        
        return self.feature_vectors, self.image_file_ids


    def perform_kmeans_clustering(self) -> np.ndarray:
        """
        Performs K-Means clustering on the extracted feature vectors.
        
        Returns:
            np.ndarray: The cluster labels for each data point.
        """
        if self.feature_vectors.size == 0:
            raise ValueError("No feature vectors available. Run extract_features_from_images first.")
            
        logger.info(f"Performing K-Means clustering with K={self.n_clusters}...")
        self.kmeans_model = KMeans(
            n_clusters=self.n_clusters, 
            random_state=self.random_state, 
            n_init='auto'
        )
        self.cluster_labels = self.kmeans_model.fit_predict(self.feature_vectors)
        logger.info("Clustering completed.")
        return self.cluster_labels


    def visualize_tsne(self, n_components: int = 2, title_suffix: str = "", dim_labels: List[str] | None = None) -> None:
        """
        Applies t-SNE for dimensionality reduction and visualizes the clusters.
        """
        if self.feature_vectors.size == 0 or self.cluster_labels.size == 0:
            logger.error("Skipping t-SNE: Features or labels are missing.")
            return

        logger.info(f"Applying t-SNE to reduce to {n_components} dimensions...")
        tsne: TSNE = TSNE(n_components=n_components, random_state=self.random_state, n_jobs=-1)
        data_tsne: np.ndarray = tsne.fit_transform(self.feature_vectors)

        plt.style.use('seaborn-v0_8-whitegrid')
        
        if n_components == 3:
            fig: plt.Figure = plt.figure(figsize=(10, 8))
            ax: Axes3D = fig.add_subplot(projection='3d')
            scatter = ax.scatter(data_tsne[:, 0], data_tsne[:, 1], data_tsne[:, 2],
                                 c=self.cluster_labels, cmap='tab10', s=15, alpha=0.8)
            ax.set_title(f't-SNE 3D with K-Means Clusters {title_suffix}')
            ax.set_xlabel(dim_labels[0] if dim_labels and len(dim_labels) > 0 else 'Dim 1')
            ax.set_ylabel(dim_labels[1] if dim_labels and len(dim_labels) > 1 else 'Dim 2')
            ax.set_zlabel(dim_labels[2] if dim_labels and len(dim_labels) > 2 else 'Dim 3')

        elif n_components == 2:
            plt.figure(figsize=(10, 8))
            scatter = plt.scatter(data_tsne[:, 0], data_tsne[:, 1],
                                  c=self.cluster_labels, cmap='tab10', s=15, alpha=0.8)
            plt.colorbar(scatter, label='Cluster')
            plt.title(f't-SNE 2D with K-Means Clusters {title_suffix}')
            plt.xlabel(dim_labels[0] if dim_labels and len(dim_labels) > 0 else 'Dim 1')
            plt.ylabel(dim_labels[1] if dim_labels and len(dim_labels) > 1 else 'Dim 2')

        plt.show()


    def display_cluster_sample_images(self, dataset_path: str, num_samples_per_cluster: int = 5) -> None:
        """
        Displays a sample of unique full images for each cluster.
        """
        if self.cluster_labels.size == 0 or self.kmeans_model is None:
            logger.error("Skipping image display: Clustering has not been performed.")
            return

        unique_images_in_cluster: Dict[int, List[str]] = defaultdict(list)
        
        # Map all unique images to their corresponding cluster
        for label, img_id in zip(self.cluster_labels, self.image_file_ids):
            if img_id not in unique_images_in_cluster[label]:
                unique_images_in_cluster[label].append(img_id)

        logger.info("Displaying sample images for each cluster...")
        
        for cluster_id in range(self.kmeans_model.n_clusters):
            unique_images: List[str] = unique_images_in_cluster[cluster_id]
            
            # Select samples, ensuring we don't exceed the number of unique images
            samples_to_show = unique_images[:num_samples_per_cluster]
            
            if not samples_to_show:
                logger.warning(f"Cluster {cluster_id} has no unique images to display.")
                continue

            plt.figure(figsize=(4 * len(samples_to_show), 4))
            plt.suptitle(f"Sample Images for Cluster {cluster_id} (Total unique: {len(unique_images)})", fontsize=14)

            for i, img_name in enumerate(samples_to_show):
                img_path: Path = Path(dataset_path) / img_name
                img: np.ndarray | None = cv2.imread(str(img_path))
                
                if img is None:
                    logger.warning(f"Could not load image {img_name} for visualization. Skipping.")
                    continue

                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # Resize image for display
                h: int
                w: int
                h, w = img.shape[:2]
                new_w: int = 200
                new_h: int = int(h * (new_w / w))
                img_resized: np.ndarray = cv2.resize(img, (new_w, new_h))

                plt.subplot(1, len(samples_to_show), i + 1)
                plt.imshow(img_resized)
                plt.title(f"{img_name}", fontsize=8)
                plt.axis('off')
            
            plt.tight_layout(rect=[0, 0, 1, 0.95]) # Adjust layout to prevent title overlap
            plt.show()


    def run_analysis(self, dataset_path: str, model_path: str, num_samples: int = 5) -> None:
        """
        Executes the full clustering and visualization pipeline.
        
        Args:
            dataset_path: Path to the image folder (e.g., 'images/val').
            model_path: Path to the trained YOLO model file (e.g., 'best.pt').
            num_samples: Number of sample images to display per cluster.
        """
        logger.info("==============================================")
        logger.info("Starting Feature Clustering Analysis")
        logger.info("==============================================")
        
        try:
            logger.info(f"Loading YOLO model from {model_path}...")
            yolo_model: YOLO = YOLO(model_path)
        except Exception as e:
            logger.exception(f"Error loading YOLO model: {e}")
            return

        # STAGE 1: Feature Extraction
        self.extract_features_from_images(yolo_model, dataset_path)
        
        if self.feature_vectors.size == 0:
            logger.error("Analysis aborted: No features extracted.")
            return

        # STAGE 2: Clustering
        self.perform_kmeans_clustering()

        # STAGE 3: Visualization (t-SNE)
        self.visualize_tsne(
            n_components=2, 
            title_suffix=f"(K={self.n_clusters})", 
            dim_labels=['t-SNE Dim 1', 't-SNE Dim 2']
        )
        
        # STAGE 4: Sample Visualization
        self.display_cluster_sample_images(dataset_path, num_samples_per_cluster=num_samples)
        
        logger.info("==============================================")
        logger.info("Clustering Analysis Completed.")
        logger.info("==============================================")