import torch, mlflow
import tensorboard
from ultralytics import YOLO, settings
from ultralytics.utils.benchmarks import benchmark
from pathlib import Path


def train_model(model_path, data_yaml_path, cfg_yaml_path, project_path, name, unfreeze=0):
    mlflow.set_tracking_uri("file:///C:\\Users\\Alumno.DESKTOP-GV16N45.000\\Desktop\\object-detection-in-2D-environments-for-autonomous-driving\\mlflow")
    settings.update({"tensorboard": True, "mlflow": True})

    model = YOLO(model_path) 

    if unfreeze >= 1:
        for param in model.model.backbone.parameters():
            param.requires_grad = False
    if unfreeze >= 2:
        for param in model.model.neck.parameters():
            param.requires_grad = False
    # Not recomended to freeze head, but if you want to do it
    # unfreeze >= 3 --> fine-tuning != transfer learning
    if unfreeze >= 3:
        for param in model.model.head.parameters():
            param.requires_grad = False
        
    # Specific layer freezing example
    # for name, param in model2.model.backbone.named_parameters():
       # if "layer4" in name:  # por ejemplo, congelar solo capa 'layer4'
       # param.requires_grad = False
    
    model.train(
        data=data_yaml_path,
        cfg=cfg_yaml_path,
        project=project_path,
        name=name,
        batch=32, #< 16-32 -> 8gb vram, 32-64 -> 16gb vram
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )


def evaluate_model(model_path, val_results_path, data_yaml_path):
    
    # Load the model
    model = YOLO(model_path)

    # evaluate the model on the validation set
    # This will run inference on the validation set and return metrics
    metrics = model.val(data=data_yaml_path)
    print(metrics) # results.pandas().xywh métricas formato pandas DataFrame

    # Visualize the model predictions on the validation set
    # model.plot_results(save_dir=val_results_path)  # This will plot the results of the validation set

    # If you want to visualize a specific image, you can use:
    # results = model.predict(source='path/to/your/image.jpg', conf=0.25, save=True)  # Cambia la ruta a tu imagen
    # 'save=True' guarda las imágenes con las cajas predichas en 'runs/detect/predict' por defecto

    # También puedes iterar sobre los resultados y mostrar en consola info:
    #for r in results:
    #    print(r.boxes)  # cajas detectadas

    # Mostrar resultados
    # results.show()  # Muestra la imagen con las predicciones

    # Benchmark
    #benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)