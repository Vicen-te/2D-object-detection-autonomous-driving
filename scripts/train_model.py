import torch
import mlflow, tensorboard
from ultralytics import YOLO, settings
from ultralytics.utils.benchmarks import benchmark
from pathlib import Path


def train_model(model_path, data_yaml_path, cfg_yaml_path, project_path, name, unfreeze=0):
    #mlflow.set_tracking_uri("file:///C")
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



    # Evaluar el modelo en el conjunto de prueba
    # results = model.val()

    # Mostrar métricas de evaluación
    # print(results.pandas().xywh)  # Las métricas están en formato de pandas DataFrame




    # Hacer inferencias con el modelo entrenado
    # results = model.predict(source='path/to/your/test_image.jpg', conf=0.25)  # Cambia la ruta a tu imagen de prueba

    # Mostrar resultados
    # results.show()  # Muestra la imagen con las predicciones




    # Benchmark
    #benchmark(model="yolo11n.pt", data="coco8.yaml", imgsz=640, half=False, device=0)