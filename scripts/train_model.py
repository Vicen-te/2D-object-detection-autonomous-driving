from ultralytics import YOLO, settings
from ultralytics.utils.benchmarks import benchmark

import torch, torchvision

def train(resume=False, pretrained=False):
    #mlflow.set_tracking_uri("file:///C")
    settings.update({"tensorboard": True, "mlflow": True})

    # Cargar el modelo preentrenado (puedes elegir otro modelo como 'yolov5m', 'yolov5x', etc.)
    model = YOLO('yolo11n.pt')  # Usando el modelo YOLOv11n preentrenado
        
    # Entrenar el modelo con tus datos
    model.train(
        data='./yamls/dataset_yolo.yaml', 
        cfg="./yamls/cfg.yaml", 
        project= "./training_results",  # Carpeta donde se guardarán los resultados
        name="yolo11n_model",
        resume=resume, # Reanudar el entrenamiento desde el último punto de control
        pretrained=pretrained, # Sin Fine-tuning
        # weights='./yolo11n_model/weights/last.pt',
        device='cuda' if torch.cuda.is_available() else 'cpu',  # Usa GPU si está disponible
    )

    # El modelo entrenado se guarda automáticamente en 'runs/train/exp'

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