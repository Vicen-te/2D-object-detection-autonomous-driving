from ultralytics import YOLO, settings
from ultralytics.utils.benchmarks import benchmark

import torch, torchvision

# Datos y configuración común
data_yaml = './yamls/dataset_yolo.yaml'
cfg_yaml = './yamls/cfg.yaml'
hyp_yaml = './yamls/hyp.yaml'
project_path = './training_results'

# Función auxiliar para entrenar
def train_model(model, project, name, epochs=50, resume=False, optimizer='Adam', patience=5):
    # Cambiar optimizador en hyperparams
    hyp = None
    match optimizer:
        case 'SGD':
            hyp = {
                'optimizer': 'SGD',
                'lr0': 0.01,
                'momentum': 0.937,
                'weight_decay': 0.0005,
            }
        case 'Adam':
            hyp = {
                'optimizer': 'AdamW',
                'lr0': 0.001,
                'weight_decay': 0.01, #< l2 regularization
            }
        case 'AdamW':
            hyp = {
                'optimizer': 'Adam',
                'lr0': 0.0005,  # lr distinto para diferenciar
                'weight_decay': 0.0001,
            }
        case 'RMSprop':
            hyp = {
                'optimizer': 'RMSprop',
                'lr0': 0.0008,
                'momentum': 0.9,
                'weight_decay': 0.0003,
            }

    model.train(
        data=data_yaml,
        cfg=cfg_yaml,
        hyp=hyp_yaml if hyp is None else hyp,
        project=project,
        name=name,
        epochs=epochs,
        resume=resume,
        patience=patience,
        tensorboard=True,
        mlflow=True,
        batch=16, #< 16-32 -> 8gb vram, 32-64 -> 16gb vram
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

def train():
    #mlflow.set_tracking_uri("file:///C")
    settings.update({"tensorboard": True, "mlflow": True})

    #("./training_results/yolo11n_model/weights/last.pt")

    # 1) From scratch (pesos aleatorios)
    model1 = YOLO(cfg_yaml)  # Solo config, sin pesos preentrenados
    train_model(model1, project_path, 'model_from_scratch', epochs=50, resume=False, optimizer='SGD')

    # 2) Transfer learning (modelo preentrenado YOLO11n, congelar backbone y neck)
    model2 = YOLO('yolo11n.pt')

    # Congelar backbone y neck (head freeze)
    for param in model2.model.backbone.parameters():
        param.requires_grad = False
    for param in model2.model.neck.parameters():
        param.requires_grad = False

    #for name, param in model.model.backbone.named_parameters():
       # if "layer4" in name:  # por ejemplo, congelar solo capa 'layer4'
       # param.requires_grad = False
    train_model(model2, project_path, 'model_transfer_learning', epochs=50, resume=False, optimizer='AdamW')

    # 3) Fine-tuning (modelo preentrenado YOLO11n, entrenar todo)
    model3 = YOLO('yolo11n.pt')
    # No congelar nada, entrenar todo
    train_model(model3, project_path, 'model_finetuning', epochs=50, resume=False, optimizer='AdamW')

    # 4) From scratch (pesos aleatorios)
    model4 = YOLO(cfg_yaml)
    train_model(model4, project_path, 'model_from_scratch_rmsprop', epochs=60, resume=False, optimizer='Adam')

    # 5) Transfer learning (congelar solo backbone)
    model5 = YOLO('yolo11n.pt')
    for param in model2.model.backbone.parameters():
        param.requires_grad = False
    train_model(model5, project_path, 'model_sgd_optimizer', epochs=50, resume=False, optimizer='RMSprop')


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