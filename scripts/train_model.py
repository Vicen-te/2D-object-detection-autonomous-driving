from ultralytics import YOLO, settings
from ultralytics.utils.benchmarks import benchmark
from pathlib import Path
import torch, torchvision
import yaml

# Datos y configuración común
data_yaml_path = Path(__file__).parent.parent / "dataset" / 'dataset_yolo.yaml'
cfg_yaml_path = Path(__file__).parent.parent / "yamls" / 'cfg.yaml'
adamw_yaml_path = Path(__file__).parent.parent / "yamls" / 'adamw.yaml'
custom_model_path = Path(__file__).parent.parent / "yamls" / 'custom_model.yaml'
project_path = 'training_results'

# Función auxiliar para entrenar
def train_model(model, project, name, epochs=50, resume=False, optimizer='Adam', custom_data_aug=False, patience=5):
    # Cambiar optimizador en hyperparams
    # cfg = None
    # match optimizer:
    #     case 'SGD':
    #         cfg = {
    #             'optimizer': 'SGD',
    #             'lr0': 0.01,
    #             'weight_decay': 0.0005,
    #         }
    #     case 'AdamW':
    #         cfg = {
    #             'optimizer': 'AdamW',
    #             'lr0': 0.001,
    #             'weight_decay': 0.01, #< l2 regularization
    #         }
    #     case 'Adam':
    #         cfg = {
    #             'optimizer': 'Adam',
    #             'lr0': 0.0005,  # lr distinto para diferenciar
    #             'weight_decay': 0.0001,
    #         }
    #     case 'RMSprop':
    #         cfg = {
    #             'optimizer': 'RMSprop',
    #             'lr0': 0.0008,
    #             'weight_decay': 0.0003,
    #         }

    # if (custom_data_aug):
    #     cfg = {
    #         'auto_augment': 'none',
    #     }

    # Carga el YAML original
    #with open(cfg_yaml_path, 'r') as f:
    #    cfg_yaml = yaml.safe_load(f)

    # Actualiza los parámetros seleccionados
    #cfg_yaml.update(cfg)
    #print(f"cfg_yaml: {cfg_yaml}")

    # Guarda el YAML actualizado en el nuevo archivo
    #with open(new_cfg_yaml_path, 'w') as f:
    #    yaml.dump(cfg_yaml, f)
    
    model.train(
        data=data_yaml_path,
        cfg=adamw_yaml_path,
        project=project,
        name=name,
        epochs=epochs,
        resume=resume,
        patience=patience,
        batch=32, #< 16-32 -> 8gb vram, 32-64 -> 16gb vram
        device='cuda' if torch.cuda.is_available() else 'cpu',
    )

def train():
    #mlflow.set_tracking_uri("file:///C")
    settings.update({"tensorboard": True, "mlflow": True})

    #("./training_results/yolo11n_model/weights/last.pt")

    # 3) Fine-tuning (modelo preentrenado YOLO11n, entrenar todo)
    model3 = YOLO('yolo11n.pt')
    # No congelar nada, entrenar todo
    train_model(model3, project_path, 'model_finetuning', epochs=1000, resume=False, optimizer='AdamW')

    # 1) From scratch (pesos aleatorios)
    #model1 = YOLO(cfg_yaml_path)  # Solo config, sin pesos preentrenados
    #train_model(model1, project_path, 'model_from_scratch', epochs=1000, custom_data_aug=True, resume=False, optimizer='SGD')

    # 2) Transfer learning (modelo preentrenado YOLO11n, congelar backbone y neck)
    #model2 = YOLO('yolo11n.pt')

    # Congelar backbone y neck (head freeze)
    #for param in model2.model.backbone.parameters():
    #    param.requires_grad = False
    #for param in model2.model.neck.parameters():
    #    param.requires_grad = False

    #for name, param in model.model.backbone.named_parameters():
       # if "layer4" in name:  # por ejemplo, congelar solo capa 'layer4'
       # param.requires_grad = False
    #train_model(model2, project_path, 'model_transfer_learning', epochs=1000, resume=False, optimizer='AdamW')

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