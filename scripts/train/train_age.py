
from src.dataloader import get_dataloaders
from src.age.models.resnet_age import ResNetAge
from src.age.trainer.trainer import AgeTrainer
from src.age.models.baseline import Baseline
from src.age.utils import plot_metrics

import torch.nn as nn
import torch.optim as optim
from torchvision import transforms

def train_age_model(config):
    
    transform = transforms.Compose([
        transforms.Resize((
            config['dataset']['image_size'], 
            config['dataset']['image_size']
            )),
        transforms.ToTensor(),
        transforms.Normalize(
                    mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]
                    ),
    ])

    train_loader, val_loader = get_dataloaders(
        image_dir=config['dataset']['image_dir'],
        labels_path=config['dataset']['csv_path'],
        label_type=config['dataset']['label_type'],
        batch_size=config['dataset']['batch_size'],
        val_size=config['dataset']['val_size'],
        random_seed=config['dataset'].get('random_seed', 42), # default seed
        transform=transform,
    )
    
    if config['train']['model'] == 'baseline':
        model = Baseline()
    else: 
        model = ResNetAge()
    
    trainer = AgeTrainer(
        model=model,
        config=config
    )
    
    trainer.fit(train_loader, val_loader)

    log_path = f"checkpoints/age/{config['train']['model_name']}.csv"
    plot_metrics(log_path)
