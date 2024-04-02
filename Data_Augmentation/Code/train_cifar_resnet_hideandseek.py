import os
import lightning as L
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint, ModelSummary
from lightning_models.resnet import ResNet
import utils.custom_transforms as transforms

import torchvision.datasets as datasets
import torch.utils.data as data
import torch
from configuration import getYaml
import random
import numpy as np
import logging

#모델과 데이터
my_model = ResNet
my_data = datasets.CIFAR10

#모델 설정과 어그멘테이션 설정 
model_config_path = 'configuration/model_config.yaml'
hs_config_path ='configuration/hs_config.yaml'
model_args = getYaml(model_config_path) 
aug_args = getYaml(hs_config_path)

#저장할 루트 디렉토리   
CHECKPOINT_ROOT = "./experiment_results"
SAVE_DIR = os.path.join(CHECKPOINT_ROOT, f"{model_args['arch']}{model_args['depth']}_with_HideandSeek")

#pytorch lightning을 이용해서 모델 훈련
def train_model(args,my_model,train_loader, val_loader,saving_dir_path):
    trainer = L.Trainer(
        enable_model_summary=False,
        default_root_dir= saving_dir_path,  # Where to save models
        # We run on a single GPU (if possible)
        accelerator="cuda",
        devices=1,
        max_epochs=args['epochs'],
        callbacks=[
            #ModelSummary(max_depth=-1),
            ModelCheckpoint(
                save_weights_only=False, mode="max", monitor="val_acc",save_last=True, save_top_k=-1,every_n_epochs=1,
            ),  
            LearningRateMonitor("epoch"),
        ],  
        precision="bf16-mixed",
    )  
    trainer.logger._log_graph = True  # If True, we plot the computation graph in tensorboard
    trainer.logger._default_hp_metric = None  # Optional logging argument that we don't need
    
    model = my_model(args)
    trainer.fit(model, train_loader, val_loader)
    test_last_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    model = my_model.load_from_checkpoint(
        trainer.checkpoint_callback.best_model_path
    )  # Load best checkpoint after training
    
    # Test best model on validation and test set
    test_best_result = trainer.test(model, dataloaders=val_loader, verbose=False)
    result = { "last_test": 100-test_last_result[0]["test_acc"], "best_test": 100-test_best_result[0]["test_acc"]}

    #rename best model path
    dir_best = os.path.dirname(trainer.checkpoint_callback.best_model_path)
    file_best = os.path.basename(trainer.checkpoint_callback.best_model_path)
    os.rename(trainer.checkpoint_callback.best_model_path,os.path.join(dir_best,"best_"+ file_best) )
    print(f"save path of best model :  {trainer.checkpoint_callback.best_model_path}")
    
    return model, result
    
def main():
    # Validate dataset
    assert model_args['dataset'] == 'cifar10' or model_args['dataset'] == 'cifar100', 'Dataset can only be cifar10 or cifar100.'
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    # Random seed
    if model_args['manualSeed'] is None:
        model_args['manualSeed'] = random.randint(1, 10000)
    random.seed(model_args['manualSeed'])
    torch.manual_seed(model_args['manualSeed'])
    np.random.seed(model_args['manualSeed'])
    if use_cuda:
        torch.cuda.manual_seed_all(model_args['manualSeed'])
    #torch.set_float32_matmul_precision('high')
    print('==> Preparing dataset %s' %model_args['dataset'] )
    

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.HideAndSeek(probability = aug_args['p'],
                               grid_ratio= aug_args['grid_ratio'], 
                               patch_probabilty= aug_args['patch_p'],
                               value="Z")
    ])
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    
    
    #cifar10 with image id
    trainset = my_data(root='./data', train=True, download=True,transform=transform_train)
    trainloader = data.DataLoader(trainset, batch_size=model_args['train_batch'], shuffle=True, num_workers=model_args['workers'],persistent_workers=True)

    testset = my_data(root='./data', train=False, download=True, transform=transform_test)
    testloader = data.DataLoader(testset, batch_size=model_args['test_batch'], shuffle=False, num_workers=model_args['workers'], persistent_workers=True)
    #train
    model, result=train_model(args=model_args,
                              train_loader=trainloader,
                              val_loader=testloader,
                              saving_dir_path=SAVE_DIR,
                              my_model= my_model
                              )
    print()
    print(f"test error rate ||  {result}")
    print(f"paramters : {sum(np.prod(p.shape) for p in model.parameters())}")
    return result

if __name__ == "__main__":
    try_num= 5
    last = []
    best = []
    for do in range(try_num):
        print(f"-------start train {do+1}-------")
        result=main()
        last.append(result['last_test'])
        best.append(result['best_test'])
        print(f"-------finish train {do+1}-------")
    print(f"all jobs are finished")
    print(f"last results {last}")
    print(f"best results {best}")
    print(f"last mean {np.mean(last)}  last std {np.std(last)}")
    print(f"best mean {np.mean(best)}  last std {np.std(best)}")