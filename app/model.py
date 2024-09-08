import wandb
import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet

#TODO: remember to delete the use of the loadotenv
# and os.getenv entries when we use the docker image later
from loadotenv import load_env

#load_env(file_loc= '/workspaces/fruit-classifier-endpoint/app/.env')

MODELS_DIR = 'models'
MODEL_FILE_NAME = 'model.pth'

CATEGORIES = ['freshapples', 'freshbanana', 'freshorages',
              'rottenapples', 'rottenbanana', 'rottenorages']

#todo: delete
# print(os.getenv('WANDB_API_KEY'))

def download_artifact():
    assert 'WANDB_API_KEY' in os.environ, 'Please enter WANDB_API_KEY as environmental variable.'

    wandb.login() # here we get access to the artifact registry
    # go to artifact registry in wandb to find this full path
    #anke-braun86-personal/banana_apple_orange/restnet18:v0
    wandb_org = os.environ.get('WANDB_ORG')
    wandb_project = os.environ.get('WANDB_PROJECT')
    wandb_model_name = os.environ.get('WANDB_MODEL_NAME')
    wandb_model_version = os.environ.get('WANDB_MODEL_VERSION')

    artifact_path = f'{wandb_org}/{wandb_project}/{wandb_model_name}:{wandb_model_version}'
    print(artifact_path)
    artifact = wandb.Api().artifact(artifact_path, type='model')
    artifact.download(root=MODELS_DIR)


#download_artifact()

def get_raw_model() -> ResNet:
    N_CLASSES = 6

    model = resnet18(weights=None)

    # check that this architecture is the same in your Kaggle notebook
    model.fc = nn.Sequential(
        nn.Linear(512, 512),
        nn.ReLU(),
        nn.Linear(512, N_CLASSES)
    )

    return model

def load_model() -> ResNet:
    """this returns the model its wandb weights"""
    download_artifact()
    model = get_raw_model()
    model_state_dict_path = os.path.join(MODELS_DIR, MODEL_FILE_NAME)
    model_state_dict = torch.load(model_state_dict_path, map_location = 'cpu')
    model.load_state_dict(model_state_dict, strict = True) # This was strict = False in the code
    # This turns off BatchNorm and droput this is inference state (not training)
    model.eval()
    return model

def load_transforms() -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225]),
    ])



