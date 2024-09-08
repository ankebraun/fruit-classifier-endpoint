import wandb
import os
import torch
from torch import nn
from torchvision import transforms
from torchvision.models import resnet18, ResNet

#TODO: remember to delete the use of the loadotenv
# and os.getenv entries when we use the docker image later
from loadotenv import load_env

load_env(file_loc= '/workspaces/fruit-classifier-endpoint/app/.env')

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
download_artifact()




