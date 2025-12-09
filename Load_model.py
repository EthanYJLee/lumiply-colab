from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
import torch
import cv2
import numpy as np
from PIL import Image
import os

# 2. 모델 경로
# 학습 때 사용했던 Base Model (Cross-Attention 튜닝된 모델)
BASE_MODEL_PATH = "./crossattn_checkpoints/best_crossattn_offwhite.ckpt"

# 3. 기타 설정
CONFIG_PATH = "./models/cldm_v21_LumiNet.yaml"

def load_model():
    print("🚀 모델 로딩 및 검증 시작...")
    config = OmegaConf.load(CONFIG_PATH)
    model = instantiate_from_config(config.model).cpu()
    model.add_new_layers()
    
    # 1. 기본 가중치 로드
    if os.path.exists(BASE_MODEL_PATH):
        print(f"📦 Base Model 로드: {BASE_MODEL_PATH}")
        model.load_state_dict(load_state_dict(BASE_MODEL_PATH, location='cpu'), strict=False)
    else:
        print(f"❌ Base Model 없음: {BASE_MODEL_PATH}"); exit()
        
    model.cuda()
    model.eval()
    return model

if __name__ == '__main__':
    load_model()