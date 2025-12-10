'''
train_adaptors.py는 ControlNet의 latent intrinsic encoder(light_encoder)와 MLP Adaptor(adaptor)를 학습시키는 코드입니다.
본 튜닝의 목적은 조명정보에 원하는 색상을 전이시키도록하는것입니다.
데이터셋은 같은 공간이지만 다른 조명조건을 가진 'off'과 'color' 이미지를 가진 폴더들을 보유한 상태에서 학습을 진행하여야합니다. 
학습 중에는 'off'와 'white_ref'를 입력받아 U-net에서 생성된 노이즈와 실제값인 'color'에서 생성된 노이즈를 비교해 Loss를 정의하고
Loss가 줄어드는 방향으로 진행하게됩니다.
여기서 'off'는 조명이 꺼진사진, 'color'는 색이 다른 조명이 켜진사진, 'white_ref'는 흰색이미지로 '조명을 켜'라는 일종의 입력신호입니다.
노이즈가 서로 같으면 같은 이미지를 생성하기때문에 Loss가 적으면 on과 비슷한 이미지를 생성해 내고 있다라고 볼 수 있습니다.
매 epoch마다 현재 가중치를 pth파일로 저장하며, 현재 가중치로 추론한 검증이미지를 출력합니다.
'''




import torch
import os
import cv2
import numpy as np
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from cldm.model import load_state_dict
from cldm.ddim_hacked import DDIMSampler

# ==================================================================================
# [1] 데이터셋 클래스
# off와 color 페어 있는 폴더 찾기
# ==================================================================================
class ColorLightDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_pairs = []
        
        print(f"\n📂 '{root_dir}' 데이터 스캔 중...")
        
        # dataset 구조: root/00001/off.jpg, color.jpg
        for root, dirs, files in os.walk(root_dir): # off와 color가 있는 디렉토리 스캔
            if 'off.jpg' in files and 'red.jpg' in files: # color 변경 시 여기 수정 ex) red > yellow / 파일명 color아님 주의
                off_path = os.path.join(root, 'off.jpg')
                color_path = os.path.join(root, 'red.jpg') # color 변경 시 여기 수정 ex) red > yellow / 파일명 color아님 주의
                self.data_pairs.append({'off': off_path, 'color': color_path})

        if len(self.data_pairs) > 0:
            print(f"  ✅ 총 {len(self.data_pairs)}개의 데이터 쌍을 찾았습니다.")
        else:
            print("  ❌ 경고: 데이터를 찾지 못했습니다. 경로를 확인해주세요.")

    def __len__(self):
        return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        
        # 이미지 로드 및 리사이즈 (512x512)
        src_off = Image.open(item['off']).convert("RGB").resize((512, 512), Image.BICUBIC) # off 이미지 로드
        tgt_color = Image.open(item['color']).convert("RGB").resize((512, 512), Image.BICUBIC) # color 이미지 로드
        white_ref = Image.new("RGB", (512, 512), (255, 255, 255)) #white_ref 생성

        # 1. Hint 구성 (off, white_reference)
        t_off = torch.from_numpy(np.array(src_off).astype(np.float32)/255.0).permute(2,0,1)
        t_white = torch.from_numpy(np.array(white_ref).astype(np.float32)/255.0).permute(2,0,1)
        
        # Hint: OFF + White = control Net으로 전달
        hint = torch.cat((t_off, t_white), dim=0)
        # color 이미지 (정답값)
        t_color = torch.from_numpy(np.array(tgt_color).astype(np.float32)/255.0).permute(2,0,1)
        jpg = (t_color * 2.0) - 1.0
        
        return {"jpg": jpg, "hint": hint}

# ==================================================================================
# [2] 학습 메인 함수
# ==================================================================================
def train_color():
    # --- 설정 ---
    save_dir = "./adaptors_red"  # color 변경 시 여기 수정     
    log_img_dir = "./train_log_red"   # color 변경 시 여기 수정
    dataset_root = "./images"    # 학습데이터 디렉토리
    
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(log_img_dir, exist_ok=True)
    
    print("\n [Color Light Training] 학습 준비...")
    
    # 1. 모델 로드
    config = OmegaConf.load('./models/cldm_v21_LumiNet.yaml')
    config.model.params.control_stage_config.params.use_checkpoint = False
    config.model.params.unet_config.params.use_checkpoint = False
    
    model = instantiate_from_config(config.model).cpu()
    model.add_new_layers() # Layer 초기화
    
    # 가중치 로드
    ckpt_path = "./ckpt/trained_crossattn.ckpt" # 학습된 cross_attention ckpt
    if os.path.exists(ckpt_path):
        model.load_state_dict(load_state_dict(ckpt_path, 'cpu'), strict=False)
        print("📦 모델 로드 완료")
    
    model.train().cuda()
    
    # 2. 학습 대상 설정 (Adaptor + Encoder)
    trainable_params = []
    for param in model.parameters(): param.requires_grad = False # 전체 파라미터 동결
    
    if hasattr(model.control_model, 'prior_extracter'):
        adaptor = model.control_model.prior_extracter.light_decoder
        encoder = model.control_model.prior_extracter.model_latents.light_encoder
        # light_encoder와 adaptor는 동결해제
        for param in adaptor.parameters(): param.requires_grad = True; trainable_params.append(param) #adaptor 동결해체
        for param in encoder.parameters(): param.requires_grad = True; trainable_params.append(param) #encoder 동결해체
            
        print(" Light Encoder & Adaptor 학습 모드 설정됨")

    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5) # optimizer 설정
    

    
    # 3. 데이터 로더
    train_dataset = ColorLightDataset(os.path.join(dataset_root, 'train')) #./images/train
    val_dataset = ColorLightDataset(os.path.join(dataset_root, 'validation')) # ./images/validation
    
    train_dataloader = DataLoader(train_dataset, batch_size=5, shuffle=True, num_workers=2) # batch size설정
    val_dataloader = DataLoader(val_dataset, batch_size=5, shuffle=False, num_workers=2) # batch size설정
    
    # 검증용 고정 샘플 (학습 중 이미지 생성을 위해 validation에서 1개 확보)
    viz_batch = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False))) if len(val_dataset) > 0 else None

    if len(train_dataset) == 0: print("❌ 학습 데이터가 없습니다."); return

    epochs = 50 # epoch 설정

    print(f"\n 학습 루프 시작 (Total Epochs: {epochs})")
    
    for epoch in range(epochs):
        # ----------------------
        # [A] Train Loop
        # ----------------------
        model.train()
        train_loss = 0
        for batch in train_dataloader:
            x = batch["jpg"].cuda()     # 정답값(color)
            hint = batch["hint"].cuda() # off + white
            
            with torch.no_grad():
                z = model.get_first_stage_encoding(model.encode_first_stage(x)).detach()
                # 정답 이미지(x)를 VAE를 통해 Latent 공간(z)으로 압축
            
            c = {"c_concat": [hint], "c_crossattn": [model.get_learned_conditioning([""] * x.shape[0])]}
            # c_concat(이미지 힌트)은 ControlNet으로, c_crossattn(더미 프롬프트)은 Diffusion 모델로 전달

            loss, _ = model(z, c) #./ldm/models/diffusion/ddpm.py의 p_losses함수 호출
            # Latent(z)에 노이즈를 추가하고, 모델이 그 노이즈를 얼마나 잘 예측하는지 계산 (MSE Loss)
            # 노이즈 예측을 잘 할수록 실제값과 비슷한 이미지 생성
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
        avg_train_loss = train_loss / len(train_dataloader)

        # ----------------------
        # [B] Validation Loop (Loss 계산)
        # ----------------------
        model.eval()
        val_loss = 0
        if len(val_dataloader) > 0:
            with torch.no_grad():
                for batch in val_dataloader:
                    x_val = batch["jpg"].cuda() #정답값
                    hint_val = batch["hint"].cuda() #off + white
                    
                    z_val = model.get_first_stage_encoding(model.encode_first_stage(x_val)).detach()
                    c_val = {"c_concat": [hint_val], "c_crossattn": [model.get_learned_conditioning([""] * x_val.shape[0])]}
                    
                    # Validation Loss 계산
                    v_loss, _ = model(z_val, c_val)
                    val_loss += v_loss.item()
            
            avg_val_loss = val_loss / len(val_dataloader)
        else:
            avg_val_loss = 0.0

        print(f"Epoch {epoch+1:03d}/{epochs} | Train Loss: {avg_train_loss:.6f} | Val Loss: {avg_val_loss:.6f}")

        # ----------------------
        # [C] 매 에포크 현재 가중치 pth로 저장
        # ----------------------
        save_name = f"color_epoch_{epoch+1:03d}.pth"
        save_path = os.path.join(save_dir, save_name)
        
        save_dict = {
            'light_encoder': encoder.state_dict(),
            'light_decoder': adaptor.state_dict()
        }
        torch.save(save_dict, save_path)
        print(f"  {save_name} 저장 완료")

        # ----------------------
        # [D] 매 에포크 검증 이미지 생성
        # 학습이 잘 되고 있는지 눈으로 확인 (현재 가중치로 추론)
        # ----------------------
        if viz_batch is not None:
            sampler = DDIMSampler(model)
            with torch.no_grad():
                c_cat = viz_batch["hint"].cuda() # off + white
                # Unconditional Conditioning
                c_uncond = model.get_unconditional_conditioning(c_cat.shape[0])
                cond = {"c_concat": [c_cat], "c_crossattn": [c_uncond]} # controlNet으로 이동
                
                # Sampling
                shape = (4, 512 // 8, 512 // 8)
                samples, _ = sampler.sample(50, 1, shape, cond, verbose=False, unconditional_guidance_scale=9.0)
                #Diffusion에서 추론
                
                # Decoding
                x_sample = model.decode_first_stage(samples)
                x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
                x_sample = x_sample.cpu().permute(0, 2, 3, 1).numpy()[0] * 255
                
                # Image Saving
                img_save_path = os.path.join(log_img_dir, f"val_epoch_{epoch+1:03d}.jpg")
                cv2.imwrite(img_save_path, cv2.cvtColor(x_sample.astype(np.uint8), cv2.COLOR_RGB2BGR))
                print(f"   📸 검증 이미지 생성 완료: {img_save_path}")

    print("\n🎉 color 학습 완료!")

if __name__ == '__main__':
    train_color()