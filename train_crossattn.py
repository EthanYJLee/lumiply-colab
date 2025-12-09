import torch
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import glob
import time
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from omegaconf import OmegaConf
from ldm.util import instantiate_from_config
from cldm.model import load_state_dict
from ldm.modules.attention import SpatialTransformer
from cldm.ddim_hacked import DDIMSampler

# ==================================================================================
# [1] 데이터셋 클래스
# ==================================================================================
class LightingDataset(Dataset):
    def __init__(self, root_dir):
        self.root_dir = root_dir
        self.data_pairs = []
        
        print(f"🔍 '{root_dir}' 데이터 스캔 중...")
        # 모든 하위 폴더 탐색
        for root, dirs, files in os.walk(root_dir):
            off_f, on_f = None, None
            for f in sorted(files):
                lower = f.lower()
                if not lower.endswith(('.jpg', '.jpeg', '.png', '.avif', '.webp')): continue #이미지만 탐색
                
                if 'off' in lower: off_f = os.path.join(root, f)
                elif 'on' in lower and 'color' not in lower: on_f = os.path.join(root, f)
            
            if off_f and on_f: #on, off가 둘 다 있는 디렉토리 찾기
                self.data_pairs.append({'off': off_f, 'on': on_f})

        print(f"  ✅ 총 {len(self.data_pairs)}개의 쌍을 찾았습니다.")

    def __len__(self): return len(self.data_pairs)

    def __getitem__(self, idx):
        item = self.data_pairs[idx]
        
        src_off = Image.open(item['off']).convert("RGB").resize((512, 512), Image.BICUBIC) #off이미지 로드
        src_on  = Image.open(item['on']).convert("RGB").resize((512, 512), Image.BICUBIC) #on이미지 로드
        
        # White Reference 생성
        white_ref = Image.new("RGB", (512, 512), (255, 255, 255))
        
        t_off   = torch.from_numpy(np.array(src_off).astype(np.float32)/255.0).permute(2,0,1) #정규화
        t_white = torch.from_numpy(np.array(white_ref).astype(np.float32)/255.0).permute(2,0,1)
        t_on    = torch.from_numpy(np.array(src_on).astype(np.float32)/255.0).permute(2,0,1)
        
        # Hint: OFF + White = control Net으로 전달
        hint = torch.cat((t_off, t_white), dim=0)
        # Target: ON = 정답값
        jpg = (t_on * 2.0) - 1.0
        
        return {"jpg": jpg, "hint": hint}

# ==================================================================================
# [2] 유틸리티: 그래프 그리기
# ==================================================================================
def plot_loss_graph(train_losses, val_losses, save_path="loss_graph.png"):
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Train Loss', color='blue', alpha=0.7)
    plt.plot(val_losses, label='Validation Loss', color='orange', linewidth=2)
    plt.title('Training & Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path)
    plt.close()
    print(f"📊 그래프 저장 완료: {save_path}")

# ==================================================================================
# [3] 학습 메인 함수
# ==================================================================================
def train_step_safe():
    # --- 설정 ---
    # 경로
    train_root = "./images/train" # train 데이터 디렉토리
    val_root   = "./images/validation" # validation 데이터 디렉토리
    
    # 저장소
    save_dir = "./crossattn_checkpoints" # ckpt 저장 위치
    sample_dir = "./train_log_images" # 검증 이미지 저장 위치
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(sample_dir, exist_ok=True)
    
    # 하이퍼파라미터
    batch_size = 5  
    epochs = 50
    learning_rate = 1e-5
    
    print("\n Cross-Attention 학습")
    print(f"   - Batch Size: {batch_size}")
    print(f"   - Max Epochs: {epochs}")

    # 1. 모델 로드
    config = OmegaConf.load('./models/cldm_v21_LumiNet.yaml')
    config.model.params.control_stage_config.params.use_checkpoint = False
    config.model.params.unet_config.params.use_checkpoint = False
    
    model = instantiate_from_config(config.model).cpu()
    model.add_new_layers()
    
    # 기본 가중치 로드
    if os.path.exists("./ckpt/LumiNet.ckpt"):
        model.load_state_dict(load_state_dict("./ckpt/LumiNet.ckpt", 'cpu'), strict=False)
        print("📦 기본 모델 가중치 로드됨")
    
    model.train().cuda()

    # 2. 학습 대상 설정 (Cross-Attention Only)
    trainable_params = []
    for param in model.parameters(): param.requires_grad = False # 모든 파라미터 동결
    
    attn_count = 0
    for name, module in model.named_modules():
        if isinstance(module, SpatialTransformer): # Diffusion에서 cross attention만 학습 대상에 포함
            for param in module.parameters():
                param.requires_grad = True # cross attention만 동결 해제
                trainable_params.append(param)
            attn_count += 1
            
    print(f"🔓 Cross-Attention Layer ({attn_count}개) 해동 완료")
    
    optimizer = torch.optim.AdamW(trainable_params, lr=learning_rate) # optimizer 설정
    
    # 3. 데이터 로더
    train_dataset = LightingDataset(train_root)
    val_dataset = LightingDataset(val_root)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    # 검증용 고정 샘플 (학습 중 이미지 생성을 위해 validation에서 1개 확보)
    fixed_val_batch = next(iter(DataLoader(val_dataset, batch_size=1, shuffle=False)))

    # -------------------------------------------------------
    # [Resume Logic] 중단된 학습 이어하기
    # -------------------------------------------------------
    start_epoch = 0
    best_val_loss = float('inf')
    train_loss_history = []
    val_loss_history = []
    
    # 마지막 체크포인트 찾기
    resume_ckpt = os.path.join(save_dir, "last_checkpoint.pth")
    if os.path.exists(resume_ckpt):
        print(f" 복구 파일 발견! 학습을 재개합니다: {resume_ckpt}")
        checkpoint = torch.load(resume_ckpt)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        train_loss_history = checkpoint['train_loss_history']
        val_loss_history = checkpoint['val_loss_history']
        print(f"   ▶ {start_epoch} Epoch부터 시작합니다.")
    else:
        print("🆕 새로운 학습을 시작합니다.")

    # -------------------------------------------------------
    # 학습 루프
    # -------------------------------------------------------
    for epoch in range(start_epoch, epochs):
        epoch_start_time = time.time()
        model.train()
        train_loss_sum = 0
        
        # (A) Training
        for batch in train_loader:
            x = batch["jpg"].cuda() # 정답값(on)
            hint = batch["hint"].cuda() # off + white
            
            with torch.no_grad():
                z = model.get_first_stage_encoding(model.encode_first_stage(x)).detach() 
                # 정답 이미지(x)를 VAE를 통해 Latent 공간(z)으로 압축
            
            c = {"c_concat": [hint], "c_crossattn": [model.get_learned_conditioning([""] * x.shape[0])]} 
            # c_concat(이미지 힌트)은 ControlNet으로, c_crossattn(더미 프롬프트)은 Diffusion 모델로 전달
            t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=model.device).long()
            
            loss, _ = model.p_losses(z, c, t) #./ldm/models/diffusion/ddpm.py의 p_losses함수 호출
            # Latent(z)에 노이즈를 추가하고, 모델이 그 노이즈를 얼마나 잘 예측하는지 계산 (MSE Loss)
            # 노이즈 예측을 잘 할수록 실제값과 비슷한 이미지 생성
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            
        avg_train_loss = train_loss_sum / len(train_loader)
        train_loss_history.append(avg_train_loss)
        
        # (B) Validation
        model.eval()
        val_loss_sum = 0
        with torch.no_grad():
            for batch in val_loader:
                x = batch["jpg"].cuda() # 정답값
                hint = batch["hint"].cuda() # off + white
                z = model.get_first_stage_encoding(model.encode_first_stage(x)).detach() # 정답값
                c = {"c_concat": [hint], "c_crossattn": [model.get_learned_conditioning([""] * x.shape[0])]}
                t = torch.randint(0, model.num_timesteps, (z.shape[0],), device=model.device).long()
                
                # Validation은 Loss만 계산
                loss, _ = model.p_losses(z, c, t)
                val_loss_sum += loss.item()
                
        avg_val_loss = val_loss_sum / len(val_loader)
        val_loss_history.append(avg_val_loss)
        
        # (C) 결과 출력
        elapsed = time.time() - epoch_start_time
        print(f"Epoch {epoch+1}/{epochs} ({elapsed:.1f}s) | Train: {avg_train_loss:.5f} | Val: {avg_val_loss:.5f}")

        # (D) 체크포인트 저장 (안전장치)
        # 1. "Last" Checkpoint (매번 덮어쓰기 - 복구용)
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'train_loss_history': train_loss_history,
            'val_loss_history': val_loss_history
        }, resume_ckpt)
        
        # 2. "Periodic" Checkpoint (10 Epoch 마다)
        if (epoch + 1) % 10 == 0:
            periodic_path = os.path.join(save_dir, f"epoch_{epoch+1:03d}.ckpt")
            torch.save(model.state_dict(), periodic_path)
            print(f"   💾 정기 저장 완료: {periodic_path}")
            
            # 그래프 업데이트
            plot_loss_graph(train_loss_history, val_loss_history, "loss_graph_crossattn.png")

        # 3. "Best" Checkpoint (신기록 달성 시)
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            print(f"   🌟 Best Validation Loss! -> best_crossattn.ckpt 저장")
            torch.save(model.state_dict(), os.path.join(save_dir, "best_crossattn.ckpt"))

        # (E) 검증 이미지 생성 (매 Epoch)
        # 학습이 잘 되고 있는지 눈으로 확인 (현재 가중치로 추론)
        sampler = DDIMSampler(model)
        with torch.no_grad():
            c_cat = fixed_val_batch["hint"].cuda()
            c = model.get_unconditional_conditioning(1)
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            shape = (4, 512 // 8, 512 // 8)
            
            samples, _ = sampler.sample(50, 1, shape, cond, verbose=False, unconditional_guidance_scale=9.0)
            x_sample = model.decode_first_stage(samples)
            x_sample = torch.clamp((x_sample + 1.0) / 2.0, min=0.0, max=1.0)
            x_sample = x_sample.cpu().permute(0, 2, 3, 1).numpy()[0] * 255
            
            img_path = os.path.join(sample_dir, f"val_ep{epoch+1:03d}.jpg")
            cv2.imwrite(img_path, cv2.cvtColor(x_sample.astype(np.uint8), cv2.COLOR_RGB2BGR))

    # 최종 그래프
    plot_loss_graph(train_loss_history, val_loss_history, "loss_graph_crossattn_final.png")
    print("\n Cross Attention 학습이 모두 완료되었습니다!")

if __name__ == '__main__':
    train_step_safe()