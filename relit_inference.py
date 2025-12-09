from share import *

from cldm.model import create_model, load_state_dict
import cv2
import numpy as np
import torch
import einops
from cldm.ddim_hacked import DDIMSampler
from PIL import Image
import os 
import random
from huggingface_hub import hf_hub_download

## using bypass-decoder for better identity preservation, details: https://arxiv.org/pdf/2403.12933
new_decoder = False
resume_path = hf_hub_download(repo_id="xyxingx/LumiNet", filename="LumiNet.ckpt")
N = 1
ddim_steps = 50
model = create_model('./models/cldm_v21_LumiNet.yaml').cpu()
model.add_new_layers()
model.concat = False
model.load_state_dict(load_state_dict(resume_path, location='cuda'))

if new_decoder:
    ae_checkpoint = hf_hub_download(repo_id="xyxingx/LumiNet", filename="new_decoder.ckpt")
    model.change_first_stage(ae_checkpoint)
    
model = model.cuda()
many_iter = 50
model.parameterization = "v"
ddim_sampler = DDIMSampler(model)

imagesets = PATH_OF_INPUT_IMAGE #add your own data_path here

refsets = PATH_OF_REFERENCE  #add your own data_path here
output_path= PATH_OF_OUTPUT
for imn in imagesets:
    img_path = PATH_OF_INPUT_IMAGE + imn
    for fn in refsets:
        shd_path = PATH_OF_INPUT_REFERENCE + fn  # keep your original path logic
        fold_name = output_path + imn + '_' + fn + '/'
        if not os.path.exists(fold_name):
            os.makedirs(fold_name)
        if len(os.listdir(fold_name)) > 20:
            print('skipping exisiting:', imn + fn)
            continue

        for i in range(many_iter):
            seeds = torch.seed()
            torch.manual_seed(seeds)
            torch.cuda.manual_seed(seeds)
            random.seed(seeds)


            input_image_full = cv2.imread(img_path)[..., ::-1] / 255.0
            input_shd_full = cv2.imread(shd_path)[..., ::-1] / 255.0
            orig_h, orig_w = input_image_full.shape[:2]
            
            input_image = cv2.resize(input_image_full, (512, 512))
            input_shd = cv2.resize(input_shd_full, (512, 512))

            control_feat = np.concatenate((input_image, input_shd), axis=2)
            control = torch.from_numpy(control_feat.copy()).float().cuda()
            control = torch.stack([control for _ in range(N)], dim=0)
            control = einops.rearrange(control, 'b h w c -> b c h w').clone()

            input_img = torch.from_numpy(input_image.copy()).float().cuda()
            input_img = input_img.unsqueeze(0)
            input_img = einops.rearrange(input_img, 'b h w c -> b c h w').clone()

            c_cat = control.cuda()
            c = model.get_unconditional_conditioning(N)
            uc_cross = model.get_unconditional_conditioning(N)
            uc_cat = c_cat
            uc_full = {"c_concat": [uc_cat], "c_crossattn": [uc_cross]}
            cond = {"c_concat": [c_cat], "c_crossattn": [c]}
            b, cch, h, w = cond["c_concat"][0].shape
            shape = (4, h // 8, w // 8)

            samples, intermediates = ddim_sampler.sample(
                ddim_steps,
                N,
                shape,
                cond,
                verbose=False,
                eta=0.0,
                unconditional_guidance_scale=9.0,
                unconditional_conditioning=uc_full
            )

            if new_decoder:
                ae_hs = model.encode_first_stage(input_img * 2 - 1)[1]
                x_samples = model.decode_new_first_stage(samples, ae_hs)
            else:
                x_samples = model.decode_first_stage(samples)

            x_samples = x_samples.squeeze(0)
            x_samples = (x_samples + 1.0) / 2.0
            x_samples = x_samples.transpose(0, 1).transpose(1, 2)
            x_samples = x_samples.cpu().numpy()
            x_samples = x_samples.clip(0, 1)
            x_samples = (x_samples * 255).astype(np.uint8)

            # --- resize back to original ratio before saving (max. 512) --- 
            # orig_w, orig_h:
            max_side = 512
            scale = min(max_side / max(orig_w, orig_h), 1.0)  # scale for resize
            
            new_w = max(1, int(round(orig_w * scale)))
            new_h = max(1, int(round(orig_h * scale)))
            interp = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
            x_samples_resized = cv2.resize(x_samples, (new_w, new_h), interpolation=interp)
            image_name = img_path.split('/')[-1]
            Image.fromarray(x_samples_resized).save(fold_name + str(seeds) + '_' + image_name)







