## Lumiply Colab (Flask Inference Server)
![cover](images/lumiply_colab_cover.png)

이 저장소는 원본 [LumiNet](https://github.com/xyxingx/LumiNet/) 코드베이스 위에, **Lumiply 프로젝트에서 사용할 Colab용 추론 서버**를 올려 둔 버전입니다.  
구성은 다음과 같이 이해해주시면 됩니다.

- `lumiply-client` (React SPA) 에서
- `lumiply-server` (FastAPI)를 거쳐
- **이 Colab 런타임에서 돌아가는 LumiNet 모델**에 조명 이미지를 요청하는 구조입니다.

#### 요구사항: 
> GPU with VRAM > 11G  
> open-clip-torch==2.0.1 (필수)  
> [허깅페이스 (필수)](https://huggingface.co/EthanYJ/Lumiply/tree/main)

---

### 1. 전체 아키텍처에서의 역할

- **역할 요약**

  - LumiNet 기반 relighting 모델을 **Colab GPU 위에서 서빙**합니다.
  - FastAPI 서버(`lumiply-server`)가 `/process` 엔드포인트로 이미지를 보내면,
    - `white, red, orange, yellow, green, blue, purple` 색상별 결과를 생성하고
    - 결과 URL을 담은 JSON을 FastAPI에 반환합니다.
  - `/health` 엔드포인트를 통해 Colab 서버 상태를 간단히 확인할 수 있습니다.

- **세 레포지토리 간 관계**

  | 컴포넌트         | 역할                                        |
  | ---------------- | ------------------------------------------- |
  | `lumiply-client` | 방 사진 업로드, 조명 배치, 결과 비교 UI     |
  | `lumiply-server` | 클라이언트 요청 수신, Colab 호출, 상태 관리 |
  | `lumiply-colab`  | LumiNet 기반 조명 생성, `/process` 제공     |

세 레포지토리에서 `lumiply-colab` 은 **모델 서버(ML backend)**, `lumiply-server` 는 **API gateway**, `lumiply-client` 는 **UX 레이어**에 해당합니다.

---

### 2. 디렉터리 구조 (요약)

```bash
lumiply-colab/
├── adaptors/                # 색상별 adaptor 가중치 (adaptor_white.pth, ... )
├── ckpt/                    # base / trained cross-attn, new_decoder 등
├── cldm/, ldm/, modi_vae/   # LumiNet 및 Stable Diffusion 관련 코드
├── images/
│   └── inference/           # /process 요청별 결과(off.png, output_*.jpg)
├── font/                    # demo용 폰트
├── models/                  # LumiNet config (cldm_v21_LumiNet.yaml 등)
├── lumiply_inference.ipynb      # FastAPI와 연동되는 Colab 메인 노트북
├── lumiply_ready_for_git.ipynb
├── relit_inference.py       # 로컬/오프라인 테스트용 스크립트
├── seed_selection.py        # 결과 중 “좋은 seed”를 찾는 optional 유틸
├── flux_cleanup.py          # FLUX 기반 후처리(clean‑up) (optional)
├── gradio_demo.py           # 단독 LumiNet 데모용 Gradio UI
├── Load_model.py            # 모델 로딩 관련 보조 스크립트
├── requirements.txt         # Colab 런타임 의존성
└── README.md
```

> 상위 디렉터리 구조는 LumiNet 원본 레포와 거의 동일하며, 
> Lumiply에 맞춰 `lumiply_inference.ipynb` / Flask 서버 부분이 추가되어 있습니다. 
> 튜닝된 체크포인트 및 어댑터는 [허깅페이스](https://huggingface.co/EthanYJ/Lumiply/tree/main)에서 다운 받으실 수 있습니다.

---

### 3. 모델 / 체크포인트 준비

이 레포는 LumiNet 원본 모델을 그대로 사용하되, Lumiply 환경에서 다음과 같은 체크포인트 구성을 가정합니다.

- `ckpt/trained_crossattn.ckpt`
  - 프로젝트에서 fine‑tune 된 cross‑attention 기반 LumiNet checkpoint
- `ckpt/new_decoder.ckpt`
  - bypass decoder (identity preservation 향상용)
- `adaptors/adaptor_{color}.pth`
  - 색상별 adaptor 가중치  
    (`white`, `red`, `orange`, `yellow`, `green`, `blue`, `purple`)

`lumiply_inference.ipynb` 상단의 핵심 설정은 대략 다음과 같은 형태입니다.

```python
BASE_MODEL_PATH = "./ckpt/trained_crossattn.ckpt"
CONFIG_PATH    = "./models/cldm_v21_LumiNet.yaml"

def initialize_engine():
    ...
    model = instantiate_from_config(config.model).cpu()
    model.add_new_layers()
    model.load_state_dict(load_state_dict(BASE_MODEL_PATH, location="cpu"), strict=False)

    new_decoder = True
    if new_decoder:
        ae_checkpoint = "./ckpt/new_decoder.ckpt"
        model.change_first_stage(ae_checkpoint)
    ...
```

각 색상에 대해 `switch_adapter(color)` 가 `adaptors/adaptor_{color}.pth`를 hot‑swap 하는 구조입니다.  
결과적으로 **하나의 베이스 모델 + 색상별 adapter** 조합으로 7가지 조명을 생성합니다.

---

### 4. 의존성 설치 (Colab 기준)

Colab 런타임에서는 아래 순서대로 실행하면 됩니다.

1. **작업 디렉터리 이동**

   ```python
   %cd /content/drive/MyDrive/ku-ai/lumiply-colab
   ```

2. **필요 패키지 설치**

   ```bash
   pip install -r requirements.txt
   ```

3. (옵션) Hugging Face 로그인이 필요한 경우

   ```python
   from huggingface_hub import login
   login()  # HF 토큰 입력
   ```

환경을 한 번 맞춰 두면, 이후에는 Colab 런타임 재시작 시 위 과정을 다시 수행하면 됩니다.

---

### 5. Flask + ngrok 서버 구조

`lumiply_inference.ipynb` 의 주요 셀은 크게 세 부분으로 나뉩니다.

#### 5‑1. 모델/엔진 초기화

- `initialize_engine()`
  - LumiNet base 모델과 bypass decoder를 메모리에 올립니다.
  - 전역 변수 `global_model`, `global_sampler` 에 보관해 요청마다 재사용합니다.
- `run_inference_single_image(off_path, color, ...)`
  - 입력 `off.png` 를 기준으로 512×512 해상도의 입력을 생성합니다.
  - `hint = concat(off_resized, white_ref)` 형태로 control 신호를 구성합니다.
  - DDIM Sampler로 latent 샘플링 후,
  - new decoder + identity feature(`ae_hs`)를 이용해 원본 해상도로 디코딩합니다.
  - `output_{color}.jpg` 로 저장하고, 해당 경로를 반환합니다.

#### 5‑2. Flask 엔드포인트

```python
app = Flask(__name__)
CORS(app)

INFERENCE_ROOT = "./images/inference"

@app.route("/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "message": "Colab 서버가 실행 중입니다.",
        "timestamp": datetime.now().isoformat(),
    }), 200

@app.route("/process", methods=["POST"])
def process_image():
    # 1) multipart/form-data 에서 image, job_id, color 읽기
    # 2) /images/inference/{job_id}/off.png 로 저장
    # 3) run_inference_single_image(off_path, color) 호출
    # 4) output_{color}.jpg 의 public URL 을 JSON 으로 응답
```

요청 포맷 (FastAPI → Colab):

- **`POST /process`**
  - `files["image"]`: 합성된 off 이미지 (`image/png`)
  - `form["job_id"]`: FastAPI 에서 생성한 job ID
  - `form["color"]`: `"white" | "red" | ... | "purple"`
  - `form["callback_url"]`: 현재는 사용하지 않지만, 향후 push 방식 연동을 위한 예약 필드

응답 예시:

```json
{
  "job_id": "c1f0dffe-...",
  "status": "completed",
  "result": {
    "images": {
      "red": "https://<ngrok-domain>/static/inference/<job_id>/output_red.jpg"
    },
    "input_image_url": "https://<ngrok-domain>/static/inference/<job_id>/off.png"
  },
  "message": "색상 이미지 생성 완료"
}
```

> FastAPI 쪽 `send_to_colab` 이 색상별로 `/process` 를 여러 번 호출하고,  
> 반환된 URL 들을 모아서 최종적으로 7색 결과 JSON을 구성합니다.

#### 5‑3. ngrok + 백그라운드 서버

- `SharedDataMiddleware` 로 `/static/inference` 경로에 `INFERENCE_ROOT`를 마운트합니다.
- `FlaskServerThread` 로 Flask 앱을 백그라운드 스레드에서 실행합니다.
- `ngrok.set_auth_token(...)` 후 `ngrok.connect(5000)` 으로 public URL 을 획득합니다.
- 성공 시 대략 다음과 같은 로그가 출력됩니다.

```text
✅ Colab 서버가 시작되었습니다!
🌐 Public URL: https://undented-....ngrok-free.dev
📝 FastAPI의 COLAB_WEBHOOK_URL 환경 변수에 다음 URL을 설정하세요:
   https://undented-....ngrok-free.dev/process
```

여기서 `/process` 가 붙은 URL 전체를 `lumiply-server/.env` 의 `COLAB_WEBHOOK_URL` 로 사용합니다.

---

### 6. Lumiply 연동 절차 요약

Lumiply 전체 플로우를 정리하면 다음과 같습니다.

1. **Colab 런타임 준비**
   - 이 레포를 마운트하고, `lumiply_inference.ipynb` 셀을 위에서 아래로 순서대로 실행합니다.
   - 마지막에 출력되는 `/process` URL 을 복사합니다.
2. **FastAPI 서버 설정 (`lumiply-server`)**
   - `.env` 의 `COLAB_WEBHOOK_URL` 에 위 URL 을 설정합니다.
   - `uvicorn main:app --reload` 로 서버를 띄웁니다.
3. **프론트엔드 실행 (`lumiply-client`)**
   - `npm start` 후 `http://localhost:3000` 접속
   - 방 사진 업로드 → 조명 배치 → “적용” 클릭 → 결과 비교/저장까지 end‑to‑end 로 확인 가능합니다.

위 흐름을 맞춰 두면 Colab 세션이 끊어졌을 때도  
“Colab 재시작 → `/process` URL 갱신 → FastAPI `.env` 변경” 순서로 쉽게 복구할 수 있습니다.

---

### 7. 로컬 테스트 (FastAPI 없이)

FastAPI/프론트엔드 없이 Colab/로컬에서 LumiNet 결과만 빠르게 보고 싶다면 다음 스크립트를 사용할 수 있습니다.

- **`relit_inference.py`**

```bash
python relit_inference.py
```

- 코드 상단에서 입력 경로, 참조 경로, DDIM step 등을 직접 지정하고 실행하면,  
  로컬 폴더에 `output_*.png` 결과가 생성됩니다.

또는, LumiNet 자체를 체험해 보고 싶을 경우 `gradio_demo.py` 를 실행해 간단한 웹 데모 UI를 띄울 수 있습니다.

```bash
python gradio_demo.py
```

---

### 8. 라이선스 및 원저작자 크레딧

이 레포는 [LumiNet 논문](https://arxiv.org/abs/2412.00177), [LumiNet Github](https://github.com/xyxingx/LumiNet/) 공식 코드와 모델을 기반으로 하며,  
원저작자의 라이선스를 그대로 따릅니다.

LumiNet / Latent‑Intrinsics 관련 연구 결과를 사용하거나 인용하는 경우,  
아래와 같이 원 논문을 함께 인용해 주시는 것이 적절합니다.

- LumiNet: *“LumiNet: Latent Intrinsics Meets Diffusion Models for Indoor Scene Relighting”* (CVPR 2025)
- Latent‑Intrinsics: *“Latent Intrinsics Emerge from Training to Relight”* (NeurIPS 2024)

> 이 README는 “Lumiply 프로젝트에서 이 Colab 레포를 어떤 방식으로 사용하는지”를 설명하기 위한 문서입니다.  
> LumiNet 자체의 학술적 설명, 공식 인용문은 원 LumiNet 레포와 논문을 참고해 주시기를 바랍니다.