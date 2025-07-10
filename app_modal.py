
import modal
app = modal.App("sd-fastapi")

image = (
    modal.Image.debian_slim()
    .pip_install(
        "fastapi",
        "uvicorn",
        "diffusers==0.25.0",
        "transformers",
        "accelerate",
        "safetensors",
        "huggingface_hub==0.25.2",
        "torch",
        "pillow"
    )
)

volume = modal.Volume.from_name("sd-cache", create_if_missing=True)
MODEL_DIR = "/model"

with image.imports():
    import torch
    from diffusers import StableDiffusionPipeline
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from pydantic import BaseModel
    import io
    import os
    import tempfile

@app.cls(
    gpu="A10G",  
    image=image,
    volumes={MODEL_DIR: volume},
    scaledown_window=300,  
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class StableDiffusionModel:
    @modal.enter() 
    def setup(self):
        torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

        self.pipe = StableDiffusionPipeline.from_pretrained(
            "runwayml/stable-diffusion-v1-5",
            torch_dtype=torch_dtype,
            cache_dir=MODEL_DIR,
            use_auth_token=os.environ.get("HF_TOKEN"),
            safety_checker=None,  
            requires_safety_checker=False
        )
        
        if torch.cuda.is_available():
            self.pipe = self.pipe.to("cuda")
        
        self.pipe.enable_attention_slicing()
        
        try:
            self.pipe.enable_xformers_memory_efficient_attention()
        except Exception as e:
            print(f"Could not enable xformers: {e}")
        
        print("Pipeline loaded and optimized")

    @modal.method()
    def generate(self, prompt: str, num_inference_steps: int = 20, guidance_scale: float = 7.5) -> bytes:
        generator = None
        if torch.cuda.is_available():
            generator = torch.Generator(device="cuda").manual_seed(42)
        
        result = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            generator=generator
        )
        
        image = result.images[0]

        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

@app.function(image=image)
@modal.asgi_app()
def fastapi_app():
    web_app = FastAPI(title="Stable Diffusion API", version="1.0.0")
    
    model = StableDiffusionModel()

    class Txt2ImgRequest(BaseModel):
        prompt: str
        num_inference_steps: int = 20
        guidance_scale: float = 7.5

    @web_app.get("/")
    async def root():
        return {"message": "Stable Diffusion API is running"}

    @web_app.get("/health")
    async def health():
        return {"status": "healthy"}

    @web_app.post("/txt2img/")
    async def txt2img(request: Txt2ImgRequest):
        print(f"Received prompt: {request.prompt}")
        print(f"Steps: {request.num_inference_steps}, Guidance: {request.guidance_scale}")
        
        try:
            image_bytes = model.generate.remote(
                request.prompt,
                request.num_inference_steps,
                request.guidance_scale,
            )

            with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp_file:
                tmp_file.write(image_bytes)
                tmp_path = tmp_file.name

            print(f"Generated image saved to {tmp_path}")
            
            return FileResponse(
                tmp_path,
                media_type="image/png",
                filename=f"generated_image.png",
                headers={"Content-Disposition": "attachment; filename=generated_image.png"}
            )
            
        except Exception as e:
            print(f"Error generating image: {e}")
            return {"error": str(e)}, 500

    return web_app

