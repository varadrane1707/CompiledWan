import torch
import torch.distributed as dist

import gc
import time
import logging
from PIL import Image
import argparse
from datetime import datetime
import GPUtil
import json
import os

from para_attn.context_parallel import init_context_parallel_mesh
from para_attn.context_parallel.diffusers_adapters import parallelize_pipe
from para_attn.parallel_vae.diffusers_adapters import parallelize_vae

# from diffusers import AutoencoderKLWan, WanImageToVideoPipeline,WanTransformer3DModel
from diffusers import AutoencoderKLWan
from pipeline_wan import WanImageToVideoPipeline
from transformer_wan import WanTransformer3DModel

from diffusers.utils import export_to_video, load_image
from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
from transformers import CLIPVisionModel, UMT5EncoderModel
from diffusers.video_processor import VideoProcessor

logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# from diffusers import attention_backend

class CompilationConfig():
    cache_type: str = "none"  # Disable caching for temporal coherence test
    cache_threshold: float = 0.05
    compilation : bool = False
    model_id : str = "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers"
    warmup : bool = True
    quantization : bool = False
    use_sage_attention : bool = False
    use_context_parallel : bool = False  # Disable context parallel for temporal coherence test

class CompiledWanModel():
    
    def __init__(self,config: CompilationConfig):
        print(f"Initializing CompiledWanModel pipeline with model")
        self.use_context_parallel = config.use_context_parallel
        if self.use_context_parallel:
            dist.init_process_group(backend='nccl', init_method='env://')
            rank = dist.get_rank()
            print(f"Rank: {rank}")
            torch.cuda.set_device(rank)
        start_load_time = time.time()
        
        self.model_id = config.model_id
        self.cache_type = config.cache_type
        self.cache_threshold = config.cache_threshold
        self.compilation = config.compilation
        self.warmup = config.warmup
        self.use_sage_attention = config.use_sage_attention
        self.pipe = None
        self.quantization = config.quantization
        
        if self.model_id == "Wan-AI/Wan2.1-I2V-14B-480P-Diffusers":
            self.flow_shift = 3.0
        elif self.model_id == "Wan-AI/Wan2.1-I2V-14B-720P-Diffusers":
            self.flow_shift = 5.0
        else:
            self.flow_shift = 3.0
        
        try:
            self.load_model()
            self.optimize_pipe()
            print(f"Pipeline initialized {self.model_id} in {time.time() - start_load_time} seconds")
            if self.warmup:
                self.warmup_step()
        except Exception as e:
            print(f"Error loading model: {e}")
            raise e
                
    def max_memory_usage(self):
        """Get maximum GPU memory usage in MB"""
        gpu = GPUtil.getGPUs()[0]
        return gpu.memoryTotal
    
    def get_gpu_memory_usage(self):
        """Get current GPU memory usage in MB"""
        gpu = GPUtil.getGPUs()[0]
        return gpu.memoryUsed
    
    def log_gpu_memory_usage(self, message=""):
        """Log current GPU memory usage"""
        memory_used = self.get_gpu_memory_usage()
        memory_total = self.max_memory_usage()
        memory_percent = (memory_used / memory_total) * 100
        print(f"GPU Memory Usage {message}: {memory_used:.2f}MB / {memory_total:.2f}MB ({memory_percent:.2f}%)")
        return memory_used
                
    def load_model(self):
        if self.pipe is None:
            self.text_encoder = UMT5EncoderModel.from_pretrained(self.model_id, subfolder="text_encoder", torch_dtype=torch.bfloat16)
            self.vae = AutoencoderKLWan.from_pretrained(self.model_id, subfolder="vae", torch_dtype=torch.float32)
            self.image_encoder = CLIPVisionModel.from_pretrained(self.model_id, subfolder="image_encoder", torch_dtype=torch.float32)
            self.transformer = WanTransformer3DModel.from_pretrained(self.model_id, subfolder="transformer", torch_dtype=torch.bfloat16)
            # self.transformer.enable_layerwise_casting(storage_dtype=torch.float8_e4m3fn, compute_dtype=torch.bfloat16)
            # self.vae_scale_factor_temporal = 2 ** sum(self.vae.temperal_downsample) if getattr(self, "vae", None) else 4
            # self.vae_scale_factor_spatial = 2 ** len(self.vae.temperal_downsample) if getattr(self, "vae", None) else 8
            # self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)
            
            self.pipe = WanImageToVideoPipeline.from_pretrained(
                self.model_id,
                vae=self.vae,
                transformer=self.transformer,
                text_encoder=self.text_encoder,
                image_encoder=self.image_encoder,
                torch_dtype=torch.bfloat16
            )
            self.pipe.scheduler = UniPCMultistepScheduler.from_config(self.pipe.scheduler.config, flow_shift=self.flow_shift)
            self.pipe.to("cuda")
            
            self.pipe
            
    def apply_cache(self):
        if self.cache_type == "fbcache":
            from para_attn.first_block_cache.diffusers_adapters import apply_cache_on_pipe
            apply_cache_on_pipe(self.pipe , residual_diff_threshold=self.cache_threshold)
            print(f"Cache applied with fbcache and residual_diff_threshold = {self.cache_threshold}")
        elif self.cache_type == "teacache":
            from cachify import teacache_forward
            import types
            # Properly bind the teacache_forward as a method
            self.pipe.transformer.forward = types.MethodType(teacache_forward, self.pipe.transformer)
            self.pipe.transformer.__class__.enable_teacache = True
            self.pipe.transformer.__class__.cnt = 0
            self.pipe.transformer.__class__.num_steps = 30
            self.pipe.transformer.__class__.rel_l1_thresh = self.cache_threshold 
            self.pipe.transformer.__class__.accumulated_rel_l1_distance = 0
            self.pipe.transformer.__class__.previous_modulated_input = None
            self.pipe.transformer.__class__.previous_residual = None
            return
        else:
            raise ValueError(f"Invalid cache type: {self.cache_type}")
            
    def apply_sage_attention(self):
        if self.use_sage_attention:
            print("Applying Sage Attention!")
            from sageattention import sageattn
            from modile_model_sage import set_sage_attn_wan
            with torch.autocast("cuda", torch.bfloat16, cache_enabled=False):
                set_sage_attn_wan(self.pipe.transformer, sageattn) 
    def apply_quantization(self):
        if self.quantization:
            print("Applying Quantization!")
            from torchao.quantization import quantize_, float8_dynamic_activation_float8_weight, float8_weight_only
            quantize_(self.pipe.text_encoder, float8_weight_only())
            quantize_(self.pipe.transformer, float8_dynamic_activation_float8_weight())
        
    def optimize_pipe(self):
        if self.use_context_parallel:
            parallelize_vae(self.vae)
            parallelize_pipe( 
                self.pipe,
                mesh=init_context_parallel_mesh(
                    self.pipe.device.type,
                ),
            )
        self.apply_quantization()
        self.apply_sage_attention()
        self.apply_cache()
        
        # if self.compilation:
        #     torch._inductor.config.reorder_for_compute_comm_overlap = True
        #     self.pipe.transformer = torch.compile(self.pipe.transformer, mode="max-autotune-no-cudagraphs")
    
    def warmup_step(self):
        print("Running Warm Up!")
        prompt = "A car driving on a road"
        negative_prompt = "blurry, low quality, dark"
        image = load_image("https://storage.googleapis.com/falserverless/gallery/car_720p.png")
        start_time = time.time()
        self.log_gpu_memory_usage("before warmup")
        with torch.inference_mode():
            self.pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image=image,
            num_inference_steps=30,
            height=576,
            width=1024,
            num_frames=81,
                guidance_scale=5.0
            )
        self.get_matrix(start_time,time.time(),576,1024)
        print("Warm Up Completed!")
    
    def get_matrix(self,start_time : int,end_time : int,height : int,width : int):
        print("-"*40)
        print(f"Order_ID : {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print(f"inference time : {end_time - start_time}")
        print(f"height : {height}")
        print(f"width : {width}")
        print("-"*40)
        print("\n")
        
    def generate_video(self,prompt : str,negative_prompt : str,image : Image.Image,num_frames : int = 81,guidance_scale : float = 5.0,num_inference_steps : int = 30,height : int = 576,width : int = 1024,fps : int = 16):
        with torch.inference_mode():
           output = self.pipe(
            image=image,
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            output_type="pil",
            ).frames[0]
        
        import uuid
        if self.use_context_parallel:
            if dist.get_rank() == 0:
               
              export_to_video(output, f"outputs/compiled_{uuid.uuid4()}.mp4", fps=fps)
              return output
        else:
            export_to_video(output, f"outputs/single_compiled_{uuid.uuid4()}.mp4", fps=fps)
            return output
        
    def shutdown(self):
        if self.use_context_parallel:
            dist.destroy_process_group()
            print("Pipeline shutdown completed")
        
        
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    # parser.add_argument("--model_id", type=str, default="Wan-AI/Wan2.1-I2V-14B-720P-Diffusers")
    # parser.add_argument("--attention_type", type=str, default="sage")
    # parser.add_argument("--cache_type", type=str, default="teacache")
    # parser.add_argument("--cache_threshold", type=float, default=0.1)
    # parser.add_argument("--compilation", type=bool, default=False)
    # parser.add_argument("--use_sage_attention", type=bool, default=True)
    # parser.add_argument("--quantization", type=bool, default=False)
    
    os.environ["MASTER_ADDR"] = "localhost"  # or the IP of the master node
    os.environ["MASTER_PORT"] = "29500"      # any free port                # rank of this process
    os.environ["WORLD_SIZE"] = str(8)
    
    config = CompilationConfig()
    model = CompiledWanModel(config)
    
    with open("inputs.json", "r") as f:
            inputs = json.load(f)
            
    for i in range(0,len(inputs)):
        prompt = inputs[str(i+1)]["prompt"]
        negative_prompt = inputs[str(i+1)]["negative_prompt"]
        image = load_image(inputs[str(i+1)]["image"])
        image = image.resize((1280,720))
        start_time = time.time()
        latents = model.generate_video(prompt=prompt,negative_prompt=negative_prompt,image=image,height=720,width=1280,num_frames=81,guidance_scale=5.0,num_inference_steps=30,fps=16)
        end_time = time.time()
        model.get_matrix(start_time,end_time,1280,720)
    
    model.shutdown()