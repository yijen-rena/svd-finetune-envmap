import os
os.environ["HF_HOME"] = "/ocean/projects/cis250002p/rju/huggingface/hub/"

import glob

import torch
from diffusers import UNetSpatioTemporalConditionModel, StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_video, export_to_gif

from PIL import Image


def resize_image(image, size=(1024, 576)):
    resized_img = image.resize(size, Image.Resampling.LANCZOS)
    return resized_img


# unet = UNetSpatioTemporalConditionModel.from_pretrained(
#     "/path/to/unet",
#     subfolder="unet",
#     torch_dtype=torch.float16,
#     low_cpu_mem_usage=False,
# )

pipe = StableVideoDiffusionPipeline.from_pretrained(
    "stabilityai/stable-video-diffusion-img2vid",
    # unet=unet,
    low_cpu_mem_usage=False,
    torch_dtype=torch.float16,
    variant="fp16",
    local_files_only=False,
)
pipe.to("cuda:0")

input_dir = 'wan2-outputs'
output_dir = 'outputs/wan2-imgs'
img_paths = glob.glob(f'{input_dir}/**/*.png', recursive=True)

for img_path in img_paths:
    # image = load_image(img_path)
    output_path = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(img_path))[0]}.mp4")
    if os.path.exists(output_path):
        print(f"Skipping {img_path} because {output_path} exists")
        continue
    
    image = Image.open(img_path)
    image = resize_image(image)
    # image = image.resize((1024, 576))

    generator = torch.manual_seed(-1)
    with torch.inference_mode():
        frames = pipe(image,
                    num_frames=25,
                    width=1024,
                    height=576,
                    decode_chunk_size=8, generator=generator, motion_bucket_id=127, fps=8, num_inference_steps=30).frames[0]
    export_to_video(frames, output_path, fps=5)

