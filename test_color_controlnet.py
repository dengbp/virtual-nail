from diffusers import StableDiffusionControlNetPipeline, ControlNetModel
from PIL import Image
import numpy as np
import cv2

controlnet = ControlNetModel.from_pretrained("lllyasviel/sd-controlnet-canny", torch_dtype="auto")
pipe = StableDiffusionControlNetPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    controlnet=controlnet,
    torch_dtype="auto"
).to("cuda")

init_img = Image.open("data/test_images/11111.jpg").convert("RGB")
img_np = np.array(init_img)
canny = cv2.Canny(img_np, 100, 200)
canny_img = Image.fromarray(canny)

result = pipe(
    prompt="fingernails with glossy nail polish, realistic, high detail",
    image=init_img,
    control_image=canny_img,
    guidance_scale=7.5,
    num_inference_steps=30
).images[0]
result.save("test_result.png") 