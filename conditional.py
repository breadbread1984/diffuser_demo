#!/usr/bin/python3

from absl import flags, app
import torch
from transformers import CLIPTextModel, CLIPTokenizer
from diffusers import AutoencoderKL, UNer2DConditionalModel, PNDMScheduler
import cv2

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_string('text', default = 'a photograph of an astronaut riding a horse', help = 'prompt for generation')
  flags.DEFINE_string('output', default = 'output.png', help = 'path to output image')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cpu', 'cuda'}, help = 'device to use')

def main(unused_argv):
  vae = AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae", use_safetensors=True)
  tokenizer = CLIPTokenizer.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="tokenizer")
  text_encoder = CLIPTextModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="text_encoder", use_safetensors=True)
  unet = UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet", use_safetensors=True)
  scheduler = PNDMScheduler.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="scheduler")
  vae.to(FLAGS.device)
  text_encoder.to(FLAGS.device)
  unet.to(FLAGS.device)
  text_input = tokenizer(FLAGS.text, return_tensors = "pt").to(FLAGS.device)
  text_embeddings = text_encoder(**text_input).last_hidden_state
  latents = torch.randn((1, unet.in_channels, 64, 64), device=FLAGS.device)
  scheduler.set_timesteps(50)
  for t in scheduler.timesteps:
    # Predict noise
    with torch.no_grad():
      noise_pred = unet(latents, t, encoder_hidden_states=text_embeddings).sample
      # Compute the previous noisy sample x_t -> x_t-1
      latents = scheduler.step(noise_pred, t, latents).prev_sample
  with torch.no_grad():
    image = vae.decode(latents).sample
  image = (image / 2 + 0.5).clamp(0, 1)
  image = image.cpu().permute(0, 2, 3, 1).numpy()
  image = (image * 255).round().astype("uint8")[0]
  image = Image.fromarray(image)
  image.save('output.png')

if __name__ == "__main__":
  add_option()
  app.run(main)
