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
  text_input = tokenizer(FLAGS.text, padding = 'max_length', max_length = tokenizer.model_max_length, truncation = True, return_tensors = "pt")
  with torch.no_grad():
    text_embeddings = text_encoder(text_input.input_ids.to(FLAGS.device))[0]
  
