#!/usr/bin/python3

from absl import flags, app
from diffusers import DDPMPipeline

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('model', enum_values = {'cifar10', 'celeba', 'bedroom', 'cat', 'church'}, default = 'celeba', help = 'model to use')
  flags.DEFINE_string('output', default = 'output.png', help = 'path to output image')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  models = {'cifar10': 'google/ddpm-cifar10-32',
            'celeba': 'google/ddpm-celebahq-256',
            'cat': 'google/ddpm-cat-256',
            'church': 'google/ddpm-church-256',
            'bedroom': 'google/ddpm-bedroom-256'}
  ddpm = DDPMPipeline.from_pretrained(models[FLAGS.model]).to(FLAGS.device)
  image = ddpm(num_inference_steps = 25).images[0]
  image.save(FLAGS.output)

if __name__ == "__main__":
  add_options()
  app.run(main)

