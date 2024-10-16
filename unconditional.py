#!/usr/bin/python3

from absl import flags, app
from diffusers import DDPMPipeline

FLAGS = flags.FLAGS

def add_options():
  flags.DEFINE_enum('category', enum_values = {'cifar10', 'celeba', 'bedroom', 'cat', 'church'}, default = 'celeba', help = 'pretrained model to use')
  flags.DEFINE_enum('model', enum_values = {'ddpm', 'ddim'}, default = 'ddpm', help = 'which model to use')
  flags.DEFINE_string('output', default = 'output.png', help = 'path to output image')
  flags.DEFINE_enum('device', default = 'cuda', enum_values = {'cuda', 'cpu'}, help = 'device to use')

def main(unused_argv):
  models = {'ddpm':{'cifar10': 'google/ddpm-cifar10-32',
                    'celeba': 'google/ddpm-celebahq-256',
                    'cat': 'google/ddpm-cat-256',
                    'church': 'google/ddpm-church-256',
                    'bedroom': 'google/ddpm-bedroom-256'},
            'ddim':{'celeba': 'fusing/ddim-celeba-hq',
                    'church': 'fusing/ddim-lsun-church',
                    'bedroom': 'fusing/ddim-lsun-bedroom'}}
  ddpm = DDPMPipeline.from_pretrained(models[FLAGS.model][FLAGS.category]).to(FLAGS.device)
  image = ddpm(num_inference_steps = 25).images[0]
  image.save(FLAGS.output)

if __name__ == "__main__":
  add_options()
  app.run(main)

