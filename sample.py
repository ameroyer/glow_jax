from jax.random import PRNGKey, normal
import jax.numpy as jnp
import flax

from model import GLOW

import argparse
from utils import plot_image_grid
from functools import partial


def postprocess(x, num_bits):
    """Map [-0.5, 0.5] quantized images to uint space"""
    num_bins = 2 ** num_bits
    x = jnp.floor((x + 0.5) * num_bins)
    x *= 256. / num_bins
    return jnp.clip(x, 0, 255).astype(jnp.uint8)

def sample(model, 
           params,  
           shape=None, 
           sampling_temperature=1.0, 
           key=PRNGKey(0),
           postprocess_fn=None, 
           save_path=None,
           display=True):
    """Sampling only requires a call to the reverse pass of the model"""
    zL = normal(key, shape)
    y, *_ = model.apply(params, zL, sampling_temperature=sampling_temperature, reverse=True)
    if postprocess_fn is not None:
        y = postprocess_fn(y)
    plot_image_grid(y, save_path=save_path, display=display,
                    title=None if save_path is None else save_path.rsplit('.', 1)[0].rsplit('/', 1)[-1])
    return y


parser = argparse.ArgumentParser(description='Sample from pretrained model.')
parser.add_argument('num_samples', type=int, help='number of samples')
parser.add_argument('-t', '--temperature', default=0.7, type=float, help='Temperature')
parser.add_argument('-s', '--seed', type=int, default=0, help='random seed')
parser.add_argument('--model_path', type=str, default="pretrained/model_epoch=013.weights", help='Model path')
args = parser.parse_args()


model = GLOW(K=16, L=3, nn_width=512, learn_top_prior=True)

with open(args.model_path, 'rb') as f:
    params = model.init(PRNGKey(args.seed), jnp.zeros((
        args.num_samples, 64, 64, 3)))
    params = flax.serialization.from_bytes(params, f.read())
    
sample(model, params, shape=(args.num_samples, 8, 8, 3 * 16),
       key=PRNGKey(args.seed), 
       sampling_temperature=args.temperature,
       postprocess_fn=partial(postprocess, num_bits=5),
       save_path=f"sample_seed={args.seed}_t={args.temperature}.png")