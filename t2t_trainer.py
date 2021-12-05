"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_trainer
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer

import problems
import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('extra_tokens', 32 , 'extra tokens to be added into encoding vectors')



@registry.register_hparams
def transformer_extra_tokens():
  hparams = transformer.transformer_base()
  hparams.add_hparam("extra_tokens", FLAGS.extra_tokens)

  return hparams

@registry.register_hparams
def transformer_tall9_extra_tokens():
  hparams = transformer.transformer_big()
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 9
  hparams.num_heads = 12
  hparams.add_hparam("extra_tokens", FLAGS.extra_tokens)
  return hparams


@registry.register_hparams
def transformer_tall_18_18():
  hparams = transformer.transformer_tall9()
  hparams.num_encoder_layers = 18
  hparams.num_decoder_layers = 18
  return hparams

@registry.register_hparams
def transformer_tall_12_24():
  hparams = transformer.transformer_tall9()
  hparams.num_encoder_layers = 12
  hparams.num_decoder_layers = 24
  return hparams

@registry.register_hparams
def transformer_tall_24_12():
  hparams = transformer.transformer_tall9()
  hparams.num_encoder_layers = 24
  hparams.num_decoder_layers = 12
  return hparams


if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(t2t_trainer.main)
