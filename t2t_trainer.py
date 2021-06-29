"""Train and evaluate."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


from tensor2tensor.bin import t2t_trainer
from tensor2tensor.models import transformer
from tensor2tensor.utils import registry
from tensor2tensor.models import transformer
from tensor2tensor.layers import modalities

import problems
import tensorflow as tf
from tensor2tensor.utils.t2t_model import _create_target_modality, log_warn, log_info

import collections
import six

flags = tf.flags
FLAGS = flags.FLAGS
flags.DEFINE_integer('extra_tokens', 32 , 'extra tokens to be added into encoding vectors')



@registry.register_hparams
def transformer_tall9():
  hparams = transformer.transformer_big()
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 9
  hparams.num_heads = 12
  return hparams
  

@registry.register_hparams
def transformer_tall9_extra_tokens():
  hparams = transformer.transformer_big()
  hparams.hidden_size = 768
  hparams.filter_size = 3072
  hparams.num_hidden_layers = 9
  hparams.num_heads = 12
  hparams.extra_tokens = FLAGS.extra_tokens
  return hparams

@registry.register_model
class Transformerextratokens(transformer.Transformer):
  """Attention net.  See file docstring."""

  def __init__(self, *args, **kwargs):
    super(Transformerextratokens, self).__init__(*args, **kwargs)

  def bottom(self, features):
    """Transforms features to feed into body.

    Args:
      features: dict of str to Tensor. Typically it is the preprocessed data
        batch after Problem's preprocess_example().

    Returns:
      transformed_features: dict of same key-value pairs as features. The value
        Tensors are newly transformed.
    """
    if not self._problem_hparams:
      log_warn("Without a Problem, T2TModel.bottom is a passthrough.")
      return features

    transformed_features = collections.OrderedDict()
    all_previous_modalities = []
    target_modality = _create_target_modality(self._problem_hparams.modality)

    # Transform features via its corresponding modality.
    for feature_name, modality in sorted(
        six.iteritems(self._problem_hparams.modality)):
      if feature_name not in features:
        tf.logging.warning("Missing feature %s - ignoring." % feature_name)
        continue
      vocab_size = self._problem_hparams.vocab_size[feature_name]
      if vocab_size is not None and hasattr(self._hparams, "vocab_divisor"):
        vocab_size += (-vocab_size) % self._hparams.vocab_divisor
      modality_name = self._hparams.name.get(
          feature_name,
          modalities.get_name(modality))(self._hparams, vocab_size)
      # Use if-else clauses to preserve behavior of previous changes: namely,
      # the variable scope name for the targets feature if there is only one
      # target modality; and to reuse variable scopes for only input modalities.
      if feature_name in target_modality:
        if len(target_modality) > 1:
          variable_scope_name = "%s/%s" % (modality_name, feature_name)
        else:
          variable_scope_name = modality_name
        bottom = self._hparams.bottom.get(
            feature_name,
            modalities.get_targets_bottom(modality))
        # TODO(aidangomez): share variables?
        with tf.variable_scope(variable_scope_name) as vs:
          self._add_variable_scope(variable_scope_name, vs)
          log_info("Transforming feature '%s' with %s.targets_bottom",
                   feature_name,
                   modality_name)
          transformed_features[feature_name] = bottom(features[feature_name],
                                                      self._hparams,
                                                      vocab_size)
      else:
        bottom = self._hparams.bottom.get(feature_name,
                                          modalities.get_bottom(modality))
        do_reuse = modality_name in all_previous_modalities
        with tf.variable_scope(modality_name, reuse=do_reuse) as vs:
          self._add_variable_scope(modality_name, vs)
          log_info("Transforming feature '%s' with %s.bottom",
                   feature_name,
                   modality_name)
          transformed_features[feature_name] = bottom(features[feature_name],
                                                      self._hparams,
                                                      vocab_size)
        all_previous_modalities.append(modality_name)

    inputs_tensor = features['inputs']
    inputs_shape = inputs_tensor.shape
    batch_size = inputs_shape[0]


    num_special_tokens = batch_size
    special_token_id = 21223

    special_tokens = tf.constant([[ [[21223+x]] for x in range(0, num_special_tokens)]])
    # special_tokens = tf.reshape(special_tokens, [batch_size, num_special_tokens, 1, 1])
    # special_tokens = tf.repeat(special_tokens, batch_size, axis=1)
    special_tokens = tf.repeat(special_tokens, batch_size, axis=0)
    
    print(special_tokens.shape)
    inputs_tensor = tf.concat([special_tokens, inputs_tensor], 1)
    features['inputs'] = inputs_tensor
    
    print("features", features) 
    

    for key in features:
      if key not in transformed_features:
        # For features without a modality, we pass them along as is
        transformed_features[key] = features[key]
      else:
        # Other features get passed along with the "raw" suffix
        transformed_features[key + "_raw"] = features[key]

    transformed_features['inputs'] = tf.cast(inputs_tensor, dtype=tf.float32)

    print("transformed_features", transformed_features)
    return transformed_features

if __name__ == '__main__':
  tf.logging.set_verbosity(tf.logging.INFO)
  tf.app.run(t2t_trainer.main)
