r"""Decode from trained T2T models.

Mimic t2t-decoder binary.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import operator
import os
import re
import string
import sys
import time

import numpy as np
import six
from tqdm import tqdm

from tensor2tensor.bin import t2t_trainer
from tensor2tensor.data_generators import problem  # pylint: disable=unused-import
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib
from tensor2tensor.utils import usr_dir, registry
import copy
import tensorflow as tf
import lib

flags = tf.flags
FLAGS = flags.FLAGS


def create_hp_and_estimator(
    problem_name, data_dir, checkpoint_path, decode_to_file=None):
  trainer_lib.set_random_seed(FLAGS.random_seed)
  
  hp = trainer_lib.create_hparams(
      FLAGS.hparams_set,
      FLAGS.hparams,
      data_dir=os.path.expanduser(data_dir),
      problem_name=problem_name)
  # hp.model_dir = checkpoint_path
  # hp.output_dir = checkpoint_path
  decode_hp = decoding.decode_hparams(FLAGS.decode_hparams)
  decode_hp.shards = FLAGS.decode_shards
  decode_hp.shard_id = FLAGS.worker_id
  decode_in_memory = FLAGS.decode_in_memory or decode_hp.decode_in_memory
  decode_hp.decode_in_memory = decode_in_memory
  decode_hp.decode_to_file = decode_to_file
  decode_hp.decode_reference = None

  FLAGS.checkpoint_path = checkpoint_path

  config = t2t_trainer.create_run_config(hp)
  hp.add_hparam("model_dir", config.model_dir)
  estimator = create_estimator(
      FLAGS.model,
      hp,
      checkpoint_path)

  # estimator = trainer_lib.create_estimator(
  #     FLAGS.model,
  #     hp,
  #     t2t_trainer.create_run_config(hp),
  #     decode_hparams=decode_hp,
  #     use_tpu=FLAGS.use_tpu)
  return hp, decode_hp, estimator

def create_estimator(model_name, hparams, init_checkpoint):
  """Create a T2T Estimator."""
  model_fn = get_model_fn(model_name, hparams, init_checkpoint)
  run_config = t2t_trainer.create_run_config(hparams)
  if FLAGS.use_tpu:
    estimator = tf.contrib.tpu.TPUEstimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
        use_tpu=FLAGS.use_tpu,
        train_batch_size=32,
        eval_batch_size=32,
        predict_batch_size=32
    )
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        model_dir=run_config.model_dir,
        config=run_config,
    )

  return estimator

global_logits = None


def get_scaffold_fn(ckpt_path):
  variable_map = get_variable_map(ckpt_path)

  def scaffold_fn():
    tf.train.init_from_checkpoint(ckpt_path, variable_map)
    return tf.train.Scaffold()

  return scaffold_fn


def get_variable_map(ckpt_path):
  """Initialize variables from given directory."""

  def _get_body_name(var_name):
    if '/body/' not in var_name:
      return var_name
    body_name = var_name.split('/body/')
    assert len(body_name) == 2, var_name
    return body_name[-1]

  ckpt_vars = {}
  for var_name, shape in tf.train.list_variables(ckpt_path):
    ckpt_vars[_get_body_name(var_name)] = var_name, shape

  variable_map = {}
  fails = []
  for var in tf.contrib.framework.get_trainable_variables():
    model_var_name = var.op.name
    model_var_body_name = _get_body_name(model_var_name)
    if model_var_body_name in ckpt_vars:
      ckpt_var_name, shape = ckpt_vars[model_var_body_name]
      assert var.shape.as_list() == list(shape)
      tf.logging.info('>>> LOAD ckpt var {} to model var {}'.format(
          ckpt_var_name, model_var_name))
      variable_map[ckpt_var_name] = var
    else:
      fails.append(model_var_name)

  for model_var_name in fails:
    tf.logging.info('>>> FAIL to find {} in checkpoint'.format(
        model_var_name))

  if fails:
    raise ValueError('Failed to initialize from checkpoint.')

  return variable_map



def get_model_fn(model_name, hparams, init_checkpoint):
  """Get model fn."""
  model_cls = registry.model(model_name)

  def model_fn(features, labels, mode, params=None, config=None):
    """Model fn."""
    _, _ = params, labels
    hparams_ = copy.deepcopy(hparams)

    # Instantiate model
    data_parallelism = None
    if not FLAGS.use_tpu and config:
      data_parallelism = config.data_parallelism
    reuse = tf.get_variable_scope().reuse
    this_model = model_cls(
        hparams_,
        # Always build model with EVAL mode to turn off all dropouts.
        tf.estimator.ModeKeys.EVAL,
        data_parallelism=data_parallelism,
        decode_hparams=None,
        _reuse=reuse)

    print("features ", features)
    logits, losses_dict = this_model(features)
    global_logits = logits

    # Accumulate losses
    loss_ = sum(losses_dict[key] for key in sorted(losses_dict.keys()))
    
    scaffold_fn = get_scaffold_fn(init_checkpoint)
    
    # scaffold_fn = (this_model.get_scaffold_fn(init_checkpoint)
    #                if FLAGS.load_checkpoint else None)


    print('logits ', logits)

    loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
              from_logits=True, reduction='none')

    mask = tf.math.equal(features['targets'], 0)
    mask = 1.0 - tf.cast(mask, dtype=tf.float32)


    loss = loss_object(features['targets'], logits)
    loss *= mask

    mask = tf.math.reduce_sum(mask, axis=1, keepdims=True)
    loss = tf.math.reduce_sum(loss, axis=1, keepdims=True) / mask

    loss = tf.reshape(loss, [-1, 1])
    # (4,256,1,1)

    
    
    # loss = tf.math.reduce_sum(loss).numpy()
    
    if mode == tf.estimator.ModeKeys.TRAIN:
      # Dummy spec, only for caching checkpoint purpose
      return tf.estimator.EstimatorSpec(
          mode=mode,
          loss=tf.constant(0.0),
          train_op=tf.no_op())

    if FLAGS.use_tpu:
      predict_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=loss_,
          predictions={"logits": logits, "targets": features['targets'], "loss": loss},
          scaffold_fn=scaffold_fn)
      print("predict_spec in model_fn ", predict_spec)
    else:
      scaffold_fn()
      predict_spec = tf.estimator.EstimatorSpec(
          mode=mode,
          predictions=logits)
    return predict_spec

  return model_fn






def backtranslate_interactively(
    from_problem, to_problem,
    from_data_dir, to_data_dir,
    from_ckpt, to_ckpt):
  
  from_hp, from_decode_hp, from_estimator = create_hp_and_estimator(
      from_problem, from_data_dir, from_ckpt)
  
  to_hp, to_decode_hp, to_estimator = create_hp_and_estimator(
      to_problem, to_data_dir, to_ckpt)


  class BackTranslateHelper(object):

    def __init__(self):
      self.current_input = None
      self.translated = {
          'medical': 'y tế',
          'y tế': 'medical',
          'software': 'phần mềm',
          'phần mềm': 'software',
          'law': 'luật',
          'luật': 'law',
          'book': 'sách',
          'sách': 'book',
          'movie': 'phim',
          'phim': 'movie',
          'TED': 'TED'
      }

    def interactive_text_input(self):
      while True:
        if sys.version_info >= (3, 0):
          input_text = input('>>> ')
        else:
          input_text = raw_input('>>> ')
        if input_text == 'q':
          break
        
        if '\\' in input_text and input_text.split('\\')[-1].strip() == 'TED talk':
          input_text = input_text.split(' \\')[0] + ' \\ TED'

        self.current_input = input_text
        yield input_text

    def intermediate_lang_processor(self, intermediate_lang):
      for text in intermediate_lang:
        text = text.replace('&apos;', "'")
        text = text.replace('&quot;', "'")
        text = text.replace('&#91;', "(")
        text = text.replace('&#93;', ")")
                   
        if '\\' in text:
          print('Translated      :', text.split('\\')[0])
        else:
          print('Translated      :', text)
        
        # Fix the input to meet the translation model requirement:
        if '\\' in self.current_input and ' \\ ' not in self.current_input:
          input_text = self.current_input.split('\\')[0].strip()
          tag = self.current_input.split('\\')[1].strip()
          self.current_input =  input_text + ' \\ ' + tag

        if ' \\ ' in self.current_input and ' \\ ' not in text:
          tag = self.current_input.split('\\')[1].strip()
          text += ' \\ ' + self.translated[tag]
        elif '\\' not in self.current_input and '\\' in text:
          text = text.split('\\')[0]
        yield lib.fix_contents(text)


  helper = BackTranslateHelper()

  print('Loading from {} ..'.format(from_ckpt))
  intermediate_lang = decode_interactively(
    from_estimator, helper.interactive_text_input(), 
    from_problem, from_hp, from_decode_hp, from_ckpt)

  print('Loading from {} ..'.format(to_ckpt))
  outputs = decode_interactively(
    to_estimator, helper.intermediate_lang_processor(intermediate_lang), 
    to_problem, to_hp, to_decode_hp, to_ckpt)

  for output in outputs:
    if ' \\' in  output:
        output = output.split(' \\')[0]
    print('Back-translated : {}'.format(
        output.replace('&apos;', "'")
              .replace('&quot;', "'")
              .replace('&#91;', "(")
              .replace('&#93;', ")")))


def decode_interactively(estimator,
                         input_generator,
                         problem_name,
                         hparams,
                         checkpoint_path=None):

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]

  length = getattr(hparams, "length", 0) or hparams.max_length

  def input_fn_gen():
    for line in input_generator:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      else:
        ids = targets_vocab.encode(line)
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      np_ids = np.array(ids, dtype=np.int32)
      yield dict(
          inputs=np_ids.reshape((length, 1, 1))
      )

  def input_fn(params):
    return tf.data.Dataset.from_generator(
      input_fn_gen,
      output_types=dict(
          inputs=tf.int32,
      ),
      output_shapes=dict(
          inputs=(length, 1, 1)
      )
    ).batch(1)

  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  for result in result_iter:
    print(result['logits'])


def evaluate_interactively(estimator,
                        hparams,
                        decode_hp,
                        inputs_file,
                        targets_file,
                        loss_to_file,
                        checkpoint_path=None):
  decode_hp.batch_size = 1
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)
  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]
  problem_name = FLAGS.problem

  tf.logging.info(f"Performing evaluating iteratively for file input_file: {inputs_file} and target_file: {targets_file}")
  
  inputs = []
  targets = []

  for i, t in zip(open(inputs_file, encoding='utf-8'), open(targets_file, encoding='utf-8')):
    inputs.append(i)
    targets.append(t)


  if estimator.config.use_tpu:
    length = getattr(hparams, "length", 0) or hparams.max_length
    batch_ids = []

    # Get ids and input function for prediction
    for line in inputs:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      else:
        ids = targets_vocab.encode(line)
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      batch_ids.append(ids)
    np_ids = np.array(batch_ids, dtype=np.int32)

    # Get ids of output and function for evaluation
    batch_output_ids = []
    for line in targets:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      batch_output_ids.append(ids)
    np_output_ids = np.array(batch_output_ids, dtype=np.int32)

    loss_array = []
    for np_id, np_output_id in tqdm(zip(np_ids, np_output_ids)):
      def eval_input_fn(params):
        batch_size = params["batch_size"]
        dataset = tf.data.Dataset.from_tensor_slices(({"inputs": np.array([np_id], dtype=np.int32), "targets": np.array([np_output_id], dtype=np.int32)}))
        dataset = dataset.map(
          lambda ex: ({"inputs": tf.reshape(ex["inputs"], (length, 1, 1)), "targets": tf.reshape(ex["targets"], (length, 1, 1))}, tf.reshape(ex["targets"], (length, 1, 1))) )
        dataset= dataset.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
        return dataset

      estimator.evaluate(eval_input_fn, steps=1, checkpoint_path=checkpoint_path)

    loss_filename = decoding._add_shard_to_filename(loss_to_file, decode_hp)
    tf.logging.info("Writing decodes into %s" % loss_filename)
    outfile = tf.gfile.Open(loss_filename, "w")

    for l in loss_array:
      outfile.write(f'{l}\n')

    outfile.flush()
    outfile.close()
    


def evaluate_from_file_fn(estimator,
                        hparams,
                        decode_hp,
                        inputs_file,
                        targets_file,
                        loss_to_file,
                        checkpoint_path=None):
  # decode_hp.batch_size = 10
  if not decode_hp.batch_size:
    decode_hp.batch_size = 256
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)
  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]
  problem_name = FLAGS.problem

  filename = decoding._add_shard_to_filename(inputs_file, decode_hp)
  outfilename = decoding._add_shard_to_filename(targets_file, decode_hp)

  tf.logging.info("Performing decoding from file (%s)." % filename)
  inputs = []
  targets = []

  for i, t in zip(open(inputs_file, encoding='utf-8'), open(targets_file, encoding='utf-8')):
    inputs.append(i)
    targets.append(t)


  if estimator.config.use_tpu:
    length = getattr(hparams, "length", 0) or hparams.max_length
    batch_ids = []

    # Get ids and input function for prediction
    for line in inputs:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      else:
        ids = targets_vocab.encode(line)
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      batch_ids.append(ids)
    np_ids = np.array(batch_ids, dtype=np.int32)

    # Get ids of output and function for evaluation
    batch_output_ids = []
    for line in targets:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      batch_output_ids.append(ids)
    np_output_ids = np.array(batch_output_ids, dtype=np.int32)

      
    def eval_input_fn(params):
      print(params)
      dataset = tf.data.Dataset.from_tensor_slices(({"inputs": np_ids, "targets": np_output_ids}))
      # dataset = dataset.map(
      #   lambda ex: ({"inputs": tf.reshape(ex["inputs"], (length, 1, 1)), "targets": tf.reshape(ex["targets"], (length, 1, 1))}, tf.reshape(ex["targets"], (length, 1, 1))) )

      dataset = dataset.map(
        lambda ex: ({"inputs": tf.reshape(ex["inputs"], (length, 1, 1)), "targets": tf.reshape(ex["targets"], (length, 1, 1))}, 
          tf.reshape(ex["targets"], (length, 1, 1))))


      # dataset = dataset.map(
      #   lambda ex: ({"features": tf.reshape(ex["inputs"], (length, 1, 1)), "labels": tf.reshape(ex["targets"], (length, 1, 1))}))
      # dataset = dataset.batch(params['batch_size'])
      dataset= dataset.apply(tf.contrib.data.batch_and_drop_remainder(params['batch_size']))
      return dataset
  # try:
  predict_specs = estimator.predict(eval_input_fn, yield_single_examples=False, checkpoint_path=checkpoint_path)
  
  # print(predict_specs)
  # print(len(predict_specs))

  tf.logging.info("Writing loss into %s" % loss_to_file)
  outfile = tf.gfile.Open(loss_to_file, "w")
  for predict_spec in predict_specs:
  # print(predict_spec['logits'].shape)
  # print(predict_spec['targets'].shape)
    print(predict_spec['loss'].shape)
    for l in predict_spec['loss']:
      outfile.write(f'{l[0]}\n')

  outfile.flush()
  outfile.close()

  
  
  
  # except:
  #   print("estimator evaluate done, global logits ", global_logits)
  # print(loss)


def decode_from_file_fn_(estimator,
                        filename,
                        hparams,
                        decode_hp,
                        decode_to_file=None,
                        checkpoint_path=None):
  """Compute predictions on entries in filename and write them out."""
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]
  problem_name = FLAGS.problem
  filename = decoding._add_shard_to_filename(filename, decode_hp)
  tf.logging.info("Performing decoding from file (%s)." % filename)
  if has_input:
    sorted_inputs, sorted_keys = decoding._get_sorted_inputs(
        filename, decode_hp.delimiter)
  else:
    sorted_inputs = decoding._get_language_modeling_inputs(
        filename, decode_hp.delimiter, repeat=decode_hp.num_decodes)
    sorted_keys = range(len(sorted_inputs))
  num_sentences = len(sorted_inputs)
  num_decode_batches = (num_sentences - 1) // decode_hp.batch_size + 1

  if estimator.config.use_tpu:
    length = getattr(hparams, "length", 0) or hparams.max_length
    batch_ids = []
    for line in sorted_inputs:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      else:
        ids = targets_vocab.encode(line)
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      batch_ids.append(ids)
    np_ids = np.array(batch_ids, dtype=np.int32)
    def input_fn(params):
      batch_size = params["batch_size"]
      dataset = tf.data.Dataset.from_tensor_slices({"inputs": np_ids})
      dataset = dataset.map(
          lambda ex: {"inputs": tf.reshape(ex["inputs"], (length, 1, 1))})
      dataset = dataset.batch(batch_size)
      return dataset
  else:
    def input_fn():
      input_gen = decoding._decode_batch_input_fn(
          num_decode_batches, sorted_inputs,
          inputs_vocab, decode_hp.batch_size,
          decode_hp.max_input_size,
          task_id=decode_hp.multiproblem_task_id, has_input=has_input)
      gen_fn = decoding.make_input_fn_from_generator(input_gen)
      example = gen_fn()
      return decoding._decode_input_tensor_to_features_dict(example, hparams, decode_hp)
  decodes = []
  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  start_time = time.time()
  total_time_per_step = 0
  total_cnt = 0

  def timer(gen):
    while True:
      try:
        start_time = time.time()
        item = next(gen)
        elapsed_time = time.time() - start_time
        yield elapsed_time, item
      except StopIteration:
        break

  for elapsed_time, result in timer(result_iter):
    if decode_hp.return_beams:
      beam_decodes = []
      beam_scores = []
      output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in result:
        if np.isscalar(result["scores"]):
          result["scores"] = result["scores"].reshape(1)
        scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(output_beams):
        tf.logging.info("BEAM %d:" % k)
        score = scores and scores[k]
        _, decoded_outputs, _ = decoding.log_decode_results(
            result["inputs"],
            beam,
            problem_name,
            None,
            inputs_vocab,
            targets_vocab,
            log_results=decode_hp.log_results,
            skip_eos_postprocess=decode_hp.skip_eos_postprocess)
        beam_decodes.append(decoded_outputs)
        if decode_hp.write_beam_scores:
          beam_scores.append(score)
      if decode_hp.write_beam_scores:
        decodes.append("\t".join([
            "\t".join([d, "%.2f" % s])
            for d, s in zip(beam_decodes, beam_scores)
        ]))
      else:
        decodes.append("\t".join(beam_decodes))
    else:
      _, decoded_outputs, _ = decoding.log_decode_results(
          result["inputs"],
          result["outputs"],
          problem_name,
          None,
          inputs_vocab,
          targets_vocab,
          log_results=decode_hp.log_results,
          skip_eos_postprocess=decode_hp.skip_eos_postprocess)
      decodes.append(decoded_outputs)
    total_time_per_step += elapsed_time
    total_cnt += result["outputs"].shape[-1]
  duration = time.time() - start_time
  tf.logging.info("Elapsed Time: %5.5f" % duration)
  tf.logging.info("Averaged Single Token Generation Time: %5.7f "
                  "(time %5.7f count %d)" %
                  (total_time_per_step / total_cnt,
                   total_time_per_step, total_cnt))
  if decode_hp.batch_size == 1:
    tf.logging.info("Inference time %.4f seconds "
                    "(Latency = %.4f ms/setences)" %
                    (duration, 1000.0*duration/num_sentences))
  else:
    tf.logging.info("Inference time %.4f seconds "
                    "(Throughput = %.4f sentences/second)" %
                    (duration, num_sentences/duration))

  # If decode_to_file was provided use it as the output filename without change
  # (except for adding shard_id if using more shards for decoding).
  # Otherwise, use the input filename plus model, hp, problem, beam, alpha.
  decode_filename = decode_to_file if decode_to_file else filename
  if not decode_to_file:
    decode_filename = decoding._decode_filename(decode_filename, problem_name, decode_hp)
  else:
    decode_filename = decoding._add_shard_to_filename(decode_filename, decode_hp)
  tf.logging.info("Writing decodes into %s" % decode_filename)
  outfile = tf.gfile.Open(decode_filename, "w")
  for index in range(len(sorted_inputs)):
    special_chars = ["\a", "\n", "\f", "\r", "\b"]
    output = decodes[sorted_keys[index]]
    for c in special_chars:
      output = output.replace(c, ' ')
    try:
      outfile.write("%s%s" % (output, decode_hp.delimiter))
    except:
      outfile.write("%s" % decode_hp.delimiter)
  outfile.flush()
  outfile.close()
    

def decode_from_file_fn(estimator,
                        filename,
                        hparams,
                        decode_hp,
                        decode_to_file=None,
                        checkpoint_path=None):
  """Compute predictions on entries in filename and write them out."""
  if not decode_hp.batch_size:
    decode_hp.batch_size = 32
    tf.logging.info(
        "decode_hp.batch_size not specified; default=%d" % decode_hp.batch_size)

  # Inputs vocabulary is set to targets if there are no inputs in the problem,
  # e.g., for language models where the inputs are just a prefix of targets.
  p_hp = hparams.problem_hparams
  has_input = "inputs" in p_hp.vocabulary
  inputs_vocab_key = "inputs" if has_input else "targets"
  inputs_vocab = p_hp.vocabulary[inputs_vocab_key]
  targets_vocab = p_hp.vocabulary["targets"]
  problem_name = FLAGS.problem
  filename = decoding._add_shard_to_filename(filename, decode_hp)
  tf.logging.info("Performing decoding from file (%s)." % filename)
  if has_input:
    sorted_inputs, sorted_keys = decoding._get_sorted_inputs(
        filename, decode_hp.delimiter)
  else:
    sorted_inputs = decoding._get_language_modeling_inputs(
        filename, decode_hp.delimiter, repeat=decode_hp.num_decodes)
    sorted_keys = range(len(sorted_inputs))
  num_sentences = len(sorted_inputs)
  num_decode_batches = (num_sentences - 1) // decode_hp.batch_size + 1

  if estimator.config.use_tpu:
    length = getattr(hparams, "length", 0) or hparams.max_length
    batch_ids = []
    for line in sorted_inputs:
      if has_input:
        ids = inputs_vocab.encode(line.strip()) + [1]
      else:
        ids = targets_vocab.encode(line)
      if len(ids) < length:
        ids.extend([0] * (length - len(ids)))
      else:
        ids = ids[:length]
      batch_ids.append(ids)
    np_ids = np.array(batch_ids, dtype=np.int32)
    def input_fn(params):
      batch_size = params["batch_size"]
      dataset = tf.data.Dataset.from_tensor_slices({"inputs": np_ids})
      dataset = dataset.map(
          lambda ex: {"inputs": tf.reshape(ex["inputs"], (length, 1, 1))})
      dataset = dataset.batch(batch_size)
      return dataset
  else:
    def input_fn():
      input_gen = decoding._decode_batch_input_fn(
          num_decode_batches, sorted_inputs,
          inputs_vocab, decode_hp.batch_size,
          decode_hp.max_input_size,
          task_id=decode_hp.multiproblem_task_id, has_input=has_input)
      gen_fn = decoding.make_input_fn_from_generator(input_gen)
      example = gen_fn()
      return decoding._decode_input_tensor_to_features_dict(example, hparams, decode_hp)
  decodes = []
  result_iter = estimator.predict(input_fn, checkpoint_path=checkpoint_path)

  start_time = time.time()
  total_time_per_step = 0
  total_cnt = 0

  def timer(gen):
    while True:
      try:
        start_time = time.time()
        item = next(gen)
        elapsed_time = time.time() - start_time
        yield elapsed_time, item
      except StopIteration:
        break

  for elapsed_time, result in timer(result_iter):
    if decode_hp.return_beams:
      beam_decodes = []
      beam_scores = []
      output_beams = np.split(result["outputs"], decode_hp.beam_size, axis=0)
      scores = None
      if "scores" in result:
        if np.isscalar(result["scores"]):
          result["scores"] = result["scores"].reshape(1)
        scores = np.split(result["scores"], decode_hp.beam_size, axis=0)
      for k, beam in enumerate(output_beams):
        tf.logging.info("BEAM %d:" % k)
        score = scores and scores[k]
        _, decoded_outputs, _ = decoding.log_decode_results(
            result["inputs"],
            beam,
            problem_name,
            None,
            inputs_vocab,
            targets_vocab,
            log_results=decode_hp.log_results,
            skip_eos_postprocess=decode_hp.skip_eos_postprocess)
        beam_decodes.append(decoded_outputs)
        if decode_hp.write_beam_scores:
          beam_scores.append(score)
      if decode_hp.write_beam_scores:
        decodes.append("\t".join([
            "\t".join([d, "%.2f" % s])
            for d, s in zip(beam_decodes, beam_scores)
        ]))
      else:
        decodes.append("\t".join(beam_decodes))
    else:
      _, decoded_outputs, _ = decoding.log_decode_results(
          result["inputs"],
          result["outputs"],
          problem_name,
          None,
          inputs_vocab,
          targets_vocab,
          log_results=decode_hp.log_results,
          skip_eos_postprocess=decode_hp.skip_eos_postprocess)
      decodes.append(decoded_outputs)
    total_time_per_step += elapsed_time
    total_cnt += result["outputs"].shape[-1]
  duration = time.time() - start_time
  tf.logging.info("Elapsed Time: %5.5f" % duration)
  tf.logging.info("Averaged Single Token Generation Time: %5.7f "
                  "(time %5.7f count %d)" %
                  (total_time_per_step / total_cnt,
                   total_time_per_step, total_cnt))
  if decode_hp.batch_size == 1:
    tf.logging.info("Inference time %.4f seconds "
                    "(Latency = %.4f ms/setences)" %
                    (duration, 1000.0*duration/num_sentences))
  else:
    tf.logging.info("Inference time %.4f seconds "
                    "(Throughput = %.4f sentences/second)" %
                    (duration, num_sentences/duration))

  # If decode_to_file was provided use it as the output filename without change
  # (except for adding shard_id if using more shards for decoding).
  # Otherwise, use the input filename plus model, hp, problem, beam, alpha.
  decode_filename = decode_to_file if decode_to_file else filename
  if not decode_to_file:
    decode_filename = decoding._decode_filename(decode_filename, problem_name, decode_hp)
  else:
    decode_filename = decoding._add_shard_to_filename(decode_filename, decode_hp)
  tf.logging.info("Writing decodes into %s" % decode_filename)
  outfile = tf.gfile.Open(decode_filename, "w")
  for index in range(len(sorted_inputs)):
    special_chars = ["\a", "\n", "\f", "\r", "\b"]
    output = decodes[sorted_keys[index]]
    for c in special_chars:
      output = output.replace(c, ' ')
    try:
      outfile.write("%s%s" % (output, decode_hp.delimiter))
    except:
      outfile.write("%s" % decode_hp.delimiter)
  outfile.flush()
  outfile.close()

  # output_dir = os.path.join(estimator.model_dir, "decode")
  # tf.gfile.MakeDirs(output_dir)

  # run_postdecode_hooks(DecodeHookArgs(
  #     estimator=estimator,
  #     problem=hparams.problem,
  #     output_dirs=[output_dir],
  #     hparams=hparams,
  #     decode_hparams=decode_hp,
  #     predictions=list(result_iter)
  # ), None)


def t2t_decoder(problem_name, data_dir, 
                decode_from_file, decode_to_file,
                checkpoint_path):
  hp, decode_hp, estimator = create_hp_and_estimator(
      problem_name, data_dir, checkpoint_path, decode_to_file)

      
  decode_from_file_fn_(
      estimator, decode_from_file,
      hp, decode_hp, decode_to_file,
      checkpoint_path=checkpoint_path)


def t2t_decoder_eval(problem_name, data_dir, 
                inputs_file, targets_file, loss_to_file,
                checkpoint_path):
  hp, decode_hp, estimator = create_hp_and_estimator(
      problem_name, data_dir, checkpoint_path, loss_to_file)
  evaluate_from_file_fn(
      estimator,
      hp, decode_hp, inputs_file, targets_file, loss_to_file,
      checkpoint_path=checkpoint_path)

# def t2t_decoder_evaluate(problem_name, data_dir, 
#                 decode_from_file, decode_to_file,
#                 checkpoint_path):
  