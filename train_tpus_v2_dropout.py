import os
import shlex
from time import sleep
from multiprocessing import Process
import subprocess

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('index', 'envi' , 'task to train')
flags.DEFINE_integer('index', 0, 'index to train')
# os.system("pip install google-colab")
from google.colab import auth
auth.authenticate_user()
print('authenticated')


TPU_ADDRESSES = [
    '10.104.189.2',
    '10.8.243.82',
    '10.1.159.90',
    '10.89.131.146',
]

if FLAGS.index <= 1:
    task = 'envi'
else:
    task = 'vien'

print('Training Task ' + task)
print('TPU Address' + TPU_ADDRESSES[FLAGS.index])

DROPOUT_RATE_CONSTANTS = [
  (0.4, 0.1, 0.1),
  (0.3, 0.3, 0.3)
]
# task = FLAGS.task
# assert len(TPU_ADDRESSES) == len(LEARNING_RATE_CONSTANTS)
l = []
for index in range(0,len(TPU_ADDRESSES)):
    total_train_steps = 500000
    use_tpu = True
    TPU_ADDRESS = TPU_ADDRESSES[index]
    DROPOUT_RATE_CONSTANT = DROPOUT_RATE_CONSTANTS[index % len(DROPOUT_RATE_CONSTANTS)]
    
    dropout = '_'.join(map(str, DROPOUT_RATE_CONSTANT))
    train_output_dir = f'gs://best_vi_translation/checkpoints/translate_{task}_iwslt32k_tall_18_18_lr2_{dropout}drop/'
    # train_data_dir = f'gs://best_vi_translation/data/translate_{task}_iwslt32k_v2_2nd_release/'
    train_data_dir = f'gs://best_vi_translation/data/translate_{task}_iwslt32k_v2_2nd_release/'

    hparams_str = ('learning_rate_cosine_cycle_steps=2000000,'
                'max_length=128,batch_size=4096,'  # real batch_size = 4096/128
                'learning_rate_constant=2.0,'
                'layer_prepostprocess_dropout={},attention_dropout={},relu_dropout={}').format(*DROPOUT_RATE_CONSTANT)
    hparams_set = f'transformer_tall_18_18'
    model = 'transformer'
    problem = f'translate_{task}_iwslt32k'
    # sleep(5)
    l.append(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model={model} --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu}")
    # subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} > nohup_{task}_{subset}.txt"))

print(f"training {task}")
print(f"drop out {DROPOUT_RATE_CONSTANTS[FLAGS.index % len(DROPOUT_RATE_CONSTANTS)]}")
print(l[FLAGS.index])
# os.system(f'gsutil cp gs://best_vi_translation/checkpoints/pseudo_label_multicc_translate_envi_iwslt32k/subset/{FLAGS.index}/model.ckpt-1000.index .')
os.system(l[FLAGS.index])
