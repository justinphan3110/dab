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
    '10.7.143.146',
    '10.13.130.210',
    '10.111.251.154',
    '10.124.138.194',
    '10.69.84.242',
    '10.75.247.74',
]

LEARNING_RATE_CONSTANTS = [
    '2.0',
    '1.0', 
    '4.0'
]

if FLAGS.index <= 2:
    task = 'envi'
else:
    task = 'vien'

print('Training Task ' + task)
print('TPU Address' + TPU_ADDRESSES[FLAGS.index])
print('LR : ' + LEARNING_RATE_CONSTANTS[FLAGS.index])

# task = FLAGS.task
assert len(TPU_ADDRESSES) == len(LEARNING_RATE_CONSTANTS)
l = []
for index in range(0,len(TPU_ADDRESSES)):
    total_train_steps = 500000
    use_tpu = True
    TPU_ADDRESS = TPU_ADDRESSES[index]
    LEARNING_RATE_CONSTANT = LEARNING_RATE_CONSTANTS[index]
    
    train_output_dir = f'gs://best_vi_translation/checkpoints/translate_{task}_iwslt32k_tall_18_18_{LEARNING_RATE_CONSTANT}lr/'
    # train_data_dir = f'gs://best_vi_translation/data/translate_{task}_iwslt32k_v2_2nd_release/'
    train_data_dir = f'gs://best_vi_translation/data/translate_{task}_iwslt32k_v2_2nd_release/'

    hparams_str = ('learning_rate_cosine_cycle_steps=2000000,'
                'max_length=128,batch_size=4096,'  # real batch_size = 4096/128
                'learning_rate_constant={}').format(LEARNING_RATE_CONSTANT)
    hparams_set = f'transformer_tall_12_24'
    model = 'transformer'
    problem = f'translate_envi_iwslt32k'
    # sleep(5)
    l.append(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model={model} --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu}")
    # subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} > nohup_{task}_{subset}.txt"))

print(l[FLAGS.index])
# os.system(f'gsutil cp gs://best_vi_translation/checkpoints/pseudo_label_multicc_translate_envi_iwslt32k/subset/{FLAGS.index}/model.ckpt-1000.index .')
os.system(l[FLAGS.index])
    # subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} > nohup_{task}_{subset}.txt"))
