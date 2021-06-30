import os
import shlex
from time import sleep
from multiprocessing import Process
import subprocess

import tensorflow as tf

flags = tf.flags
FLAGS = flags.FLAGS
# flags.DEFINE_string('task', 'envi' , 'task to train')
# flags.DEFINE_integer('subset', 5, 'number of subset')
# os.system("pip install google.colab")

TPU_ADDRESSES = [
    '10.35.148.218',
    '10.103.25.194',
    '10.92.239.162',
    '10.59.165.42',
    '10.103.53.154',
    '10.50.80.26',
    '10.62.129.234',
    '10.123.242.250',
    '10.91.100.146',
    '10.126.254.18'
]

task = 'envi'
for subset in range(0,1):
    total_train_steps = 500000
    use_tpu = True
    TPU_ADDRESS = TPU_ADDRESSES[subset]
    train_output_dir = f'gs://best_vi_translation/checkpoints/pseudo_label_multicc_translate_{task}_iwslt32k/subset/{subset}/'
    train_data_dir = f'gs://best_vi_translation/data/pseudo_label_multicc_translate_{task}_iwslt32k/subset/{subset}/'
    hparams_str = ('learning_rate_cosine_cycle_steps={},'
                'max_length=128,batch_size=4096,'  # real batch_size = 4096/128
                'learning_rate_constant=2.0').format(2000000)
    print(hparams_str)
    hparams_set = 'transformer_tall9'
    problem = f'pseudo_label_multicc_translate_{task}_iwslt32k'
    sleep(5)
    subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 > nohup_{task}_{subset}.txt"))

# task = 'vien'
# for subset in range(0,5):
#     total_train_steps = 500000
#     use_tpu = True
#     TPU_ADDRESS = TPU_ADDRESSES[5+subset]
#     train_output_dir = f'gs://best_vi_translation/checkpoints/pseudo_label_multicc_translate_{task}_iwslt32k/subset/{subset}/'
#     train_data_dir = f'gs://best_vi_translation/data/pseudo_label_multicc_translate_{task}_iwslt32k/subset/{subset}/'
#     hparams_str = ('learning_rate_cosine_cycle_steps={},'
#                 'max_length=128,batch_size=4096,'  # real batch_size = 4096/128
#                 'learning_rate_constant=2.0').format(2000000)
#     print(hparams_str)
#     hparams_set = 'transformer_tall9'
#     problem = f'pseudo_label_multicc_translate_{task}_iwslt32k'
#     sleep(5)
#     subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 > nohup_{task}_{subset}.txt"))
