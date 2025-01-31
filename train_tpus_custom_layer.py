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
    '10.104.150.114',
    '10.54.142.130',
    '10.31.66.74',
    '10.37.178.250',
    '10.106.141.10',
    '10.8.135.170',
    '10.2.56.42',
    '10.112.216.98',
    '10.28.26.194',
    '10.65.207.202'
]

task = 'vien'

l = []
for index, (encoder_layers,decoder_layers) in enumerate([[12,6], [15,3]]):
    total_train_steps = 500000
    use_tpu = True
    TPU_ADDRESS = TPU_ADDRESSES[index]
    train_output_dir = f'gs://best_vi_translation/checkpoints/custom_layer_translate_class11_pure_{task}_iwslt32k/{encoder_layers}_{decoder_layers}/'
    train_data_dir = f'gs://best_vi_translation/data/translate_class11_pure_{task}_iwslt32k/'

    hparams_str = ('learning_rate_cosine_cycle_steps={},'
                'max_length=128,batch_size=4096,'  # real batch_size = 4096/128
                'learning_rate_constant=2.0').format(2000000)
    hparams_set = f'transformer_tall_{encoder_layers}_{decoder_layers}'
    model = 'transformer'
    problem = f'translate_class11_pure_{task}_iwslt32k'
    # sleep(5)
    l.append(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model={model} --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu}")
    # subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} > nohup_{task}_{subset}.txt"))

print(l[FLAGS.index])
# os.system(f'gsutil cp gs://best_vi_translation/checkpoints/pseudo_label_multicc_translate_envi_iwslt32k/subset/{FLAGS.index}/model.ckpt-1000.index .')
os.system(l[FLAGS.index])
    # subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} > nohup_{task}_{subset}.txt"))
