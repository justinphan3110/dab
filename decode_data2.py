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
flags.DEFINE_string('task', 'envi', 'type of task to train')

# os.system("pip install google-colab")
from google.colab import auth
auth.authenticate_user()
print('authenticated')


TPU_ADDRESSES = [
    '10.52.19.138',
]

task = FLAGS.task

l = []
for index in range(0,1):
    total_train_steps = 500000
    use_tpu = True
    TPU_ADDRESS = TPU_ADDRESSES[index]
    train_output_dir = f'gs://vien-translation/envi_iswlt15/models_v2/TOK/{task}/tall12_24/model.ckpt-500000'
    train_output_dir = f'gs://best_vi_translation/checkpoints/translate_envi_iwslt32k_tall_12_24_2nd_release/model.ckpt-500000'
    train_data_dir = f'gs://best_vi_translation/data/translate_{task}_iwslt32k_v2_2nd_release'

    hparams_str = ('learning_rate_cosine_cycle_steps={},'
                'max_length=128,batch_size=4096,'  # real batch_size = 4096/128
                'learning_rate_constant=2.0').format(2000000)
    hparams_set = f'transformer_tall_12_24'
    model = 'transformer'
    problem = f'translate_envi_iwslt32k'
    # sleep(5)
    l.append(f"python3 t2t_decoder.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 \
        --model={model} --hparams_set={hparams_set} --hparams={hparams_str} \
        --problem={problem} --data_dir={train_data_dir} \
        --decode_from_file=total_without_software.{task[0:2]} \
        --decode_to_file=./output.txt\
        --output_dir={train_output_dir} \
        --use_tpu={use_tpu}")
    # subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} > nohup_{task}_{subset}.txt"))

os.system('gsutil cp gs://best_vi_translation/raw/testsets/total_without_software.* .')
print(l[FLAGS.index])
# os.system(f'gsutil cp gs://best_vi_translation/checkpoints/pseudo_label_multicc_translate_envi_iwslt32k/subset/{FLAGS.index}/model.ckpt-1000.index .')
os.system(l[FLAGS.index])
    # subprocess.Popen(shlex.split(f"python3 t2t_trainer.py --cloud_tpu_name=grpc://{TPU_ADDRESS}:8470 --model=transformer --hparams_set={hparams_set} --hparams={hparams_str} --train_steps={total_train_steps} --eval_steps=20 --problem={problem} --data_dir={train_data_dir} --output_dir={train_output_dir} --use_tpu={use_tpu} > nohup_{task}_{subset}.txt"))
