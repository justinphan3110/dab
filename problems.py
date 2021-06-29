"""Back Translation to augment a dataset."""

from __future__ import print_function
from __future__ import division

from tensor2tensor.data_generators import translate_envi
from tensor2tensor.utils import registry

from tensor2tensor.data_generators import translate
from tensor2tensor.data_generators import problem

# End-of-sentence marker.
EOS = translate_envi.EOS

# For English-Vietnamese the IWSLT'15 corpus
# from https://nlp.stanford.edu/projects/nmt/ is used.



# For development 1,553 parallel sentences are used.
_VIEN_TEST_DATASETS = [[
    # "https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz",  # pylint: disable=line-too-long
    "",
    ("tst2012.vi", "tst2012.en")
]]

_ENVI_TEST_DATASETS = [[
    # "https://github.com/stefan-it/nmt-en-vi/raw/master/data/dev-2012-en-vi.tgz",  # pylint: disable=line-too-long
    "",
    ("tst2012.en", "tst2012.vi")
]]


# The original dataset has 133K parallel sentences.
_K133VIEN_TRAIN_DATASETS = [[
    "https://github.com/stefan-it/nmt-en-vi/raw/master/data/train-en-vi.tgz",  # pylint: disable=line-too-long
    ("train.vi", "train.en")
]]

# ----------------28.03.21
# Multicc split low bleu
_MULTICCMIN20LOWBLEU_CLASS11_ENVI_TRAIN_DATASETS = [
    ['', ('multicc_min20_split_0.en', 'multicc_min20_split_0.vi')],
    ['', ('train.en.fixed', 'train.vi.fixed')],  # original.
    ['', ('medical.en.fixed.filter', 'medical.vi.fixed.filter')],
    ['', ('Gnome.en.fixed.filter.train', 'Gnome.vi.fixed.filter.train')],
    ['', ('Kde4.en.fixed.filter.train', 'Kde4.vi.fixed.filter.train')],
    ['', ('law.en.fixed.filter.fixed', 'law.vi.fixed.filter.fixed')],
    ['', ('fbwiki.en.fixed', 'fbwiki.vi.fixed')],
    ['', ('qed.en.fixed.filter', 'qed.vi.fixed.filter')],
    ['', ('open_sub.en.fixed.max250k', 'open_sub.vi.fixed.max250k')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train', 'Ubuntu_tp_vi.txt.fixed.filter.train')],
    ['', ('ELRC_2922.en-vi.en.fixed', 'ELRC_2922.en-vi.vi.fixed')],
    ['', ('bible_uedin.en.fixed', 'bible_uedin.vi.fixed')],
    ['', ('m21book_add2train.en.fixed', 'm21book_add2train.vi.fixed')],
    ['', ('tatoeba.en.fixed', 'tatoeba.vi.fixed')],
    ['', ('ted2020.en.fixed.filter.fixed', 'ted2020.vi.fixed.filter.fixed')],
    ['', ('vnsn.en.filter.fixed', 'vnsn.vi.filter.fixed')],
    ['', ('youtube.fixed.en.fixed', 'youtube.fixed.vi.fixed')],
    ['', ('youtube.teded.en.fixed.filter.fixed', 'youtube.teded.vi.fixed.filter.fixed')],
  ]

@registry.register_problem
class Multiccmin20lowbleuClass11TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICCMIN20LOWBLEU_CLASS11_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_MULTICCMIN20LOWBLEU_CLASS11_VIEN_TRAIN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _MULTICCMIN20LOWBLEU_CLASS11_ENVI_TRAIN_DATASETS
]

@registry.register_problem
class Multiccmin20lowbleuClass11TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICCMIN20LOWBLEU_CLASS11_VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS

# Multicc split mid bleu

_MULTICCMIN20MIDBLEU_CLASS11_ENVI_TRAIN_DATASETS = [
    ['', ('multicc_min20_split_1.en', 'multicc_min20_split_1.vi')],
    ['', ('train.en.fixed', 'train.vi.fixed')],  # original.
    ['', ('medical.en.fixed.filter', 'medical.vi.fixed.filter')],
    ['', ('Gnome.en.fixed.filter.train', 'Gnome.vi.fixed.filter.train')],
    ['', ('Kde4.en.fixed.filter.train', 'Kde4.vi.fixed.filter.train')],
    ['', ('law.en.fixed.filter.fixed', 'law.vi.fixed.filter.fixed')],
    ['', ('fbwiki.en.fixed', 'fbwiki.vi.fixed')],
    ['', ('qed.en.fixed.filter', 'qed.vi.fixed.filter')],
    ['', ('open_sub.en.fixed.max250k', 'open_sub.vi.fixed.max250k')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train', 'Ubuntu_tp_vi.txt.fixed.filter.train')],
    ['', ('ELRC_2922.en-vi.en.fixed', 'ELRC_2922.en-vi.vi.fixed')],
    ['', ('bible_uedin.en.fixed', 'bible_uedin.vi.fixed')],
    ['', ('m21book_add2train.en.fixed', 'm21book_add2train.vi.fixed')],
    ['', ('tatoeba.en.fixed', 'tatoeba.vi.fixed')],
    ['', ('ted2020.en.fixed.filter.fixed', 'ted2020.vi.fixed.filter.fixed')],
    ['', ('vnsn.en.filter.fixed', 'vnsn.vi.filter.fixed')],
    ['', ('youtube.fixed.en.fixed', 'youtube.fixed.vi.fixed')],
    ['', ('youtube.teded.en.fixed.filter.fixed', 'youtube.teded.vi.fixed.filter.fixed')],
  ]

@registry.register_problem
class Multiccmin20midbleuClass11TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICCMIN20MIDBLEU_CLASS11_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_MULTICCMIN20MIDBLEU_CLASS11_VIEN_TRAIN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _MULTICCMIN20MIDBLEU_CLASS11_ENVI_TRAIN_DATASETS
]

@registry.register_problem
class Multiccmin20midbleuClass11TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICCMIN20MIDBLEU_CLASS11_VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


# Multicc split high bleu

_MULTICCMIN20HIGHBLEU_CLASS11_ENVI_TRAIN_DATASETS = [
    ['', ('multicc_min20_split_2.en', 'multicc_min20_split_2.vi')],
    ['', ('train.en.fixed', 'train.vi.fixed')],  # original.
    ['', ('medical.en.fixed.filter', 'medical.vi.fixed.filter')],
    ['', ('Gnome.en.fixed.filter.train', 'Gnome.vi.fixed.filter.train')],
    ['', ('Kde4.en.fixed.filter.train', 'Kde4.vi.fixed.filter.train')],
    ['', ('law.en.fixed.filter.fixed', 'law.vi.fixed.filter.fixed')],
    ['', ('fbwiki.en.fixed', 'fbwiki.vi.fixed')],
    ['', ('qed.en.fixed.filter', 'qed.vi.fixed.filter')],
    ['', ('open_sub.en.fixed.max250k', 'open_sub.vi.fixed.max250k')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train', 'Ubuntu_tp_vi.txt.fixed.filter.train')],
    ['', ('ELRC_2922.en-vi.en.fixed', 'ELRC_2922.en-vi.vi.fixed')],
    ['', ('bible_uedin.en.fixed', 'bible_uedin.vi.fixed')],
    ['', ('m21book_add2train.en.fixed', 'm21book_add2train.vi.fixed')],
    ['', ('tatoeba.en.fixed', 'tatoeba.vi.fixed')],
    ['', ('ted2020.en.fixed.filter.fixed', 'ted2020.vi.fixed.filter.fixed')],
    ['', ('vnsn.en.filter.fixed', 'vnsn.vi.filter.fixed')],
    ['', ('youtube.fixed.en.fixed', 'youtube.fixed.vi.fixed')],
    ['', ('youtube.teded.en.fixed.filter.fixed', 'youtube.teded.vi.fixed.filter.fixed')],
  ]

@registry.register_problem
class Multiccmin20highbleuClass11TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICCMIN20HIGHBLEU_CLASS11_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_MULTICCMIN20HIGHBLEU_CLASS11_VIEN_TRAIN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _MULTICCMIN20HIGHBLEU_CLASS11_ENVI_TRAIN_DATASETS
]

@registry.register_problem
class Multiccmin20highbleuClass11TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICCMIN20HIGHBLEU_CLASS11_VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS

# --------------------------28.03.21---------

_MULTICC_ENVI_TRAIN_DATASETS = [

  ['', ('train.en', 'train.vi')]
]

@registry.register_problem
class MulticcTranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICC_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_MULTICC_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("MultiCCAligned.vi.fixed.filter.filtertest", "MultiCCAligned.en.fixed.filter.filtertest")
]]

@registry.register_problem
class MulticcTranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICC_VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


_MULTICC_11CLASSES_PURE_ENVI_DATASETS = [
    ['', ('MultiCCAligned.en.fixed.filter.filtertest', 'MultiCCAligned.vi.fixed.filter.filtertest')],
    ['', ('train.en.fixed', 'train.vi.fixed')],  # original.
    ['', ('medical.en.fixed.filter', 'medical.vi.fixed.filter')],
    ['', ('Gnome.en.fixed.filter.train', 'Gnome.vi.fixed.filter.train')],
    ['', ('Kde4.en.fixed.filter.train', 'Kde4.vi.fixed.filter.train')],
    ['', ('law.en.fixed.filter.fixed', 'law.vi.fixed.filter.fixed')],
    ['', ('fbwiki.en.fixed', 'fbwiki.vi.fixed')],
    ['', ('qed.en.fixed.filter', 'qed.vi.fixed.filter')],
    ['', ('open_sub.en.fixed.max250k', 'open_sub.vi.fixed.max250k')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train', 'Ubuntu_tp_vi.txt.fixed.filter.train')],
    ['', ('ELRC_2922.en-vi.en.fixed', 'ELRC_2922.en-vi.vi.fixed')],
    ['', ('bible_uedin.en.fixed', 'bible_uedin.vi.fixed')],
    ['', ('m21book_add2train.en.fixed', 'm21book_add2train.vi.fixed')],
    ['', ('tatoeba.en.fixed', 'tatoeba.vi.fixed')],
    ['', ('ted2020.en.fixed.filter.fixed', 'ted2020.vi.fixed.filter.fixed')],
    ['', ('vnsn.en.filter.fixed', 'vnsn.vi.filter.fixed')],
    ['', ('youtube.fixed.en.fixed', 'youtube.fixed.vi.fixed')],
    ['', ('youtube.teded.en.fixed.filter.fixed', 'youtube.teded.vi.fixed.filter.fixed')],
]


_MULTICC_11CLASSES_PURE_VIEN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _MULTICC_11CLASSES_PURE_ENVI_DATASETS
]



@registry.register_problem
class TranslateMulticcClass11PureVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICC_11CLASSES_PURE_VIEN_DATASETS if train else _VIEN_TEST_DATASETS


@registry.register_problem
class TranslateMulticcClass11PureEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _MULTICC_11CLASSES_PURE_ENVI_DATASETS if train else _ENVI_TEST_DATASETS


# ------------------------

_11CLASSES_APPEND_TAG_ENVI_DATASETS = [
    ['', ('train.en.fixed.append128.tag', 'train.vi.fixed.append128.tag')],  # original.
    ['', ('medical.en.fixed.filter.append128.tag', 'medical.vi.fixed.filter.append128.tag')],
    ['', ('Gnome.en.fixed.filter.train.tag', 'Gnome.vi.fixed.filter.train.tag')],
    ['', ('Kde4.en.fixed.filter.train.tag', 'Kde4.vi.fixed.filter.train.tag')],
    ['', ('law.en.fixed.filter.fixed.tag', 'law.vi.fixed.filter.fixed.tag')],
    ['', ('fbwiki.en.fixed.tag', 'fbwiki.vi.fixed.tag')],
    ['', ('qed.en.fixed.filter.append128.tag', 'qed.vi.fixed.filter.append128.tag')],
    ['', ('open_sub.en.fixed.max250k.append128.tag', 'open_sub.vi.fixed.max250k.append128.tag')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train.tag', 'Ubuntu_tp_vi.txt.fixed.filter.train.tag')],
    ['', ('ELRC_2922.en-vi.en.fixed.append128.tag', 'ELRC_2922.en-vi.vi.fixed.append128.tag')],
    ['', ('bible_uedin.en.fixed.append128.tag', 'bible_uedin.vi.fixed.append128.tag')],
    ['', ('m21book_add2train.en.fixed.append128.tag', 'm21book_add2train.vi.fixed.append128.tag')],
    ['', ('tatoeba.en.fixed.append128.tag', 'tatoeba.vi.fixed.append128.tag')],
    ['', ('ted2020.en.fixed.filter.fixed.append128.tag', 'ted2020.vi.fixed.filter.fixed.append128.tag')],
    ['', ('vnsn.en.filter.fixed.append128.tag', 'vnsn.vi.filter.fixed.append128.tag')],
    ['', ('youtube.fixed.en.fixed.append128.tag', 'youtube.fixed.vi.fixed.append128.tag')],
    ['', ('youtube.teded.en.fixed.filter.fixed.append128.tag', 'youtube.teded.vi.fixed.filter.fixed.append128.tag')],
]


_11CLASSES_APPEND_TAG_VIEN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _11CLASSES_APPEND_TAG_ENVI_DATASETS
]


@registry.register_problem
class TranslateClass11AppendtagVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_APPEND_TAG_VIEN_DATASETS if train else _VIEN_TEST_DATASETS


@registry.register_problem
class TranslateClass11AppendtagEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_APPEND_TAG_ENVI_DATASETS if train else _ENVI_TEST_DATASETS


_11CLASSES_APPEND_ENVI_DATASETS = [
    ['', ('train.en.fixed.append128', 'train.vi.fixed.append128')],  # original.
    ['', ('medical.en.fixed.filter.append128', 'medical.vi.fixed.filter.append128')],
    ['', ('Gnome.en.fixed.filter.train.append128', 'Gnome.vi.fixed.filter.train.append128')],
    ['', ('Kde4.en.fixed.filter.train.append128', 'Kde4.vi.fixed.filter.train.append128')],
    ['', ('law.en.fixed.filter.fixed.append128', 'law.vi.fixed.filter.fixed.append128')],
    ['', ('fbwiki.en.fixed.append128', 'fbwiki.vi.fixed.append128')],
    ['', ('qed.en.fixed.filter.append128', 'qed.vi.fixed.filter.append128')],
    ['', ('open_sub.en.fixed.max250k.append128', 'open_sub.vi.fixed.max250k.append128')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train.append128', 'Ubuntu_tp_vi.txt.fixed.filter.train.append128')],
    ['', ('ELRC_2922.en-vi.en.fixed.append128', 'ELRC_2922.en-vi.vi.fixed.append128')],
    ['', ('bible_uedin.en.fixed.append128', 'bible_uedin.vi.fixed.append128')],
    ['', ('m21book_add2train.en.fixed.append128', 'm21book_add2train.vi.fixed.append128')],
    ['', ('tatoeba.en.fixed.append128', 'tatoeba.vi.fixed.append128')],
    ['', ('ted2020.en.fixed.filter.fixed.append128', 'ted2020.vi.fixed.filter.fixed.append128')],
    ['', ('vnsn.en.filter.fixed.append128', 'vnsn.vi.filter.fixed.append128')],
    ['', ('youtube.fixed.en.fixed.append128', 'youtube.fixed.vi.fixed.append128')],
    ['', ('youtube.teded.en.fixed.filter.fixed.append128', 'youtube.teded.vi.fixed.filter.fixed.append128')],
]


_11CLASSES_APPEND_VIEN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _11CLASSES_APPEND_ENVI_DATASETS
]


@registry.register_problem
class TranslateClass11AppendVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_APPEND_VIEN_DATASETS if train else _VIEN_TEST_DATASETS


@registry.register_problem
class TranslateClass11AppendEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_APPEND_ENVI_DATASETS if train else _ENVI_TEST_DATASETS


_11CLASSES_TAG_ENVI_DATASETS = [
    ['', ('train.en.fixed.tag', 'train.vi.fixed.tag')],  # original.
    ['', ('medical.en.fixed.filter.tag', 'medical.vi.fixed.filter.tag')],
    ['', ('Gnome.en.fixed.filter.train.tag', 'Gnome.vi.fixed.filter.train.tag')],
    ['', ('Kde4.en.fixed.filter.train.tag', 'Kde4.vi.fixed.filter.train.tag')],
    ['', ('law.en.fixed.filter.fixed.tag', 'law.vi.fixed.filter.fixed.tag')],
    ['', ('fbwiki.en.fixed.tag', 'fbwiki.vi.fixed.tag')],
    ['', ('qed.en.fixed.filter.tag', 'qed.vi.fixed.filter.tag')],
    ['', ('open_sub.en.fixed.max250k.tag', 'open_sub.vi.fixed.max250k.tag')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train.tag', 'Ubuntu_tp_vi.txt.fixed.filter.train.tag')],
    ['', ('ELRC_2922.en-vi.en.fixed.tag', 'ELRC_2922.en-vi.vi.fixed.tag')],
    ['', ('bible_uedin.en.fixed.tag', 'bible_uedin.vi.fixed.tag')],
    ['', ('m21book_add2train.en.fixed.tag', 'm21book_add2train.vi.fixed.tag')],
    ['', ('tatoeba.en.fixed.tag', 'tatoeba.vi.fixed.tag')],
    ['', ('ted2020.en.fixed.filter.fixed.tag', 'ted2020.vi.fixed.filter.fixed.tag')],
    ['', ('vnsn.en.filter.fixed.tag', 'vnsn.vi.filter.fixed.tag')],
    ['', ('youtube.fixed.en.fixed.tag', 'youtube.fixed.vi.fixed.tag')],
    ['', ('youtube.teded.en.fixed.filter.fixed.tag', 'youtube.teded.vi.fixed.filter.fixed.tag')],
]


_11CLASSES_TAG_VIEN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _11CLASSES_TAG_ENVI_DATASETS
]


@registry.register_problem
class TranslateClass11TagVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_TAG_VIEN_DATASETS if train else _VIEN_TEST_DATASETS


@registry.register_problem
class TranslateClass11TagEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_TAG_ENVI_DATASETS if train else _ENVI_TEST_DATASETS


_11CLASSES_PURE_ENVI_DATASETS = [
    ['', ('train.en.fixed', 'train.vi.fixed')],  # original.
    ['', ('medical.en.fixed.filter', 'medical.vi.fixed.filter')],
    ['', ('Gnome.en.fixed.filter.train', 'Gnome.vi.fixed.filter.train')],
    ['', ('Kde4.en.fixed.filter.train', 'Kde4.vi.fixed.filter.train')],
    ['', ('law.en.fixed.filter.fixed', 'law.vi.fixed.filter.fixed')],
    ['', ('fbwiki.en.fixed', 'fbwiki.vi.fixed')],
    ['', ('qed.en.fixed.filter', 'qed.vi.fixed.filter')],
    ['', ('open_sub.en.fixed.max250k', 'open_sub.vi.fixed.max250k')],
    ['', ('Ubuntu_tp_en.txt.fixed.filter.train', 'Ubuntu_tp_vi.txt.fixed.filter.train')],
    ['', ('ELRC_2922.en-vi.en.fixed', 'ELRC_2922.en-vi.vi.fixed')],
    ['', ('bible_uedin.en.fixed', 'bible_uedin.vi.fixed')],
    ['', ('m21book_add2train.en.fixed', 'm21book_add2train.vi.fixed')],
    ['', ('tatoeba.en.fixed', 'tatoeba.vi.fixed')],
    ['', ('ted2020.en.fixed.filter.fixed', 'ted2020.vi.fixed.filter.fixed')],
    ['', ('vnsn.en.filter.fixed', 'vnsn.vi.filter.fixed')],
    ['', ('youtube.fixed.en.fixed', 'youtube.fixed.vi.fixed')],
    ['', ('youtube.teded.en.fixed.filter.fixed', 'youtube.teded.vi.fixed.filter.fixed')],
]


_11CLASSES_PURE_VIEN_DATASETS = [
    [url, (vi, en)] for [url, (en, vi)] in _11CLASSES_PURE_ENVI_DATASETS
]



@registry.register_problem
class TranslateClass11PureVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_PURE_VIEN_DATASETS if train else _VIEN_TEST_DATASETS


@registry.register_problem
class TranslateClass11PureEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _11CLASSES_PURE_ENVI_DATASETS if train else _ENVI_TEST_DATASETS



@registry.register_problem
class TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K133VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


_K200ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("k200train.en", "k200train.vi")
]]

@registry.register_problem
class K200TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K200ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_K500ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("k500train.en", "k500train.vi")
]]


@registry.register_problem
class K500TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K500ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_K760ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_k760train.en", "fb_wiki_k760train.vi")
]]


@registry.register_problem
class FbWikiK760TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_K760ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_AVG61_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.avg61.en", "train.6m.avg61.vi")
]]


@registry.register_problem
class M6Avg61TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_AVG61_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_AVG25_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.avg25.en", "train.6m.avg25.vi")
]]


@registry.register_problem
class M6Avg25TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_AVG25_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_M6_LONGEST_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.longest.en", "train.6m.longest.vi")
]]


@registry.register_problem
class M6LongestTranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_LONGEST_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_SHORTEST_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.shortest.en", "train.6m.shortest.vi")
]]


@registry.register_problem
class M6ShortestTranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_SHORTEST_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M6_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m6train.en", "fb_wiki_m6train.vi")
]]


@registry.register_problem
class FbWikiM6TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M6_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M12_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m12train.en", "fb_wiki_m12train.vi")
]]


@registry.register_problem
class FbWikiM12TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M12_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M18_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m18train.en", "fb_wiki_m18train.vi")
]]


@registry.register_problem
class FbWikiM18TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M18_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M30_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m30train.en", "fb_wiki_m30train.vi")
]]


@registry.register_problem
class FbWikiM30TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M30_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_FB_WIKI_BESTM30_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_bestm30train.en", "fb_wiki_bestm30train.vi")
]]


@registry.register_problem
class FbWikiBestm30TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_BESTM30_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_BOOK_M30_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("book_m30train.en", "book_m30train.vi")
]]


@registry.register_problem
class Bookm30TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _BOOK_M30_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_AND_BOOK_M45_ENVI_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_and_book_m45train.en", "fb_wiki_and_book_m45train.vi")
]]


@registry.register_problem
class FbWikiAndBookM45TranslateEnviIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_AND_BOOK_M45_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

###################################### VI-EN #######################################

_K200VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("k200train.vi", "k200train.en")
]]


@registry.register_problem
class K200TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K200VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


_K500VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("k500train.vi", "k500train.en")
]]

@registry.register_problem
class K500TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _K500VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS



_FB_WIKI_K760VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_k760train.vi", "fb_wiki_k760train.en")
]]


@registry.register_problem
class FbWikiK760TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_K760VIEN_TRAIN_DATASETS if train else _VIEN_TEST_DATASETS


_M6_AVG61_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.avg61.vi", "train.6m.avg61.en")
]]


@registry.register_problem
class M6Avg61TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_AVG61_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_AVG25_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.avg25.vi", "train.6m.avg25.en")
]]


@registry.register_problem
class M6Avg25TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_AVG25_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS



_M6_LONGEST_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.longest.vi", "train.6m.longest.en")
]]


@registry.register_problem
class M6LongestTranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_LONGEST_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_M6_SHORTEST_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("train.6m.shortest.vi", "train.6m.shortest.en")
]]


@registry.register_problem
class M6ShortestTranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _M6_SHORTEST_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M6_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m6train.vi", "fb_wiki_m6train.en")
]]


@registry.register_problem
class FbWikiM6TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M6_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M12_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m12train.vi", "fb_wiki_m12train.en")
]]


@registry.register_problem
class FbWikiM12TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M12_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


_FB_WIKI_M18_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m18train.vi", "fb_wiki_m18train.en")
]]


@registry.register_problem
class FbWikiM18TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M18_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_FB_WIKI_M30_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_m30train.vi", "fb_wiki_m30train.en")
]]


@registry.register_problem
class FbWikiM30TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_M30_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_FB_WIKI_BESTM30_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_bestm30train.vi", "fb_wiki_bestm30train.en")
]]


@registry.register_problem
class FbWikiBestm30TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_BESTM30_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_BOOK_M30_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("book_m30train.vi", "book_m30train.en")
]]


@registry.register_problem
class Bookm30TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _BOOK_M30_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS

_FB_WIKI_AND_BOOK_M45_VIEN_TRAIN_DATASETS = [[
    "",  # pylint: disable=line-too-long
    ("fb_wiki_and_book_m45train.vi", "fb_wiki_and_book_m45train.en")
]]


@registry.register_problem
class FbWikiAndBookM45TranslateVienIwslt32k(translate_envi.TranslateEnviIwslt32k):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == translate_envi.problem.DatasetSplit.TRAIN
    return _FB_WIKI_AND_BOOK_M45_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS


# --------------------------16.06.21---------

# _PSEUDO_LABEL_MULTICC_ENVI_TRAIN_DATASETS = [
#   # ["",("MultiCCAligned.en.fixed.filter.filtertest.subset", "MultiCCAligned.vi.fixed.filter.filtertest.subset")],
#   ['https://github.com/stefan-it/nmt-en-vi/raw/master/data/train-en-vi.tgz', ('train.en', 'train.vi')]
#   ]

# _PSEUDO_LABEL_MULTICC_ENVI_TRAIN_DATASETS = [[
#     "",  # pylint: disable=line-too-long
#     ("train.en", "train.vi")
# ], 

# ["",("MultiCCAligned.en.fixed.filter.filtertest.subset", "MultiCCAligned.vi.fixed.filter.filtertest.subset")]

# ]


# _PSEUDO_LABEL_MULTICC_ENVI_TRAIN_DATASETS = [[
#     "",  # pylint: disable=line-too-long
#     ("train.en", "train.vi")
# ]]

_PSEUDO_LABEL_MULTICC_ENVI_TRAIN_DATASETS = [
    ['', ('train.en', 'train.vi')],  # original.
    ['', ('MultiCCAligned.en.fixed.filter.filtertest.subset', 'MultiCCAligned.vi.fixed.filter.filtertest.subset')]
]

@registry.register_problem
class PseudoLabelMulticcTranslateEnviIwslt32k(translate.TranslateProblem):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _PSEUDO_LABEL_MULTICC_ENVI_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS




_PSEUDO_LABEL_MULTICC_VIEN_TRAIN_DATASETS = [
    ['', ('train.vi', 'train.en')],  # original.
    ['', ('MultiCCAligned.vi.fixed.filter.filtertest.subset', 'MultiCCAligned.en.fixed.filter.filtertest.subset')]
]


@registry.register_problem
class PseudoLabelMulticcTranslateVienIwslt32k(translate.TranslateProblem):
  """Problem spec for IWSLT'15 En-Vi translation."""

  @property
  def approx_vocab_size(self):
    return 2**15  # 32768

  def source_data_files(self, dataset_split):
    train = dataset_split == problem.DatasetSplit.TRAIN
    return _PSEUDO_LABEL_MULTICC_VIEN_TRAIN_DATASETS if train else _ENVI_TEST_DATASETS