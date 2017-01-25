"""Utilities for tokenizing and vocabularies."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import os
import os.path
import sys
import re

from tensorflow.python.platform import gfile
from config.config import _START_VOCAB, EOS_ID, UNK_ID
from config.config import _buckets

def get_train_set_path(path):
  return os.path.join(path, 'train_data')

def get_test_set_path(path):
  return os.path.join(path, 'test_data')

def get_metadata_set_path(path):
  return os.path.join(path, 'metadata.csv')

def tokenize(sentence):
  """Tokenizer: split the sentence into a list of tokens."""
  sentence = clean_html(sentence)
  words = sentence.split()
  return [w for w in words if w.strip()]

def create_vocabulary(vocabulary_path, data_path, max_vocabulary_size):
  """
  Create vocabulary file (if it does not exist yet) from data file.
  Data file should have one sentence per line.
  Each sentence will be tokenized.
  Vocabulary contains the most-frequent tokens up to max_vocabulary_size.
  We write it to vocabulary_path in a one-token-per-line format, so that later
  token in the first line gets id=0, second line gets id=1, and so on.

  Args:
    vocabulary_path: path where the vocabulary will be created.
    data_path: data file that will be used to create vocabulary.
    max_vocabulary_size: limit on the size of the created vocabulary.
  """
  if not os.path.isfile(vocabulary_path):
    print("Creating vocabulary %s from data %s" % (vocabulary_path, data_path))
    vocab = {}
    with open(data_path, 'r+') as f:
      counter = 0
      for line in f:
        counter += 1
        if counter % 100000 == 0:
          print("  processing line %d" % counter)
        try:
          tokens = tokenize(line)
        except:
          print("Tokenize failure: " + line)
          continue
        for w in tokens:
          if vocab.has_key(w):
            vocab[w] += 1
          else:
            vocab[w] = 1
      vocab_list = _START_VOCAB + sorted(vocab, key=vocab.get, reverse=True)
      if len(vocab_list) > max_vocabulary_size:
        vocab_list = vocab_list[:max_vocabulary_size]
      with open(vocabulary_path, 'w+') as vocab_file:
        for w in vocab_list:
          vocab_file.write(w + b"\n")
    print('Vocabulary file created')

def initialize_vocabulary(vocabulary_path):
  """Initialize vocabulary from file.

  We assume the vocabulary is stored one-item-per-line, so a file:
    dog
    cat
  will result in a vocabulary {"dog": 0, "cat": 1}, and this function will
  also return the reversed-vocabulary ["dog", "cat"].

  Args:
    vocabulary_path: path to the file containing the vocabulary.

  Returns:
    a pair: the vocabulary (a dictionary mapping string to integers), and
    the reversed vocabulary (a list, which reverses the vocabulary mapping).

  Raises:
    ValueError: if the provided vocabulary_path does not exist.
  """
  if os.path.isfile(vocabulary_path):
    rev_vocab = []
    with open(vocabulary_path, "r+") as f:
      rev_vocab.extend(f.readlines())
    rev_vocab = [line.strip() for line in rev_vocab]
    vocab = dict([(x, y) for (y, x) in enumerate(rev_vocab)])
    return vocab, rev_vocab
  else:
    raise ValueError("Vocabulary file %s not found.", vocabulary_path)

def sentence_to_token_ids(sentence, vocabulary):
  """Convert a string to list of integers representing token-ids.

  For example, a sentence "How are you?" will be tokenized into
  ["How", "are", "you", "?"] and then lowercased.
  If vocabulary is {"how": 1, "are": 2, "you": 4, "?": 7"} 
  this function will return [1, 2, 4, 7].
  If a word isn't recognized it is replaced with a UNK_ID.

  Args:
    sentence: the plain text input (How are you?)
    vocabulary: a dictionary mapping tokens to integers.

  Returns:
    a list of integers, the token-ids for the sentence.
  """

  sentence = clean_html(sentence)
  words = tokenize(sentence)
  return [vocabulary.get(w, UNK_ID) for w in words]

def data_to_token_ids(data_path, target_path, vocabulary_path):
  """Tokenize data file and turn into token-ids using given vocabulary file.

  This function loads data line-by-line from data_path, calls the above
  sentence_to_token_ids, and saves the result to target_path. See comment
  for sentence_to_token_ids on the details of token-ids format.

  Args:
    data_path: path to the data file in one-sentence-per-line format.
    target_path: path where the file with token-ids will be created.
    vocabulary_path: path to the vocabulary file.
  """
  if not os.path.isfile(target_path):
    print("Tokenizing data in %s" % data_path)
    vocab, _ = initialize_vocabulary(vocabulary_path)
    with open(data_path, 'r+') as data_file:
      with open(target_path, "w") as tokens_file:
        counter = 0
        for line in data_file:
          counter += 1
          if counter % 100000 == 0:
            print("  tokenizing line %d" % counter)
          try:
            token_ids = sentence_to_token_ids(line, vocab)
          except:
            continue
          tokens_file.write(" ".join([str(tok) for tok in token_ids]) + "\n")

def prepare_data(train_dir, vocabulary_size):
  """
  Create vocabulary for training data and dev data.
  Tokenize our data
  Returns a tuple containing:
   - path to token-ids for the training data
   - path to token-ids for development data
   - path to our vocabulary file
  """
  # Get dialog data to the specified directory.
  train_path = get_train_set_path(train_dir)
  test_path = get_test_set_path(train_dir)

  # Create vocabularies of the appropriate sizes.
  vocab_path = os.path.join(train_dir, "vocab%d" % vocabulary_size)
  create_vocabulary(vocab_path, train_path, vocabulary_size)

  # Create token ids for the training data.
  train_ids_path = train_path + ("_ids%d" % vocabulary_size)
  data_to_token_ids(train_path, train_ids_path, vocab_path)

  # Create token ids for the development data.
  test_ids_path = test_path + ("_ids%d" % vocabulary_size)
  data_to_token_ids(test_path, test_ids_path, vocab_path)

  return (train_ids_path, test_ids_path, vocab_path)

def clean_html(html):
  """
  Copied from NLTK package.
  Remove HTML markup from the given string.

  :param html: the HTML string to be cleaned
  :type html: str
  :rtype: str
  """

  # First we remove inline JavaScript/CSS:
  cleaned = re.sub(r"(?is)<(script|style).*?>.*?(</\1>)", "", html.strip())
  # Then we remove html comments. This has to be done before removing regular
  # tags since comments can contain '>' characters.
  cleaned = re.sub(r"(?s)<!--(.*?)-->[\n]?", "", cleaned)
  # Next we can remove the remaining tags:
  cleaned = re.sub(r"(?s)<.*?>", " ", cleaned)
  # Finally, we deal with whitespace
  cleaned = re.sub(r"&nbsp;", " ", cleaned)
  cleaned = re.sub(r"  ", " ", cleaned)
  cleaned = re.sub(r"  ", " ", cleaned)
  return cleaned.strip()

def read_data(source_path, max_size=None):
  """Read data from source and target files and put into buckets.
     Data is considered one line after the other. (input -> output)
  Args:
    source_path: path to the files with token-ids for the source language.
    max_size: maximum number of lines to read, all other will be ignored;
      if 0 or None, data files will be read completely (no limit).

  Returns:
    data_set: a list of length len(_buckets); data_set[n] contains a list of
      (source, target) pairs read from the provided data files that fit
      into the n-th bucket, i.e., such that len(source) < _buckets[n][0] and
      len(target) < _buckets[n][1]; source and target are lists of token-ids.
  """
  data_set = [[] for _ in _buckets]
  with gfile.GFile(source_path, mode="r") as source_file:
    source, target = source_file.readline(), source_file.readline()
    counter = 0
    while source and target and (not max_size or counter < max_size):
      counter += 1
      if counter % 100000 == 0:
        print("  reading data line %d" % counter)
        sys.stdout.flush()
      source_ids = [int(x) for x in source.split()]
      target_ids = [int(x) for x in target.split()]
      target_ids.append(EOS_ID)
      for bucket_id, (source_size, target_size) in enumerate(_buckets):
        if len(source_ids) < source_size and len(target_ids) < target_size:
          data_set[bucket_id].append([source_ids, target_ids])
          break
      source, target = source_file.readline(), source_file.readline()
  return data_set
