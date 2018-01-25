#!/usr/bin/python3
# -*- coding: utf-8 -*-


import re
import unicodedata
import random
import torch
import collections

# TODO: Remove thats global variables

#log constant
DEBUG_LOG = False
ST_LOG = True

class Indexer:
    def __init__(self):
        self.trimmed = False
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

    def index_words(self, sentence):
        for word in sentence.split(' '):
            self.index_word(word)

    def index_word(self, word):
        if word not in self.word2index:
            self.word2index[word] = self.n_words
            self.word2count[word] = 1
            self.index2word[self.n_words] = word
            self.n_words += 1
        else:
            self.word2count[word] += 1

    # Remove words below a certain count threshold
    def trim(self, min_count):
        if self.trimmed: return
        self.trimmed = True

        keep_words = []

        for k, v in self.word2count.items():
            if v >= min_count:
                keep_words.append(k)

        print('keep_words %s / %s = %.4f' % (
            len(keep_words), len(self.word2index), len(keep_words) / len(self.word2index)
        ))

        # Reinitialize dictionaries
        self.word2index = {}
        self.word2count = {}
        self.index2word = {0: "PAD", 1: "SOS", 2: "EOS"}
        self.n_words = 3 # Count default tokens

        for word in keep_words:
            self.index_word(word)

Batch = collections.namedtuple('Batch', ['src_var', 'src_lengths', 'tgt_var', 'tgt_lengths'])

class Data(object):
    """docstring for Data."""
    def __init__(self, min_sentence_length, max_sentence_length,
                    USE_CUDA = False, PAD_TOKEN = 0, EOS_TOKEN = 2):
        super(Data, self).__init__()
        self.min_sentence_length = min_sentence_length
        self.max_sentence_length = max_sentence_length
        self.USE_CUDA = USE_CUDA
        self.PAD_TOKEN = PAD_TOKEN
        self.EOS_TOKEN = EOS_TOKEN

    def dataload(self, srcfile, tgtfile):
        if ST_LOG : print('Loading data...')
        srclines = open(srcfile).read().strip().split('\n')
        tgtlines = open(tgtfile).read().strip().split('\n')
        # print(srclines[:10],tgtlines[:10])

        if ST_LOG : print('Normalization...')
        srclines = [self._preprocessing(src) for src in srclines]
        tgtlines = [self._preprocessing(tgt) for tgt in tgtlines]
        # print(srclines[:10],tgtlines[:10])

        if ST_LOG : print('Merging to pairs...')
        pairs = [[src, tgt] for src, tgt in zip(srclines,tgtlines)]
        # print(pairs[:10])
        # print(len(pairs))

        if ST_LOG : print('Filtering...')
        pairs = self._filter_pairs(pairs)
        # print(pairs[:10])
        # print(len(pairs))

        if ST_LOG : print('Indexing...')
        self.src_indexer = Indexer()
        self.tgt_indexer = Indexer()
        for pair in pairs:
            self.src_indexer.index_words(pair[0])
            self.tgt_indexer.index_words(pair[1])
        self.digit_pairs =[]
        for pair in pairs:
            digit_pair = self._sentence_indexes(pair[0], pair[1])
            self.digit_pairs.append(digit_pair)
        # print(self.digit_pairs[:10])
        # print(len(self.digit_pairs))


    def _filter_pairs(self,pairs):
        filtered_pairs = []
        for pair in pairs:
            if len(pair[0]) >= self.min_sentence_length and len(pair[0]) <= self.max_sentence_length \
                and len(pair[1]) >= self.min_sentence_length and len(pair[1]) <= self.max_sentence_length:
                    filtered_pairs.append(pair)
        return filtered_pairs

    def _sentence_indexes(self,src_sen,tgt_sen):
        src_sen = [self.src_indexer.word2index[word] for word in src_sen.split(' ')] + [self.EOS_TOKEN]
        tgt_sen = [self.tgt_indexer.word2index[word] for word in tgt_sen.split(' ')] + [self.EOS_TOKEN]
        return src_sen, tgt_sen

    def _preprocessing(self,sentence):
        # def unicode_to_ascii(s):
        #     return ''.join(
        #         c for c in unicodedata.normalize('NFD', s)
        #         if unicodedata.category(c) != 'Mn'
        #     )
        # s = unicode_to_ascii(sentence.lower().strip())
        s = sentence.lower().strip()
        s = re.sub(r"([,.!?\-\(\)])", r" \1 ", s)
        s = re.sub(r"[^\w,.!?\-\(\)]+", r" ", s)
        s = re.sub(r"\s+", r" ", s).strip()
        # print(sentence)
        # print(s)
        return s

    def batch_generator(self, batch_size):
        random.shuffle(self.digit_pairs)
        pairs_len = len(self.digit_pairs)
        iter_num = pairs_len//batch_size
        if ST_LOG : print('Start batch generator')
        # for i in range(iter_num):
        index = -1
        index_shift = 0
        while(True):
            index+=1
            if index < iter_num:
                epoch_is_end = False
            else:
                index = 0
                index_shift = 0
                epoch_is_end = True
            # print("index = {}".format(index))
            batch = [pair for pair in self.digit_pairs[index_shift:index_shift + batch_size]]
            index_shift += batch_size
            # Zip into pairs, sort by length (descending), unzip
            batch = sorted(batch, key=lambda p: len(p[0]), reverse=True)
            src_seqs, tgt_seqs = zip(*batch)

            # For src and tgt sequences, get array of lengths and pad with 0s to max length
            src_lengths = [len(s) for s in src_seqs]
            src_padded = [self._pad_seq(s, max(src_lengths)) for s in src_seqs]
            tgt_lengths = [len(s) for s in tgt_seqs]
            tgt_padded = [self._pad_seq(s, max(tgt_lengths)) for s in tgt_seqs]

            # Turn padded arrays into (batch_size x max_len) tensors, transpose into (max_len x batch_size)
            src_var = torch.autograd.Variable(torch.LongTensor(src_padded)).transpose(0, 1)
            tgt_var = torch.autograd.Variable(torch.LongTensor(tgt_padded)).transpose(0, 1)

            if self.USE_CUDA:
                src_var = src_var.cuda()
                tgt_var = tgt_var.cuda()
            # print(batch)
            # print(src_var, src_lengths, tgt_var, tgt_lengths)
            yield Batch(src_var, src_lengths, tgt_var, tgt_lengths), epoch_is_end

    # Pad a with the PAD symbol
    def _pad_seq(self, seq, max_length):
        seq += [self.PAD_TOKEN for i in range(max_length - len(seq))]
        return seq
