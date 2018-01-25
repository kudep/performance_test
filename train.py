#!/usr/bin/python3
# -*- coding: utf-8 -*-

import collections
import argparse
import os
import dataloader
import torch
import models
from masked_cross_entropy import *
import numpy as np
import math


import time

#log constant
DEBUG_LOG = False
ST_LOG = True

PAD_TOKEN = 0
SOS_TOKEN = 1
EOS_TOKEN = 2
USE_CUDA = True
MAX_LENGTH = 20

# TODO: Select methon of get hparams, select one of get_env_args and get_console_args
def args_preproc(args):
    #path
    args.datadir = os.path.join(args.work_dir, 'data')
    args.log_file = os.path.join(args.save_dir, 'logs.pkl')
    args.serialization_dir = os.path.join(args.save_dir, 'serialization')
    args.encoder_file = os.path.join(args.serialization_dir, 'encoder')
    args.decoder_file = os.path.join(args.serialization_dir, 'decoder')
    args.srcfile = os.path.join(args.datadir, 'src.txt')
    args.tgtfile = os.path.join(args.datadir, 'tgt.txt')
    return args

def get_env_args():
    args = collections.namedtuple('args',
        ['work_dir','model_name'
        ])
    args.work_dir = os.getenv('WORKDIR')
    args.model_name = os.getenv('MODEL_NAME')
    args = args_preproc(args)
    return args

def get_console_args():
    parser = argparse.ArgumentParser()
    # Configure directory path
    parser.add_argument('-d', '--work_dir', type=str)
    parser.add_argument('-n', '--model_name', type=str)
    parser.add_argument('-s', '--save_dir', type=str)

    # Configure data preprocessing
    parser.add_argument('--min_sentence_length', type=int, default=3)
    parser.add_argument('--max_sentence_length', type=int, default=80)

    # Configure models
    parser.add_argument('--attn_model', type=str, default='general')
    parser.add_argument('--hidden_size', type=int, default=500)
    parser.add_argument('--n_layers', type=int, default=2)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--batch_size', type=int, default=50)

    # Configure training/optimization
    parser.add_argument('--clip', type=float, default=50.0)
    parser.add_argument('--learning_rate', type=float, default=0.0001)
    parser.add_argument('--decoder_learning_ratio', type=float, default=5.0)
    parser.add_argument('--n_epochs', type=int, default=500)

    # Configure log
    parser.add_argument('--plot_every', type=int, default=1)
    parser.add_argument('--print_every', type=int, default=1)

    args = parser.parse_args()
    args_preproc(args)
    return args


def train(batch, encoder, decoder, encoder_optimizer, decoder_optimizer,
                criterion, batch_size = 50, clip = 50.0, max_length=MAX_LENGTH):

    input_batches, input_lengths, target_batches, target_lengths = batch
    batch_size = input_batches.size()[1]
    # Zero gradients of both optimizers
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss = 0 # Added onto for each word

    # Run words through encoder
    encoder_outputs, encoder_hidden = encoder(input_batches, input_lengths, None)

    # Prepare input and output variables
    decoder_input = Variable(torch.LongTensor([SOS_TOKEN] * batch_size))

    decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    decoder_hidden = None # TODO: Set last_hidden of decoder lstm

    max_target_length = max(target_lengths)
    all_decoder_outputs = Variable(torch.zeros(max_target_length, batch_size, decoder.output_size))

    # Move new Variables to CUDA
    if USE_CUDA:
        decoder_input = decoder_input.cuda()
        all_decoder_outputs = all_decoder_outputs.cuda()

    # Run through decoder one time step at a time
    for t in range(max_target_length):
        decoder_output, decoder_hidden, decoder_attn = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )

        all_decoder_outputs[t] = decoder_output
        decoder_input = target_batches[t] # Next input is current target

    # Loss calculation and backpropagation
    loss = masked_cross_entropy(
        all_decoder_outputs.transpose(0, 1).contiguous(), # -> batch x seq
        target_batches.transpose(0, 1).contiguous(), # -> batch x seq
        target_lengths
    )
    loss.backward()

    # Clip gradient norms
    ec = torch.nn.utils.clip_grad_norm(encoder.parameters(), clip)
    dc = torch.nn.utils.clip_grad_norm(decoder.parameters(), clip)

    # Update parameters with optimizers
    encoder_optimizer.step()
    decoder_optimizer.step()

    return loss.data[0], ec, dc

def as_day(s):
    m = math.floor(s / 60)
    s -= m * 60
    h = math.floor(m / 60)
    m -= h * 60
    d = math.floor(h /24)
    h-=d*24
    return '%dd %dh %dm %ds' % (d, h, m, s)

def time_since(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (as_day(s), as_day(rs))

def train_seconds(since):
    now = time.time()
    s = now - since
    return s

# def evaluate(input_seq, max_length=MAX_LENGTH):


def evaluate(batch, encoder, decoder,
              src_indexer, tgt_indexer,
              max_length=MAX_LENGTH):
    # input_lengths = [len(input_seq)]
    # input_seqs = [indexes_from_sentence(input_lang, input_seq)]

    input_batches, input_lengths, target_batches, target_lengths = batch
    input_batches = input_batches.view(input_batches.size()[0], 1, input_batches.size()[1]).transpose(0, 2)
    input_batch = input_batches[0].transpose(0, 1)
    input_lengths = input_lengths[:1]
    if USE_CUDA:
        input_batch = input_batch.cuda()

    # Set to not-training mode to disable dropout
    encoder.train(False)
    decoder.train(False)

    # Run through encoder
    encoder_outputs, encoder_hidden = encoder(input_batch, input_lengths, None)

    # Create starting vectors for decoder
    decoder_input = Variable(torch.LongTensor([SOS_TOKEN]), volatile=True) # SOS
    # decoder_hidden = encoder_hidden[:decoder.n_layers] # Use last (forward) hidden state from encoder
    decoder_hidden = None

    if USE_CUDA:
        decoder_input = decoder_input.cuda()

    # Store output words and attention states
    encoder_words = [src_indexer.index2word[int(input_batch[i][0])] for i in range(input_batch.size()[0])]
    decoded_words = []
    decoder_attentions = torch.zeros(max_length + 1, max_length + 1)

    # Run through decoder
    for di in range(max_length):
        decoder_output, decoder_hidden, decoder_attention = decoder(
            decoder_input, decoder_hidden, encoder_outputs
        )
        decoder_attentions[di,:decoder_attention.size(2)] += decoder_attention.squeeze(0).squeeze(0).cpu().data

        # Choose top word from output
        if DEBUG_LOG : print("decoder_output.size() = {}".format(decoder_output.size()))
        topv, topi = decoder_output.data.topk(1)
        if DEBUG_LOG : print("decoder_output.data.topk(1) = {}".format(decoder_output.data.topk(1)))
        ni = topi[0][0]
        if DEBUG_LOG : print("topi[0][0] = {}".format(topi[0][0]))
        if ni == EOS_TOKEN:
            decoded_words.append('<EOS>')
            break
        else:
            decoded_words.append(tgt_indexer.index2word[ni])

        # Next input is chosen word
        decoder_input = Variable(torch.LongTensor([ni]))
        if USE_CUDA: decoder_input = decoder_input.cuda()

    # Set back to training mode
    encoder.train(True)
    decoder.train(True)

    return encoder_words, decoded_words, decoder_attentions[:di+1, :len(encoder_outputs)]


def model_test(args):
    # Configure directory path
    work_dir = args.work_dir
    model_name = args.model_name
    save_dir = args.save_dir

    # Configure data preprocessing
    min_sentence_length = args.min_sentence_length
    max_sentence_length = args.max_sentence_length

    # Configure models
    attn_model = args.attn_model
    hidden_size = args.hidden_size
    n_layers = args.n_layers
    dropout = args.dropout
    batch_size = args.batch_size

    # Configure training/optimization
    clip = args.clip
    learning_rate = args.learning_rate
    decoder_learning_ratio = args.decoder_learning_ratio
    n_epochs = args.n_epochs

    # Configure log
    plot_every = args.plot_every
    print_every = args.print_every

    if ST_LOG : print("work_dir = {}".format(work_dir))
    if ST_LOG : print("model_name = {}".format(model_name))
    if ST_LOG : print("save_dir = {}".format(save_dir))
    if ST_LOG : print("min_sentence_length = {}".format(min_sentence_length))
    if ST_LOG : print("max_sentence_length = {}".format(max_sentence_length))
    if ST_LOG : print("attn_model = {}".format(attn_model))
    if ST_LOG : print("hidden_size = {}".format(hidden_size))
    if ST_LOG : print("n_layers = {}".format(n_layers))
    if ST_LOG : print("dropout = {}".format(dropout))
    if ST_LOG : print("batch_size = {}".format(batch_size))
    if ST_LOG : print("clip = {}".format(clip))
    if ST_LOG : print("learning_rate = {}".format(learning_rate))
    if ST_LOG : print("decoder_learning_ratio = {}".format(decoder_learning_ratio))
    if ST_LOG : print("n_epochs = {}".format(n_epochs))
    if ST_LOG : print("plot_every = {}".format(plot_every))
    if ST_LOG : print("print_every = {}".format(print_every))


    data = dataloader.Data(min_sentence_length, max_sentence_length,
                            USE_CUDA = USE_CUDA, PAD_TOKEN = PAD_TOKEN,
                            EOS_TOKEN = EOS_TOKEN)
    data.dataload(args.srcfile, args.tgtfile)
    generator = data.batch_generator(batch_size)
    src_indexer = data.src_indexer
    tgt_indexer = data.tgt_indexer

    # Initialize models
    encoder = models.Encoder(src_indexer.n_words, hidden_size, n_layers, dropout=dropout)
    decoder = models.LuongAttentionDecoder(attn_model, hidden_size,
            tgt_indexer.n_words, n_layers, dropout=dropout, USE_CUDA = USE_CUDA)
    # Initialize optimizers and criterion
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=learning_rate * decoder_learning_ratio)
    criterion = torch.nn.CrossEntropyLoss()

    # Move models to GPU
    if USE_CUDA:
        encoder.cuda()
        decoder.cuda()



    start_time = time.time()
    plot_losses = []
    ecs = []
    dcs = []


    # args.serialization_dir
    # Begin!
    print_loss_total = 0 # Reset every print_every
    plot_loss_total = 0 # Reset every plot_every
    eca = 0
    dca = 0

    step = 0
    epoch = 0
    while True:
        # Get training data for this cycle
        step += 1
        batch, epoch_is_end = next(generator)
        if epoch_is_end:
            epoch += 1
        if epoch >= n_epochs:
            break

        # Run the train function
        loss, ec, dc = train(
            batch,
            encoder, decoder,
            encoder_optimizer,
            decoder_optimizer,
            criterion, clip
        )

        # Keep track of loss
        print_loss_total += loss
        plot_loss_total += loss
        eca += ec
        dca += dc


        if step % print_every == 0:
            print_loss_avg = print_loss_total / print_every
            print_loss_total = 0
            if epoch == 0:
                print_summary = '%s (steps = %d progress = %d%%) %.4f' % (0, step, epoch / n_epochs * 100, print_loss_avg)
            else:
                print_summary = '%s (steps = %d progress = %d%%) %.4f' % (time_since(start_time, epoch / n_epochs), step, epoch / n_epochs * 100, print_loss_avg)
            if ST_LOG : print(print_summary)

        if step % plot_every == 0:
            plot_loss_avg = plot_loss_total / plot_every
            plot_losses.append(plot_loss_avg)
            plot_loss_total = 0
            encoder_words, decoded_words, attention = evaluate(batch, encoder, decoder,
                                                src_indexer, tgt_indexer)
            if DEBUG_LOG : print("encoder_words = {}".format(encoder_words))
            if DEBUG_LOG : print("decoded_words = {}".format(decoded_words))
            if DEBUG_LOG : print("attention.size() = {}".format(attention.size()))
            # TODO: Running average helper
            ecs.append(eca / plot_every)
            dcs.append(dca / plot_every)
            eca = 0
            dca = 0




def main():
    # args = get_env_args()
    args = get_console_args()

    model_test(args)

if __name__ == '__main__':
    main()
