#!/usr/bin/python3
# -*- coding: utf-8 -*-

import torch
from torch.autograd import Variable
from torch import nn

#log constant
DEBUG_LOG = False
ST_LOG = False

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers=1, dropout=0.1):
        super(Encoder, self).__init__()

        # assert hidden_size%2 == 0, 'For bi-RNN hidden size must devide by 2'
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout = dropout
        if DEBUG_LOG : print("\nEncoder init <<<<<<<<<<<<<<<")
        if DEBUG_LOG : print("input_size = {}".format(input_size))
        if DEBUG_LOG : print("hidden_size = {}".format(hidden_size))
        if DEBUG_LOG : print("n_layers = {}".format(n_layers))
        if DEBUG_LOG : print("dropout = {}".format(dropout))

        self.embedding = nn.Embedding(input_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=self.dropout, bidirectional=True) # NOTE: Because we have bi-lstm, encoder output size is (SxBxhidden_size*2) [---><---]

    def forward(self, input_seqs, input_lengths, hidden=None):
        # Note: we run this all at once (over multiple batches of multiple sequences)
        if DEBUG_LOG : print("\nEncoder forward <<<<<<<<<<<<<<<")
        if DEBUG_LOG : print("input_seqs.size() = {}".format(input_seqs.size()))
        if DEBUG_LOG : print("input_lengths = {}".format(input_lengths))
        embedded = self.embedding(input_seqs)
        packed = nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        outputs, hidden = self.lstm(packed, hidden)
        outputs, output_lengths = nn.utils.rnn.pad_packed_sequence(outputs) # unpack (back to padded)
        if DEBUG_LOG : print(">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if DEBUG_LOG : print("outputs.size() = {}".format(outputs.size()))
        if DEBUG_LOG : print("len(hidden) = {}".format(len(hidden)))
        if DEBUG_LOG : print("hidden[0].size() = {}".format(hidden[0].size()))
        if DEBUG_LOG : print("hidden[1].size() = {}".format(hidden[1].size()))
        if DEBUG_LOG : print("output_lengths = {}".format(output_lengths))
        return outputs, hidden

class Attention(nn.Module):
    def __init__(self, method, hidden_size,
                    USE_CUDA = False):
        super(Attention, self).__init__()

        self.method = method
        self.hidden_size = hidden_size
        self.USE_CUDA = USE_CUDA
        if DEBUG_LOG : print("\nAttention init <<<<<<<<<<<<<<<")
        if DEBUG_LOG : print("method = {}".format(method))
        if DEBUG_LOG : print("hidden_size = {}".format(hidden_size))

        if self.method == 'general':
            self.attn = nn.Linear(self.hidden_size*2, hidden_size) # TODO: Remove hard code, that part for bi-lstm

        elif self.method == 'concat':
            self.attn = nn.Linear(self.hidden_size * 3, hidden_size) # TODO: Remove hard code, that part for bi-lstm
            self.v = nn.Parameter(torch.FloatTensor(1, hidden_size))

    def forward(self, hidden, encoder_outputs):
        max_len = encoder_outputs.size(0)
        this_batch_size = encoder_outputs.size(1)
        if DEBUG_LOG : print("\nAttention forward <<<<<<<<<<<<<<<")
        if DEBUG_LOG : print("hidden.size() = {}".format(hidden.size()))
        if DEBUG_LOG : print("encoder_outputs.size() = {}".format(encoder_outputs.size()))

        # Create variable to store attention energies
        attn_energies = Variable(torch.zeros(this_batch_size, max_len)) # B x S

        if self.USE_CUDA:
            attn_energies = attn_energies.cuda()

        # For each batch of encoder outputs
        for b in range(this_batch_size):
            # Calculate energy for each encoder output
            for i in range(max_len):
                attn_energies[b, i] = self.score(hidden[:, b], encoder_outputs[i, b].unsqueeze(0))

        # Normalize energies to weights in range 0 to 1, resize to 1 x B x S
        attention_weigth = torch.nn.functional.softmax(attn_energies, dim = 1).unsqueeze(1)
        if DEBUG_LOG : print("Attention forward >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if DEBUG_LOG : print("attention_weigth.size() = {}".format(attention_weigth.size()))
        return attention_weigth

    def score(self, hidden, encoder_output):

        if self.method == 'dot':
            energy = hidden.dot(encoder_output)
            return energy

        elif self.method == 'general':
            energy = self.attn(encoder_output)
            energy = hidden.dot(energy)
            return energy

        elif self.method == 'concat':
            energy = self.attn(torch.cat((hidden, encoder_output), 1))
            energy = self.v.dot(energy)
            return energy

class LuongAttentionDecoder(nn.Module):
    def __init__(self, attn_model, hidden_size, output_size, n_layers=1, dropout=0.1,
                    USE_CUDA = False):
        super(LuongAttentionDecoder, self).__init__()

        # Keep for reference
        self.attn_model = attn_model
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.dropout = dropout
        if DEBUG_LOG : print("\nLuongAttentionDecoder init <<<<<<<<<<<<<<<")
        if DEBUG_LOG : print("attn_model = {}".format(attn_model))
        if DEBUG_LOG : print("hidden_size = {}".format(hidden_size))
        if DEBUG_LOG : print("output_size = {}".format(output_size))
        if DEBUG_LOG : print("n_layers = {}".format(n_layers))
        if DEBUG_LOG : print("dropout = {}".format(dropout))

        # Define layers
        self.embedding = nn.Embedding(output_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size, hidden_size, n_layers, dropout=dropout)
        self.concat = nn.Linear(hidden_size * 3, hidden_size * 3)  # TODO: Remove hard code, that part for bi-lstm
        self.out = nn.Linear(hidden_size * 3, output_size)

        # Choose attention model
        if attn_model != 'none':
            self.attn = Attention(attn_model, hidden_size,USE_CUDA)

    def forward(self, input_seq, last_hidden, encoder_outputs):
        # Note: we run this one step at a time
        if DEBUG_LOG : print("\nLuongAttentionDecoder forward <<<<<<<<<<<<<<<")
        if DEBUG_LOG : print("input_seq.size() = {}".format(input_seq.size()))
        if DEBUG_LOG : print("len(last_hidden) = {}".format(len(last_hidden) if last_hidden else None))
        if DEBUG_LOG : print("last_hidden[0].size() = {}".format(last_hidden[0].size() if last_hidden else None))
        if DEBUG_LOG : print("last_hidden[1].size() = {}".format(last_hidden[1].size() if last_hidden else None))
        if DEBUG_LOG : print("encoder_outputs.size() = {}".format(encoder_outputs.size()))

        # Get the embedding of the current input word (last output word)
        batch_size = input_seq.size(0)
        embedded = self.embedding(input_seq)
        embedded = self.embedding_dropout(embedded)
        embedded = embedded.view(1, batch_size, self.hidden_size) # S=1 x B x N

        # Get current hidden state from input word and last hidden state
        rnn_output, hidden = self.lstm(embedded, last_hidden)

        # Calculate attention from current RNN state and all encoder outputs;
        # apply to encoder outputs to get weighted average
        attn_weights = self.attn(rnn_output, encoder_outputs)
        context = attn_weights.bmm(encoder_outputs.transpose(0, 1)) # B x S=1 x N
        if DEBUG_LOG : print("LuongAttentionDecoder forward ------------------")
        if DEBUG_LOG : print("context.size() = {}".format(context.size()))
        if DEBUG_LOG : print("LuongAttentionDecoder forward ------------------")

        # Attentional vector using the RNN hidden state and context vector
        # concatenated together (Luong eq. 5)
        rnn_output = rnn_output.squeeze(0) # S=1 x B x N -> B x N
        context = context.squeeze(1)       # B x S=1 x N -> B x N
        concat_input = torch.cat((rnn_output, context), 1)
        concat_output = torch.nn.functional.tanh(self.concat(concat_input))

        # Finally predict next token (Luong eq. 6, without softmax)
        output = self.out(concat_output)
        if DEBUG_LOG : print("LuongAttentionDecoder forward >>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>")
        if DEBUG_LOG : print("output.size() = {}".format(output.size()))
        if DEBUG_LOG : print("len(hidden) = {}".format(len(hidden)))
        if DEBUG_LOG : print("hidden[0].size() = {}".format(hidden[0].size()))
        if DEBUG_LOG : print("hidden[1].size() = {}".format(hidden[1].size()))
        if DEBUG_LOG : print("attn_weights.size() = {}".format(attn_weights.size()))

        # Return final output, hidden state, and attention weights (for visualization)
        return output, hidden, attn_weights
