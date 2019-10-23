import torch
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

from model import Encoder, Decoder, Seq2Seq, NoamOpt, DecoderLayer, EncoderLayer, SelfAttention, PositionwiseFeedforward

import spacy
import random
import math
import os
import time

SEED = 1

random.seed(SEED)
torch.manual_seed(SEED)
# torch.backends.cudnn.deterministic = True

spacy_de = spacy.load('de')
spacy_en = spacy.load('en')


def tokenize_de(text):
    """
    Tokenizes German text from a string into a list of strings
    """
    return [tok.text for tok in spacy_de.tokenizer(text)]


def tokenize_en(text):
    """
    Tokenizes English text from a string into a list of strings
    """
    return [tok.text for tok in spacy_en.tokenizer(text)]


SRC = Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)
TRG = Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', lower=True, batch_first=True)

train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG))

SRC.build_vocab(train_data, min_freq=2)
TRG.build_vocab(train_data, min_freq=2)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

BATCH_SIZE = 128

train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
    (train_data, valid_data, test_data),
    batch_size=BATCH_SIZE,
    device=device)

input_dim = len(SRC.vocab)
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1

enc = Encoder(input_dim, hid_dim, n_layers, n_heads, pf_dim, EncoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

output_dim = len(TRG.vocab)
hid_dim = 512
n_layers = 6
n_heads = 8
pf_dim = 2048
dropout = 0.1

dec = Decoder(output_dim, hid_dim, n_layers, n_heads, pf_dim, DecoderLayer, SelfAttention, PositionwiseFeedforward, dropout, device)

pad_idx = SRC.vocab.stoi['<pad>']

model = Seq2Seq(enc, dec, pad_idx, device).to(device)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')

for p in model.parameters():
    if p.dim() > 1:
        nn.init.xavier_uniform_(p)

optimizer = NoamOpt(hid_dim, 1, 2000,
                    torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))

criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)


def train(model, iterator, optimizer, criterion, clip):
    model.train()

    epoch_loss = 0

    for i, batch in enumerate(iterator):
        src = batch.src
        trg = batch.trg

        optimizer.optimizer.zero_grad()

        output = model(src, trg[:, :-1])

        # output = [batch size, trg sent len - 1, output dim]
        # trg = [batch size, trg sent len]

        output = output.contiguous().view(-1, output.shape[-1])
        trg = trg[:, 1:].contiguous().view(-1)

        # output = [batch size * trg sent len - 1, output dim]
        # trg = [batch size * trg sent len - 1]

        loss = criterion(output, trg)

        loss.backward()

        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        optimizer.step()

        epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def evaluate(model, iterator, criterion):
    model.eval()

    epoch_loss = 0

    with torch.no_grad():
        for i, batch in enumerate(iterator):
            src = batch.src
            trg = batch.trg

            output = model(src, trg[:, :-1])

            # output = [batch size, trg sent len - 1, output dim]
            # trg = [batch size, trg sent len]

            output = output.contiguous().view(-1, output.shape[-1])
            trg = trg[:, 1:].contiguous().view(-1)

            # output = [batch size * trg sent len - 1, output dim]
            # trg = [batch size * trg sent len - 1]

            loss = criterion(output, trg)

            epoch_loss += loss.item()

    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1
SAVE_DIR = 'models'
MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'transformer-seq2seq.pt')

best_valid_loss = float('inf')

if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iterator, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iterator, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    if valid_loss < best_valid_loss:
        best_valid_loss = valid_loss
        torch.save(model.state_dict(), MODEL_SAVE_PATH)

    print(
        f'| Epoch: {epoch+1:03} | Time: {epoch_mins}m {epoch_secs}s| Train Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f} | Val. Loss: {valid_loss:.3f} | Val. PPL: {math.exp(valid_loss):7.3f} |')

model.load_state_dict(torch.load(MODEL_SAVE_PATH))

test_loss = evaluate(model, test_iterator, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')
