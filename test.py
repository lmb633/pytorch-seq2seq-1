import torch
import torch.nn as nn
from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator
import spacy
import random
import math
import os
import time

pad_idx = 1

def map_val(a, b):
    if a is True:
        return 1
    return 0
def make_masks(src, trg):
    # src = [batch size, src sent len]
    # trg = [batch size, trg sent len]

    src_mask = (src != pad_idx).unsqueeze(1).unsqueeze(2)


    trg_pad_mask = (trg != pad_idx).unsqueeze(1).unsqueeze(3)
    print(src_mask)
    print(trg_pad_mask)
    print(src_mask.shape)
    print(trg_pad_mask.shape)

    trg_len = trg.shape[1]

    trg_sub_mask = torch.tril(torch.ones((trg_len, trg_len), dtype=torch.uint8))
    print(trg_sub_mask.shape)
    print(trg_sub_mask)
    trg_mask = trg_pad_mask & trg_sub_mask

    return src_mask, trg_mask


def get_data():
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

    BATCH_SIZE = 2

    train_iterator, valid_iterator, test_iterator = BucketIterator.splits(
        (train_data, valid_data, test_data),
        batch_size=BATCH_SIZE,
        device=device)
    return train_iterator, valid_iterator, test_iterator


def test(train_iterator):
    for i, batch in enumerate(train_iterator):
        if i < 0:
            src = batch.src
            trg = batch.trg
            print(i, batch)
            print(src)
            print(trg)
            mask1, mask2 = make_masks(src, trg)
            # print(mask1, mask2)
            print(mask1.shape, mask2.shape)
            length = mask1.shape[3]
            src = torch.rand((1, 1, length, length))
            length = mask2.shape[3]
            trg = torch.rand((1, 1, length, length))
            result1 = src.masked_fill(mask1 == 0, -1e10)
            result2 = trg.masked_fill(mask2 == 0, -1e10)
            print(result1)
            print(result2)


src = torch.rand((2, 6))


def map_val(a, b):
    if a is True:
        return 1
    return 0


print(src.map_(src, map_val))

# src[0][4] = 1
# src[0][5] = 1
# src[1][5] = 1
# print(src)
# mask1, mask2 = make_masks(src, src)
# src = torch.rand((1, 1, 6, 6))
# result1 = src.masked_fill(mask1 == 0, -1e10)
# result2 = src.masked_fill(mask2 == 0, -1e10)
# print(result1)
# print(result2)
