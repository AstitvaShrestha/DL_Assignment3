import pickle
from collections import Counter

import torch
from torch.utils.data import Dataset

from datasets import load_dataset
import spacy


class Multi30kDataset(Dataset):
    def __init__(self, split='train'):
        """
        Loads the Multi30k dataset and prepares tokenizers.
        """
        self.split = split
        # Load dataset from Hugging Face
        # https://huggingface.co/datasets/bentrevett/multi30k
        # TODO: Load dataset, load spacy tokenizers for de and en
        
        # --Internal config----
        self.max_len = 100  # Max sentence length for padding/truncation
        self.min_freq = 2  # Minimum frequency for a word to be included in the vocab

        # --- Special tokens----
        self.UNK_TOKEN = "<unk>"
        self.PAD_TOKEN = "<pad>"
        self.SOS_TOKEN = "<sos>"
        self.EOS_TOKEN = "<eos>"

        # Load dataset
        self.dataset = load_dataset("multi30k")
        self.data = self.dataset[split]
        
        #--- Load spacy tokenizers for German and English
        self.de_tokenizer = spacy.load("de_core_news_sm")
        self.en_tokenizer = spacy.load("en_core_web_sm")

        if split == "train":
            self.build_vocab()

        self.process_data()

    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent
        raise NotImplementedError

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        raise NotImplementedError