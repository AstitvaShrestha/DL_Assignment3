import os
import pickle
from collections import Counter

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

from datasets import load_dataset
import spacy
from tqdm import tqdm


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
        print("Loading dataset...")
        self.dataset = load_dataset("bentrevett/multi30k", split=None)
        self.data = self.dataset[split]
        
        #--- Load spacy tokenizers for German and English
        self.de_tokenizer = spacy.load("de_core_news_sm",
                                disable=["parser", "tagger", "ner"]
                            )
        
        self.en_tokenizer = spacy.load("en_core_web_sm",
                                        disable=["parser", "tagger", "ner"]
                                    )

        # Build vocab
        # if split == "train":
        #     self.build_vocab()
        
        # else:
        #     self.load_vocab()
        if (os.path.exists("src_vocab.pkl")
            and
            os.path.exists("tgt_vocab.pkl")
        ):
            self.load_vocab()

        else:
            print("Building vocabulary...")
            self.build_vocab()
            

        self.examples = [] # List to hold processed examples (token indices)
        self.process_data() # Process data to convert sentences to token indices

    def tokenize_de(self, text):
        """
        Tokenizes German text using spacy.
        """
        return [token.text.lower() for token in self.de_tokenizer(text)]
    
    def tokenize_en(self, text):
        """
        Tokenizes English text using spacy.
        """
        return [token.text.lower() for token in self.en_tokenizer(text)]
    
    def build_vocab(self):
        """
        Builds the vocabulary mapping for src (de) and tgt (en), including:
        <unk>, <pad>, <sos>, <eos>
        """
        # TODO: Create the vocabulary dictionaries or torchtext Vocab equivalent

        # Count token frequencies
        src_counter = Counter()
        tgt_counter = Counter()

        # Iterate through the TRAIN split to build vocab
        for sample in tqdm(self.dataset['train'], desc="Building vocab"):
            
            src_text = sample['de']
            tgt_text = sample['en']

            src_tokens = self.tokenize_de(src_text)
            tgt_tokens = self.tokenize_en(tgt_text)

            src_counter.update(src_tokens)
            tgt_counter.update(tgt_tokens)

        # Initialize vocab with special tokens
        self.src_vocab = {
            self.UNK_TOKEN: 0,
            self.PAD_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }

        self.tgt_vocab = {
            self.UNK_TOKEN: 0,
            self.PAD_TOKEN: 1,
            self.SOS_TOKEN: 2,
            self.EOS_TOKEN: 3,
        }

        # Add tokens that meet the frequency threshold
        for token, freq in src_counter.items():

            if freq >= self.min_freq:
                self.src_vocab[token] = len(self.src_vocab)

        
        for token, freq in tgt_counter.items():
            
            if freq >= self.min_freq:
                self.tgt_vocab[token] = len(self.tgt_vocab)

        
        # Reverse vocab for decoding
        self.src_itos = {
            idx: token for token, idx in self.src_vocab.items()
        }

        self.tgt_itos = {
            idx: token for token, idx in self.tgt_vocab.items()
        }

        # Save vocab to disk for later use

        with open('src_vocab.pkl', 'wb') as f:
            pickle.dump((self.src_vocab, self.src_itos), f)

        with open('tgt_vocab.pkl', 'wb') as f:  
            pickle.dump((self.tgt_vocab, self.tgt_itos), f)

        print(f"Source vocab size: {len(self.src_vocab)}")
        print(f"Target vocab size: {len(self.tgt_vocab)}")

    def load_vocab(self):
        """
        Loads previously saved vocabularies.
        """

        with open('src_vocab.pkl', 'rb') as f:
            self.src_vocab, self.src_itos = pickle.load(f)

        with open('tgt_vocab.pkl', 'rb') as f:
            self.tgt_vocab, self.tgt_itos = pickle.load(f)

        print(f"Source vocab size: {len(self.src_vocab)}")
        print(f"Target vocab size: {len(self.tgt_vocab)}")

    def process_data(self):
        """
        Convert English and German sentences into integer token lists using
        spacy and the defined vocabulary. 
        """
        # TODO: Tokenize and convert words to indices
        
        for sample in tqdm(self.data, desc=f"Processing {self.split}"):

            # Get raw text
            src_text = sample['de']
            tgt_text = sample['en']

            # Tokenize
            src_tokens = self.tokenize_de(src_text)
            tgt_tokens = self.tokenize_en(tgt_text)

            # Truncate long sentences
            # Reserve space for <sos> and <eos>
            src_tokens = src_tokens[:self.max_len - 2]
            tgt_tokens = tgt_tokens[:self.max_len - 2]

            # Add <sos> and <eos>
            src_tokens = [self.SOS_TOKEN] + src_tokens + [self.EOS_TOKEN]
            tgt_tokens = [self.SOS_TOKEN] + tgt_tokens + [self.EOS_TOKEN]

            # Convert tokens → indices
            # Use <unk> if token missing
            src_indices = [
                self.src_vocab.get(token, self.src_vocab[self.UNK_TOKEN]) 
                for token in src_tokens
            ]

            tgt_indices = [
                self.tgt_vocab.get(token, self.tgt_vocab[self.UNK_TOKEN]) 
                for token in tgt_tokens
            ]


            # Conver to tensors
            src_tensors = torch.tensor(src_indices, dtype=torch.long)
            tgt_tensors = torch.tensor(tgt_indices, dtype=torch.long)

            # Store processed example
            self.examples.append((src_tensors, tgt_tensors))

    def __len__(self):
        """
        Returns total number of examples.
        """
        return len(self.examples)

    def __getitem__(self, idx):
        """
        Returns one processed example.

        Args:
            idx : integer index

        Returns:
            (src_tensor, tgt_tensor)
        """
        return self.examples[idx]
    

def collate_fn(batch):
    """
    Pads source and target sequences within a batch.

    Args:
        batch:
            list of tuples:
            [(src_tensor, tgt_tensor), ...]

    Returns:
        src_batch : [batch_size, src_len]
        tgt_batch : [batch_size, tgt_len]
    """
    
    # Separate source and target sequences
    src_batch = [item[0] for item in batch]
    tgt_batch = [item[1] for item in batch]

    # Pad sequences
    # padding_value=1 corresponds to <pad>
    src_batch = pad_sequence(src_batch, batch_first=True, padding_value=1)
    tgt_batch = pad_sequence(tgt_batch, batch_first=True, padding_value=1)

    return src_batch, tgt_batch