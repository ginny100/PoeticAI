import string
import numpy as np
import requests

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataset():
    def __init__(self):
        self.dataPath = None
        self.data_list = None
        self.input_sentences = []
        self.target_sentences = []
        self.encoder_input_data = []
        self.decoder_input_data = []
        self.decoder_target_data = []

    def download(self, url):
        """
        Download dataset from url and save to file
        """
        try:
            response = requests.get(url)
            response.raise_for_status()
            
            self.dataPath = 'dataset/truyenkieu.txt'
        
            with open(self.dataPath, 'wb') as f:
                for chunk in response.iter_content(chunk_size=1024):
                    if chunk:
                        f.write(chunk)
            
            print('Downloaded dataset to ', self.dataPath)
        
        except requests.exceptions.RequestException as e:
            print('Error downloading dataset:', e)
            return
    
    def load_data(self):
        """
        Read data from file and return a list of sentences
        """
        try:
            with open(self.dataPath, 'r', encoding='utf-8') as f:
                data = f.read()
            
            # Separate data sentence by sentence and remove blank sentences
            self.data_list = [line for line in data.split('\n') if line != '']
            # Display 10 first sentences
            # print(data_list[:10])
            return self.data_list
        
        except FileNotFoundError:
            print('File not found. Please download the dataset first.')
            return None
    
    def clean_data(self):
        """
        Clean data
        """
        # Convert to lowercase
        self.data_list = [x.lower() for x in self.data_list]
        # print(self.data_list[:10])
        
        # Remove punctuation
        remove_punc = str.maketrans('', '', string.punctuation)
        removed_punc_text = []
        
        for sent in self.data_list:
            sentence = [w.translate(remove_punc) for w in sent.split(' ')]
            removed_punc_text.append(' '.join(sentence))
        self.data_list = removed_punc_text
        # print(self.data_list[:10])
        
        # Remove digits
        remove_digits = str.maketrans('', '', string.digits)
        removed_digits_text = []
        
        for sent in self.data_list:
            sentence = [w.translate(remove_digits) for w in sent.split(' ')]
            removed_digits_text.append(' '.join(sentence))
        self.data_list = removed_digits_text
        # print(self.data_list[:10])
        
        # Remove starting and ending whitespaces
        self.data_list = [st.strip() for st in self.data_list]
        # print(self.data_list[:10])

        # Remove … and – characters
        self.data_list = [st.replace('...', '') for st in self.data_list]
        self.data_list = [st.replace('-', '') for st in self.data_list]

        # Check to see if 2 sentences are on the same line
        for ins in self.data_list:
            if len(ins.split()) > 8:
                print(ins)
        
        return self.data_list
    
    def split_data(self):
        """
        Split data into input and output sequences
        """
        for index, seq_txt in enumerate(self.data_list):
            if index % 2 == 0:
                self.input_sentences.append(seq_txt)
            else:
                self.target_sentences.append(seq_txt)
        
        self.target_sentences = ['start ' + ts + ' end' for ts in self.target_sentences]

        return self.input_sentences, self.target_sentences
    
    def build_tokenizer(self, sentences):
        """
        Build tokenizer for input and target sequences
        """
        # Prepare the tokenizer
        tokenizer = Tokenizer()
        tokenizer.fit_on_texts(sentences)
        # Determine the vocabulary size
        vocab_size = len(tokenizer.word_index) + 1
        print('Vocabulary size: %d' % vocab_size)
        # Create word2index dictionary
        word2index = tokenizer.word_index
        # Create index2word dictionary
        index2word = tokenizer.index_word

        return tokenizer, word2index, index2word
    
    def tokenize(self, tokenizer, sentences, padding='pre'):
        """
        Tokenize sentences
        """
        sequences = tokenizer.texts_to_sequences(sentences)
        max_len_sequences = max([len(seq) for seq in sequences])
        padded_sequences = pad_sequences(sequences, maxlen=max_len_sequences, padding=padding)
        return padded_sequences, max_len_sequences
    
    def create_outputs(self, input_word2index, target_word2index, max_len_input_seq=100, max_len_target_seq=100):
        """
        Create Encoder Input, Decoder Input, and Decoder Output
        """
        target_vocab_size = len(target_word2index) + 1

        self.encoder_input_data = np.zeros((len(self.input_sentences), max_len_input_seq), dtype='float32')
        # print(self.encoder_input_data.shape)
        self.decoder_input_data = np.zeros((len(self.input_sentences), max_len_target_seq), dtype='float32')
        # print(self.decoder_input_data.shape)
        self.decoder_target_data = np.zeros((len(self.input_sentences), max_len_target_seq, target_vocab_size), dtype='float32')
        # print(self.decoder_target_data.shape)

        for i, (input_text, target_text) in enumerate(zip(self.input_sentences, self.target_sentences)):
            for t, word in enumerate(input_text.split()):
                self.encoder_input_data[i, t] = input_word2index[word]
            for t, word in enumerate(target_text.split()):
                # decoder_target_data is ahead of decoder_input_data by one timestep
                self.decoder_input_data[i, t] = target_word2index[word]
                if t > 0:
                    # decoder_target_data will be ahead by one timestep and will not include the start character.
                    self.decoder_target_data[i, t - 1, target_word2index[word]] = 1
        
        return self.encoder_input_data, self.decoder_input_data, self.decoder_target_data