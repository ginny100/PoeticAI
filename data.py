import string
import requests

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

class Dataset():
    def __init__(self):
        self.dataPath = None
        self.data_list = None
        self.input_sentences = []
        self.target_sentences = []

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
            
            # separate data sentence by sentence and remove blank sentences
            self.data_list = [line for line in data.split('\n') if line != '']
            # display 10 first sentences
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
    
    def build_tokenizer(self, sentences, max_words=10000):
        """
        Build tokenizer for input and target sequences
        """
        tokenizer = Tokenizer(num_words=max_words, oov_token='OOV')
        tokenizer.fit_on_texts(sentences)

        return tokenizer
    
    def tokenize(self, tokenizer, sentences, padding='pre'):
        """
        Tokenize sentences
        """
        sequences = tokenizer.texts_to_sequences(sentences)
        max_len_sequences = max([len(seq) for seq in sequences])
        padded_sequences = pad_sequences(sequences, maxlen=max_len_sequences, padding=padding)
        return padded_sequences

#################

dataset = Dataset()
dataset.download(url='https://raw.githubusercontent.com/tiensu/Natural_Language_Processing/master/Text-Generation/dataset/truyenkieu.txt')
data_list = dataset.load_data()
# print(data_list[:20])
cleaned_data = dataset.clean_data()
# print(cleaned_data[:20])
input_sentences, target_sentences = dataset.split_data()
# print(input_sentences[:20])
# print(target_sentences[:20])
input_tokenizer, target_tokenizer = dataset.build_tokenizer(input_sentences), dataset.build_tokenizer(target_sentences)
# print(input_tokenizer.word_index)
# print(target_tokenizer.word_index)
input_sequences, target_sequences = dataset.tokenize(input_tokenizer, input_sentences), dataset.tokenize(target_tokenizer, target_sentences)
print(input_sequences[:10])
print(target_sequences[:10])