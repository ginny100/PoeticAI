from data import Dataset
from model import RNN

print("Step 1: Loading data...")

dataset = Dataset()
dataset.download(url='https://raw.githubusercontent.com/tiensu/Natural_Language_Processing/master/Text-Generation/dataset/truyenkieu.txt')
data_list = dataset.load_data()
# print(data_list[:20])
cleaned_data = dataset.clean_data()
# print(cleaned_data[:20])
input_sentences, target_sentences = dataset.split_data()
# print(input_sentences[:20])
# print(target_sentences[:20])
input_tokenizer, input_word2index, input_index2word = dataset.build_tokenizer(input_sentences)
target_tokenizer, target_word2index, target_index2word= dataset.build_tokenizer(target_sentences)
# print(input_tokenizer.word_index)
# print(target_tokenizer.word_index)
# print(input_word2index)
# print(target_word2index)
input_sequences, max_len_input_seq = dataset.tokenize(input_tokenizer, input_sentences)
target_sequences, max_len_target_seq = dataset.tokenize(target_tokenizer, target_sentences)
# print(input_sequences[:10])
# print(target_sequences[:10])
encoder_input_data, decoder_input_data, decoder_target_data = dataset.create_outputs(input_word2index, target_word2index, max_len_input_seq, max_len_target_seq)
# print("Encoder Input Data:")
# print(encoder_input_data)
# print("Decoder Input Data:")
# print(decoder_input_data)
# print("Decoder Target Data:")
# print(decoder_target_data)

print("Step 2: Training model...")

