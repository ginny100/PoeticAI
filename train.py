import tensorflow as tf

from data import Dataset
from model import Seq2Seq

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
input_tokenizer, input_vocab_size, input_word2index, input_index2word = dataset.build_tokenizer(input_sentences)
target_tokenizer, target_vocab_size, target_word2index, target_index2word= dataset.build_tokenizer(target_sentences)
# print(input_tokenizer.word_index)
# print(target_tokenizer.word_index)
# print(input_word2index)
# print(target_word2index)
input_sequences, max_len_input_seq = dataset.tokenize(input_tokenizer, input_sentences)
target_sequences, max_len_target_seq = dataset.tokenize(target_tokenizer, target_sentences)
print("Train - max_len_input_seq", max_len_input_seq)
print("Train - max_len_target_seq", max_len_target_seq)
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

print("Seq2Seq Model:")
mySeq2Seq = Seq2Seq(encoder_input_data, decoder_input_data, input_vocab_size, target_vocab_size)
mySeq2Seq.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
history = mySeq2Seq.fit(
    x=[encoder_input_data, decoder_input_data],
    y=decoder_target_data,
    batch_size=64,
    epochs=500
)
model_json = mySeq2Seq.to_json()
with open("output/PoemGen.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
mySeq2Seq.save_weights("output/PoemGen_model_weight.h5")
print("Saved model to disk")
