import tensorflow as tf


class Bidirectional(tf.keras.Model):
  def __init__(self, lstm1, lstm2):
    super(Bidirectional, self).__init__()
    self.lstm1 = lstm1
    self.lstm2 = lstm2

  def call(self):
    # Left to right


    # Right to left
    pass