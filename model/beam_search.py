import numpy as np
import tensorflow as tf

class BeamSearchNode:
    """
    Node in beam search
    Each beam is represented as a node which contains
    its hidden state, previous node, word id, log probability, and length.
    """
    def __init__(self, hiddenstate, previousNode, wordId, logProb, length):
        self.h = hiddenstate[0]  # LSTM hidden state
        self.c = hiddenstate[1]  # LSTM cell state
        self.prevNode = previousNode  # parent beam
        self.wordid = wordId  # word ID
        self.logp = logProb  # log probability
        self.leng = length  # length of the sequence

    def eval(self, alpha=1.0):
        """
        Evaluate the node using length normalization
        
        Args:
            alpha: Length normalization parameter (higher for longer sequences)
        """
        # Length normalization to avoid bias towards shorter sequences
        return self.logp / float(self.leng - 1 + 1e-6)**alpha

    def __lt__(self, other):
        return self.logp < other.logp


class BeamSearch:
    """
    Beam Search implementation for sequence generation
    """
    def __init__(self, encoder_model, decoder_model, target_word2index, target_index2word):
        """
        Initialize beam search with models and token dictionaries
        
        Args:
            encoder_model: Model that encodes input sequence
            decoder_model: Model that generates output sequence
            target_word2index: Dictionary mapping words to indices
            target_index2word: Dictionary mapping indices to words
        """
        self.encoder_model = encoder_model
        self.decoder_model = decoder_model
        self.target_word2index = target_word2index
        self.target_index2word = target_index2word
        self.start_token = target_word2index.get('start', 1)
        self.end_token = target_word2index.get('end', 0)
    
    def decode(self, input_seq, beam_width=5, max_length=50, alpha=0.7):
        """
        General beam search decoding
        
        Args:
            input_seq: Input sequence to decode
            beam_width: Number of beams to keep at each step
            max_length: Maximum sequence length
            alpha: Length normalization parameter
        
        Returns:
            The best decoded sequence
        """
        # Encode the input as state vectors
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        
        # Create initial node with start token
        node = BeamSearchNode(
            hiddenstate=states_value,
            previousNode=None,
            wordId=self.start_token,
            logProb=0.0,
            length=1
        )
        
        # Initial beam has only one node
        beam = [node]
        complete_seqs = []
        complete_seqs_scores = []
        
        # Start beam search
        for step in range(max_length):
            # Store all candidates for this step
            candidates = []
            
            # For each beam
            for node in beam:
                # Get the previous word ID and hidden state
                word_id = node.wordid
                h = node.h
                c = node.c
                
                # If we reached EOS or length limit
                if word_id == self.end_token or step >= max_length:
                    complete_seqs.append(node)
                    complete_seqs_scores.append(node.eval(alpha))
                    continue
                
                # Prepare input for decoder
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = word_id
                
                # Get predictions for next word
                # Make sure h and c are properly shaped for the decoder input
                h_reshaped = np.reshape(h, (1, -1))  # Shape: (1, units)
                c_reshaped = np.reshape(c, (1, -1))  # Shape: (1, units)
                
                output_tokens, h_state, c_state = self.decoder_model.predict(
                    [target_seq, h_reshaped, c_reshaped], 
                    verbose=0
                )
                
                # Get top K tokens
                log_probs = np.log(output_tokens[0, 0])
                top_k_indices = np.argsort(-log_probs)[:beam_width]
                
                # For each of the top K tokens
                for new_word_id in top_k_indices:
                    next_log_prob = log_probs[new_word_id]
                    
                    # Skip unknown words and padding for beam expansion
                    if new_word_id == 0 or self.target_index2word.get(new_word_id) is None:
                        continue
                    
                    # Create new beam node
                    new_node = BeamSearchNode(
                        hiddenstate=[h_state[0], c_state[0]],
                        previousNode=node,
                        wordId=new_word_id,
                        logProb=node.logp + next_log_prob,
                        length=node.leng + 1
                    )
                    candidates.append(new_node)
            
            # No candidates? End search
            if len(candidates) == 0:
                break
            
            # Select top candidates for next beam
            candidates.sort(key=lambda x: x.eval(alpha), reverse=True)
            beam = candidates[:beam_width]
        
        # If we have complete sequences, pick the best one
        if complete_seqs:
            best_score = max(complete_seqs_scores)
            best_idx = complete_seqs_scores.index(best_score)
            best_node = complete_seqs[best_idx]
        else:
            # Otherwise, take the best from remaining beam
            best_node = max(beam, key=lambda x: x.eval(alpha))
        
        # Backtrack to get the sequence
        return self._get_sequence(best_node)
    
    def decode_fixed_length(self, input_seq, output_length=8, beam_width=5, alpha=0.7):
        """
        Beam search for fixed length output
        
        Args:
            input_seq: Input sequence to decode
            output_length: Desired output length
            beam_width: Number of beams to keep at each step
            alpha: Length normalization parameter
        
        Returns:
            A sequence with exactly output_length words
        """
        # Encode the input as state vectors
        states_value = self.encoder_model.predict(input_seq, verbose=0)
        
        # Initialize beam search
        node = BeamSearchNode(
            hiddenstate=states_value,
            previousNode=None,
            wordId=self.start_token,
            logProb=0.0,
            length=1
        )
        
        # Initial beam has only one node
        beam = [node]
        complete_seqs = []
        complete_seqs_scores = []
        
        # Start beam search
        for step in range(output_length + 1):  # +1 for the start token
            # Store all candidates for this step
            candidates = []
            
            # For each beam
            for node in beam:
                # Get the previous word ID and hidden state
                word_id = node.wordid
                h = node.h
                c = node.c
                
                # If we reached exact target length
                if node.leng == output_length + 1:  # +1 for the start token
                    complete_seqs.append(node)
                    complete_seqs_scores.append(node.eval(alpha))
                    continue
                
                # If we reached EOS token
                if word_id == self.end_token:
                    continue
                
                # Prepare input for decoder
                target_seq = np.zeros((1, 1))
                target_seq[0, 0] = word_id
                
                # Make sure h and c are properly shaped for the decoder input
                h_reshaped = np.reshape(h, (1, -1))  # Shape: (1, units)
                c_reshaped = np.reshape(c, (1, -1))  # Shape: (1, units)
                
                # Get predictions for next word
                output_tokens, h_state, c_state = self.decoder_model.predict(
                    [target_seq, h_reshaped, c_reshaped], 
                    verbose=0
                )
                
                # Get log probabilities
                log_probs = np.log(output_tokens[0, 0])
                top_k_indices = np.argsort(-log_probs)[:beam_width]
                
                # For each of the top K tokens
                for new_word_id in top_k_indices:
                    next_log_prob = log_probs[new_word_id]
                    
                    # Skip unknown words and padding
                    if new_word_id == 0 or self.target_index2word.get(new_word_id) is None:
                        continue
                    
                    # Create new beam node
                    new_node = BeamSearchNode(
                        hiddenstate=[h_state[0], c_state[0]],
                        previousNode=node,
                        wordId=new_word_id,
                        logProb=node.logp + next_log_prob,
                        length=node.leng + 1
                    )
                    candidates.append(new_node)
            
            # No candidates? End search
            if len(candidates) == 0:
                break
            
            # Sort candidates by score
            candidates.sort(key=lambda x: x.eval(alpha), reverse=True)
            beam = candidates[:beam_width]
        
        # If we have complete sequences, pick the best one
        if complete_seqs:
            best_score = max(complete_seqs_scores)
            best_idx = complete_seqs_scores.index(best_score)
            best_node = complete_seqs[best_idx]
        else:
            # Otherwise sort beam by length first, then score
            valid_beams = [b for b in beam if b.leng <= output_length + 1]
            if valid_beams:
                valid_beams.sort(key=lambda x: (-(output_length + 1 - x.leng), -x.eval(alpha)))
                best_node = valid_beams[0]
            else:
                best_node = max(beam, key=lambda x: x.eval(alpha))
        
        # Backtrack to get the sequence
        sequence = self._get_sequence(best_node)
        
        # Ensure exactly output_length words
        words = [w for w in sequence.split() if w]
        if len(words) > output_length:
            words = words[:output_length]
        
        return ' '.join(words)
    
    def _get_sequence(self, node):
        """
        Backtrack from final node to get the sequence
        
        Args:
            node: Final node in the beam
        
        Returns:
            String with the decoded sequence
        """
        sequence = []
        current_node = node
        
        while current_node.prevNode:
            sequence.append(current_node.wordid)
            current_node = current_node.prevNode
        
        # Reverse the sequence and convert to words
        sequence.reverse()
        
        return ' '.join([
            self.target_index2word.get(idx, '') 
            for idx in sequence 
            if idx != 0 and 
               self.target_index2word.get(idx) != 'start' and 
               self.target_index2word.get(idx) != 'end'
        ])
    
    def generate_poem(self, input_seq, max_length=50, beam_width=5):
        """
        Generate a variable-length poem
        
        Args:
            input_seq: Processed input sequence
            max_length: Maximum sequence length
            beam_width: Beam width
        
        Returns:
            Generated poem
        """
        return self.decode(input_seq, beam_width, max_length)
    
    def generate_fixed_length_poem(self, input_seq, output_length=8, beam_width=5):
        """
        Generate a fixed-length poem
        
        Args:
            input_seq: Processed input sequence
            output_length: Desired number of words in output
            beam_width: Beam width
        
        Returns:
            Generated poem with exactly output_length words (when possible)
        """
        return self.decode_fixed_length(input_seq, output_length, beam_width) 