from transformers import AutoTokenizer, AutoModel
import torch
import numpy as np

# Load data2vec-text-base model and tokenizer
model_name = "./data2vec-text-base_v2.pt"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModel.from_pretrained(model_name)

def get_text_embeddings(text, word_timings, sampling_rate=30):
    """
    Generate text embeddings aligned to the word timings at the specified sampling rate.

    Args:
        text (str): Input text sequence.
        word_timings (list of tuples): List of (word, start_time, end_time) for each word in the text.
        sampling_rate (int): Sampling rate in Hz for replicating embeddings. Defaults to 30 Hz.

    Returns:
        np.ndarray: 2D array of shape (total_frames, 768) containing the text-embedding sequence.
    """
    # Tokenize input text
    tokens = tokenizer(text, return_tensors="pt")

    # Get the last hidden layer of the model
    with torch.no_grad():
        outputs = model(**tokens)
    last_hidden_states = outputs.last_hidden_state  # Shape: (1, seq_len, 768)

    # Convert to numpy for further processing
    token_embeddings = last_hidden_states.squeeze(0).cpu().numpy()

    # Map token embeddings to word timings
    text_embedding_sequence = []
    for word, start_time, end_time in word_timings:
        duration = end_time - start_time
        num_frames = int(duration * sampling_rate)

        # Find the token index for the word
        word_tokens = tokenizer(word, add_special_tokens=False).input_ids
        word_token_embeddings = [token_embeddings[idx] for idx in range(len(word_tokens))]

        # Average embeddings for all tokens of the word
        avg_word_embedding = np.mean(word_token_embeddings, axis=0)

        # Replicate the averaged embedding for the duration of the word
        text_embedding_sequence.extend([avg_word_embedding] * num_frames)

    return np.array(text_embedding_sequence)

# Example usage
if __name__ == "__main__":
    input_text = "This is a sample sentence."
    # Example word timings (word, start_time, end_time in seconds)
    word_timings = [
        ("This", 0.0, 0.3),
        ("is", 0.3, 0.5),
        ("a", 0.5, 0.6),
        ("sample", 0.6, 1.0),
        ("sentence", 1.0, 1.5)
    ]

    embeddings = get_text_embeddings(input_text, word_timings, sampling_rate=30)
    print("Generated text embeddings of shape:", embeddings.shape)
