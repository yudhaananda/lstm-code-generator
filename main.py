import torch
import torch.nn as nn
from transformers import PreTrainedModel, PretrainedConfig
import json
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.json
    seed_text = data['seed_text']

    # Call the generate_code function with the seed text and length
    generated_code = generate_code(seed_text, 8)  # Function you implemented

    return jsonify({"generated_code": generated_code})

if __name__ == "__main__":
    app.run(debug=True)

# Load the JSON file into a Python dictionary
with open('tokenizer.json', 'r') as file:
    token = json.load(file)

# Now `data_dict` is a dictionary with the contents of the JSON file
print(token)

word_to_idx = token['word_to_idx']  # Kamus kata ke indeks
idx_to_word = {i: word for word, i in word_to_idx.items()}
vocab_size=len(set(word_to_idx))

# Hyperparameters
hidden_size = 128

class LSTMModelConfig(PretrainedConfig):
    model_type = "LSTM"

    def __init__(self, vocab_size, hidden_size, num_layers=1, **kwargs):
        super().__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers

# 3. Model LSTM di PyTorch
class LSTMModel(PreTrainedModel):
    def __init__(self, config:LSTMModelConfig):
        super(LSTMModel, self).__init__(config)
        self.embedding = nn.Embedding(config.vocab_size, config.hidden_size)
        self.lstm = nn.LSTM(config.hidden_size, config.hidden_size, config.num_layers, batch_first=True)
        self.fc = nn.Linear(config.hidden_size, config.vocab_size)
    
    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out[:, -1, :])  # Mengambil output dari LSTM pada waktu terakhir
        return out, hidden

    def init_hidden(self, batch_size):
        # Inisialisasi hidden state dan cell state dengan nol
        return (torch.zeros(1, batch_size, config.hidden_size).to(device),
                torch.zeros(1, batch_size, config.hidden_size).to(device))
    
    def save_pretrained(self, save_directory):
        # Save model parameters
        torch.save(self.state_dict(), f"{save_directory}/pytorch_model.bin")

# 4. Melatih Model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

config= LSTMModelConfig(vocab_size, hidden_size)
model = LSTMModel(config=config).to(device)
model.load_state_dict(torch.load("./pytorch_model.bin", map_location=torch.device(device)))

def generate_code(seed_text, length):
    model.eval()
    generated = seed_text
    input_seq =  [word_to_idx[word] for word in seed_text.split()]
    input_seq = torch.tensor(input_seq, dtype=torch.long).unsqueeze(0).to(device)
    
    hidden = model.init_hidden(1)
    
    with torch.no_grad():
        count=0
        while True:
            output, hidden = model(input_seq, hidden)
            predicted_word_idx = torch.argmax(output).item()
            predicted_word = idx_to_word[predicted_word_idx]
            generated += ' '+predicted_word
            
            # Update input sequence with the predicted character
            input_seq = torch.cat([input_seq[:, 1:], torch.tensor([[predicted_word_idx]], dtype=torch.long).to(device)], dim=1)
            if count>=length or predicted_word == '}':
                break
            count+=1
    return generated
