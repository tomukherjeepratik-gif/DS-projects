"""
PyTorch Encoder-Decoder Architecture for Image Captioning
Combines a CNN Encoder (ResNet feature extractor) with an LSTM Decoder.
"""

import torch
import torch.nn as nn
import torchvision.models as models

class EncoderCNN(nn.Module):
    """
    CNN Encoder: Uses a pre-trained ResNet-50 backbone to extract high-level visual features
    and projects them into an embedding space of dimension `embed_size`.
    """
    def __init__(self, embed_size, train_CNN=False):
        super(EncoderCNN, self).__init__()
        self.train_CNN = train_CNN
        resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Freeze ResNet parameters if not fine-tuning
        for param in resnet.parameters():
            param.requires_grad = train_CNN
            
        # Replace the final FC classification layer with an embedding projection
        modules = list(resnet.children())[:-1]  # Remove linear classification layer
        self.resnet = nn.Sequential(*modules)
        self.embed = nn.Linear(resnet.fc.in_features, embed_size)
        self.ln = nn.LayerNorm(embed_size)
        self.relu = nn.ReLU()

    def forward(self, images):
        """
        Extract visual features from images.
        Input shape: (batch_size, 3, 224, 224)
        Output shape: (batch_size, embed_size)
        """
        with torch.set_grad_enabled(self.train_CNN):
            features = self.resnet(images)  # (batch_size, 2048, 1, 1)
        features = features.view(features.size(0), -1)  # (batch_size, 2048)
        features = self.ln(self.embed(features))
        return self.relu(features)


class DecoderRNN(nn.Module):
    """
    LSTM Decoder: Takes image features from the CNN Encoder as initial input,
    and autoregressively predicts word tokens using an LSTM network.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1, dropout=0.3):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0
        )
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, features, captions):
        """
        Forward pass for training.
        captions shape: (batch_size, max_seq_len)
        features shape: (batch_size, embed_size)
        """
        # Exclude <END> token from input captions for teacher forcing
        embeddings = self.dropout(self.embed(captions[:, :-1]))
        
        # Concatenate image features as the first timestep input token
        inputs = torch.cat((features.unsqueeze(1), embeddings), dim=1)
        
        hiddens, _ = self.lstm(inputs)  # (batch_size, max_seq_len, hidden_size)
        outputs = self.linear(hiddens)  # (batch_size, max_seq_len, vocab_size)
        return outputs

    def caption_image(self, image_features, vocabulary, max_length=20):
        """
        Greedy search decoding to generate caption for a single image feature vector.
        """
        result_caption = []
        states = None
        # image_features is (embed_size,), make it (1, 1, embed_size)
        inputs = image_features.unsqueeze(0).unsqueeze(0)

        for _ in range(max_length):
            hiddens, states = self.lstm(inputs, states) # hiddens: (1, 1, hidden_size)
            output = self.linear(hiddens)  # output: (1, 1, vocab_size)
            predicted = output.argmax(dim=-1) # predicted: (1, 1)
            predicted_idx = predicted.item()

            result_caption.append(predicted_idx)
            
            # Check for <END> token
            if vocabulary.itos.get(predicted_idx) == "<END>":
                break

            # Next input is the embedding of predicted word: (1, 1, embed_size)
            inputs = self.embed(predicted)

        return vocabulary.decode(result_caption)


class EncoderDecoderModel(nn.Module):
    """
    Combined Encoder-Decoder Model Architecture.
    """
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=1):
        super(EncoderDecoderModel, self).__init__()
        self.encoderCNN = EncoderCNN(embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoderCNN(images)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def generate_caption(self, image, vocabulary, max_length=20):
        """
        Generates a text description for a given preprocessed image tensor.
        """
        self.eval()
        with torch.no_grad():
            features = self.encoderCNN(image.unsqueeze(0))
            caption = self.decoderRNN.caption_image(features[0], vocabulary, max_length)
        return caption
