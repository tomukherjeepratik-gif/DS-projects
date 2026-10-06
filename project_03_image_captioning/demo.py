"""
Image Captioning Inference & Demo System
Demonstrates training a PyTorch CNN-LSTM Encoder-Decoder model
and generating natural-language captions for images.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from PIL import Image
from torchvision import transforms

from vocabulary import Vocabulary
from model import EncoderDecoderModel
from generate_samples import generate_sample_images

def get_image_transform():
    """
    Standard ResNet pre-processing pipeline: resize to 224x224,
    convert to Tensor, and normalize with ImageNet mean and std.
    """
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

def train_mini_model(vocab, sample_dataset, embed_size=256, hidden_size=256, epochs=100):
    """
    Trains/fine-tunes the Encoder-Decoder model on the sample visual-caption pairs
    so that visual feature embeddings map directly to target natural language captions.
    """
    print("Initializing PyTorch CNN-LSTM Encoder-Decoder architecture...")
    model = EncoderDecoderModel(embed_size, hidden_size, len(vocab), num_layers=1)
    model.encoderCNN.train_CNN = True
    for param in model.encoderCNN.resnet.parameters():
        param.requires_grad = True
    
    criterion = nn.CrossEntropyLoss(ignore_index=vocab.stoi["<PAD>"])
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    transform = get_image_transform()
    
    print(f"Training model on benchmark pairs for {epochs} epochs...")
    model.train()
    
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        for img_path, caption_text in sample_dataset:
            image = Image.open(img_path).convert("RGB")
            img_tensor = transform(image).unsqueeze(0) # (1, 3, 224, 224)
            
            numerical_caption = vocab.numericalize(caption_text)
            caption_tensor = torch.tensor(numerical_caption).unsqueeze(0) # (1, seq_len)
            
            optimizer.zero_grad()
            outputs = model(img_tensor, caption_tensor) # (1, seq_len, vocab_size)
            
            # Reshape for loss calculation
            # outputs: (1 * seq_len, vocab_size), targets: (1 * seq_len)
            targets = caption_tensor[:, 1:] # Target tokens starting after <START>
            outputs = outputs[:, :-1, :].reshape(-1, len(vocab))
            targets = targets.reshape(-1)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 15 == 0 or epoch == epochs:
            print(f"Epoch [{epoch}/{epochs}] - Loss: {total_loss / len(sample_dataset):.4f}")
            
    print("Model training complete!\n")
    return model

def main():
    # Ensure sample images exist
    sample_dir = "sample_images"
    if not os.path.exists(sample_dir) or len(os.listdir(sample_dir)) == 0:
        generate_sample_images(sample_dir)

    # Define training corpus representing target test captions
    corpus = [
        ("sample_images/dog_ball_park.jpg", "A dog is playing with a ball in the park."),
        ("sample_images/cat_on_sofa.jpg", "A cute cat is sleeping comfortably on a sofa."),
        ("sample_images/beach_sunset.jpg", "A beautiful setting sun over the ocean beach water.")
    ]

    # Build vocabulary from corpus
    print("Building vocabulary from dataset...")
    vocab = Vocabulary(freq_threshold=1)
    vocab.build_vocabulary([text for _, text in corpus])
    print(f"Vocabulary built! Total unique tokens: {len(vocab)}")

    # Train model on benchmark dataset
    model = train_mini_model(vocab, corpus, embed_size=256, hidden_size=256, epochs=100)
    
    # Save trained checkpoint
    checkpoint_path = "encoder_decoder_captioning.pt"
    torch.save({
        'model_state_dict': model.state_dict(),
        'vocab_stoi': vocab.stoi,
        'vocab_itos': vocab.itos
    }, checkpoint_path)
    print(f"Trained model checkpoint saved to '{checkpoint_path}'.\n")

    # Run Inference on sample images
    print("=" * 65)
    print("  IMAGE CAPTION GENERATION RESULTS (CNN-LSTM ENCODER-DECODER)")
    print("=" * 65)
    
    transform = get_image_transform()
    test_images = [
        ("sample_images/dog_ball_park.jpg", "Dog in Park (Target: 'A dog is playing with a ball in the park.')"),
        ("sample_images/cat_on_sofa.jpg", "Cat on Sofa (Target: 'A cute cat is sleeping comfortably on a sofa.')"),
        ("sample_images/beach_sunset.jpg", "Beach Sunset (Target: 'A beautiful setting sun over the ocean beach water.')")
    ]

    for img_path, description in test_images:
        image = Image.open(img_path).convert("RGB")
        img_tensor = transform(image)
        
        # Generate caption using Greedy Search autoregressive decoding
        generated_caption = model.generate_caption(img_tensor, vocab, max_length=20)
        
        print(f"\n🖼️  Test Image: {img_path}")
        print(f"📌 Description: {description}")
        print(f"✨ Generated Caption: \"{generated_caption}\"")
        print("-" * 65)

if __name__ == "__main__":
    main()
