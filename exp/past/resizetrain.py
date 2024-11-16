
import sys 
sys.path.append('.')
import torch
import torch.nn as nn
import torch.optim as optim 

from utils import default_args
from main import post_process_args
from datasets import get_dataloader
from networks.model_list import mobilenet

def main():
    args = default_args()
    args = post_process_args(args)
    for image_size in [16, 32]:
        print()
        print(f'===== image size is {image_size} ======')
        print()
        args.image_size = image_size
        train_loader, test_loader = get_dataloader(args)

        model = mobilenet(args)
        model.to(args.device)

        criterion = nn.CrossEntropyLoss()  
        optimizer = optim.Adam(model.parameters())

        # Step 3: Training loop
        num_epochs = 50
        for epoch in range(num_epochs):
            model.train()  # Set the model to training mode
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(args.device), labels.to(args.device)
                optimizer.zero_grad()
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()

                running_loss += loss.item() * images.size(0)
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            epoch_loss = running_loss / total
            train_accuracy = 100.0 * correct / total
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {train_accuracy:.2f}%')

            model.eval()  # Set the model to evaluation mode
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in test_loader:
                    images, labels = images.to(args.device), labels.to(args.device)

                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    val_loss += loss.item() * images.size(0)
                    _, predicted = outputs.max(1)
                    val_total += labels.size(0)
                    val_correct += predicted.eq(labels).sum().item()

                # Calculate validation loss and accuracy
                val_loss /= val_total
                val_accuracy = 100.0 * val_correct / val_total
                print(f'Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%')

if __name__ == '__main__':
    main()
