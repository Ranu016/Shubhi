import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

# Keep the dataset on the CPU; move each batch to the selected device.
inputs = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
targets = torch.tensor([[3.0], [5.0], [7.0], [9.0]])

dataset = TensorDataset(inputs, targets)
data_loader = DataLoader(dataset, batch_size=2, shuffle=True)

model = nn.Linear(in_features=1, out_features=1).to(device)
loss_function = nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Train the model.
for epoch in range(501):
    total_loss = 0.0

    for batch_inputs, batch_targets in data_loader:
        batch_inputs = batch_inputs.to(device)
        batch_targets = batch_targets.to(device)

        predictions = model(batch_inputs)
        loss = loss_function(predictions, batch_targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * batch_inputs.size(0)

    average_loss = total_loss / len(dataset)
    if epoch % 100 == 0:
        print(f"Epoch {epoch}: loss={average_loss:.4f}")

print("Learned weight:", model.weight.item())
print("Learned bias:", model.bias.item())

# Evaluate on x values that weren't used during training.
model.eval()
test_inputs = torch.tensor([[5.0], [6.0]], device=device)
test_targets = torch.tensor([[11.0], [13.0]], device=device)

with torch.inference_mode():
    test_predictions = model(test_inputs)
    test_loss = loss_function(test_predictions, test_targets)

print("Test predictions:", test_predictions.flatten().tolist())
print("Test targets:", test_targets.flatten().tolist())
print("Test MSE:", test_loss.item())