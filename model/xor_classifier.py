import torch
from torch import nn


class XORNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.hidden = nn.Linear(2, 8)
        self.output = nn.Linear(8, 1)

    def forward(self, inputs):
        hidden_values = torch.relu(self.hidden(inputs))
        return self.output(hidden_values)


torch.manual_seed(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Training device:", device)

# XOR: output 1 when the two inputs differ, otherwise output 0.
inputs = torch.tensor(
    [
        [0.0, 0.0],
        [0.0, 1.0],
        [1.0, 0.0],
        [1.0, 1.0],
    ],
    device=device,
)
targets = torch.tensor([[0.0], [1.0], [1.0], [0.0]], device=device)

model = XORNet().to(device)

loss_function = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

# Train on all four examples together.
for epoch in range(1001):
    logits = model(inputs)
    loss = loss_function(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 200 == 0:
        print(f"Epoch {epoch}: loss={loss.item():.4f}")

# Convert the model's scores into 0/1 predictions.
model.eval()
with torch.inference_mode():
    probabilities = torch.sigmoid(model(inputs))
    predictions = (probabilities >= 0.5).float()

print("Expected:", targets.flatten().tolist())
print("Predicted:", predictions.flatten().tolist())