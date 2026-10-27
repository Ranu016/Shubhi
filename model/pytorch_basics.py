"""A first PyTorch exercise: tensors, shapes, operations, and devices."""

import torch


def main() -> None:
    # A scalar has no dimensions; a vector has one dimension; a matrix has two.
    scalar = torch.tensor(7)
    vector = torch.tensor([1.0, 2.0, 3.0])
    matrix = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

    print("Scalar:", scalar, "shape:", tuple(scalar.shape))
    print("Vector:", vector, "shape:", tuple(vector.shape))
    print("Matrix:\n", matrix, "shape:", tuple(matrix.shape))

    # Indexes start at 0: select an element, a row, or a column.
    print("Element at row 2, column 1:", matrix[1, 0])
    print("First row:", matrix[0, :])
    print("First column:", matrix[:, 0])

    # Reshape changes the dimensions but keeps the same values.
    flat = matrix.reshape(4)
    column = matrix.reshape(4, 1)
    print("Flat shape:", tuple(flat.shape), "values:", flat)
    print("Column shape:", tuple(column.shape), "values:\n", column)

    # Elementwise operations keep the same shape.
    print("Vector + 10:", vector + 10)
    print("Matrix * 2:\n", matrix * 2)

    # Autograd calculates how much the output changes when x changes.
    x = torch.tensor(2.0, requires_grad=True)
    y = x**2 + 3 * x + 1
    y.backward()
    print("y:", y.item())
    print("Gradient of y with respect to x:", x.grad.item())

    # Train one weight so that weight * 3 approaches the target value 6.
    weight = torch.tensor(0.0, requires_grad=True)
    input_value = torch.tensor(3.0)
    target = torch.tensor(6.0)
    learning_rate = 0.1

    for step in range(20):
        prediction = weight * input_value
        loss = (prediction - target) ** 2
        loss.backward()

        # Update the weight without adding this update to the gradient graph.
        with torch.no_grad():
            weight -= learning_rate * weight.grad

        # Clear this step's gradient before the next one.
        weight.grad = None

        if step % 5 == 0:
            print(
                f"Step {step}: loss={loss.item():.4f}, "
                f"weight={weight.item():.4f}"
            )

    print("Learned weight:", weight.item())

    # Use the GPU when available; otherwise use the CPU.
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    matrix_on_device = matrix.to(device)
    print("Selected device:", device)
    print("Matrix device:", matrix_on_device.device)
    print("Matrix sum:", matrix_on_device.sum().item())

    if device.type == "cuda":
        print("GPU:", torch.cuda.get_device_name(0))


if __name__ == "__main__":
    main()