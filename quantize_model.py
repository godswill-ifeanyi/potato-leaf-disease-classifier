import torch

from model import PotatoCNN


# --------------------------------------------------
# 1. Device
# --------------------------------------------------

device = torch.device("cpu")


# --------------------------------------------------
# 2. Create the original model
# --------------------------------------------------

model = PotatoCNN(num_classes=3)


# --------------------------------------------------
# 3. Load the trained weights
# --------------------------------------------------

model.load_state_dict(
    torch.load(
        "best_potato_model.pth",
        map_location=device
    )
)


# --------------------------------------------------
# 4. Put model in evaluation mode
# --------------------------------------------------

model.eval()


# --------------------------------------------------
# 5. Apply dynamic INT8 quantization
# --------------------------------------------------

quantized_model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.Linear},
    dtype=torch.qint8
)


# --------------------------------------------------
# 6. Save the quantized model
# --------------------------------------------------

torch.save(
    quantized_model.state_dict(),
    "best_potato_model_int8.pth"
)


print("Quantized model saved successfully!")