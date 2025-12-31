import torch
import torchvision.transforms as transforms
from PIL import Image
import cv2
from torchvision import models
import torch.nn as nn

# Classes
included_classes = ['Hazardous', 'Organic', 'Recyclable']
num_classes = len(included_classes)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# Load Model
model = models.resnet50(weights=None)
model.fc = nn.Linear(model.fc.in_features, num_classes)
model.load_state_dict(torch.load("best_model_resnet50.pth", map_location=device))
model.to(device)
model.eval()

# Preprocessing
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225]),
])

confidence_threshold = 0.6

# Webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("ERROR: Webcam not accessible.")
    exit()

print("Press 's' to save image | Press 'q' to quit")

save_index = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Frame grab failed.")
        break

    # Convert BGR frame → RGB → PIL
    img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        prob = torch.nn.functional.softmax(output[0], dim=0)

    conf, idx = torch.max(prob, 0)

    if conf.item() < confidence_threshold:
        label = f"Other ({conf.item()*100:.1f}%)"
    else:
        label = f"{included_classes[idx]} ({conf.item()*100:.1f}%)"

    # Display label
    cv2.putText(frame, label, (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Waste Classifier", frame)

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break
    elif key == ord("s"):
        filename = f"classified_{save_index}_{label.replace(' ', '_').replace('(', '').replace(')', '')}.jpg"
        cv2.imwrite(filename, frame)
        print("Saved:", filename)
        save_index += 1

cap.release()
cv2.destroyAllWindows()
