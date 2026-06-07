import tkinter as tk
from tkinter import filedialog

from PIL import Image, ImageTk
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms

from model_param_matched import LightweightCIFAR10Classifier


# =========================
# CIFAR-10 類別名稱
# =========================
classes = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


# =========================
# 基本設定
# =========================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 你的 train_param_matched.py 預設輸出位置
checkpoint_path = "runs/lightweight_cifar10_param_matched/best.pt"


# =========================
# 載入模型
# =========================
def load_model():
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # 從 checkpoint 裡面讀 k
    # 如果沒有 args，就預設 k=89
    if "args" in checkpoint and "k" in checkpoint["args"]:
        k = checkpoint["args"]["k"]
    else:
        k = 89

    model = LightweightCIFAR10Classifier(num_classes=10, k=k).to(device)

    # 你的 train.py 是存 checkpoint["model_state"]
    model.load_state_dict(checkpoint["model_state"])

    model.eval()

    best_acc = checkpoint.get("best_acc", None)
    epoch = checkpoint.get("epoch", None)

    print("Model loaded successfully.")
    print("Device:", device)
    print("Checkpoint:", checkpoint_path)

    if epoch is not None:
        print("Epoch:", epoch)

    if best_acc is not None:
        print(f"Best Accuracy: {best_acc * 100:.2f}%")

    return model


model = load_model()


# =========================
# 圖片前處理
# 必須跟 train.py 的 test_transform 一致
# =========================
transform = transforms.Compose([
    transforms.Resize((32, 32)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=(0.4914, 0.4822, 0.4465),
        std=(0.2470, 0.2435, 0.2616)
    ),
])


# =========================
# 預測單張圖片
# =========================
def predict_image(image_path):
    image = Image.open(image_path).convert("RGB")

    input_tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(input_tensor)
        probabilities = F.softmax(outputs, dim=1)
        confidence, predicted = torch.max(probabilities, dim=1)

    pred_class = classes[predicted.item()]
    conf = confidence.item() * 100

    top5_prob, top5_idx = torch.topk(probabilities, k=5, dim=1)

    top5_results = []
    for i in range(5):
        class_name = classes[top5_idx[0][i].item()]
        prob = top5_prob[0][i].item() * 100
        top5_results.append((class_name, prob))

    return image, pred_class, conf, top5_results


# =========================
# GUI：選擇圖片
# =========================
def choose_image():
    file_path = filedialog.askopenfilename(
        title="選擇要測試的圖片",
        filetypes=[
            ("Image files", "*.jpg *.jpeg *.png *.bmp *.webp"),
            ("All files", "*.*")
        ]
    )

    if not file_path:
        return

    image, pred_class, conf, top5_results = predict_image(file_path)

    # 顯示圖片
    display_img = image.resize((256, 256))
    tk_img = ImageTk.PhotoImage(display_img)

    image_label.config(image=tk_img)
    image_label.image = tk_img

    # 顯示結果
    result_text = f"Prediction: {pred_class}\nConfidence: {conf:.2f}%\n\nTop-5 Results:\n"

    for class_name, prob in top5_results:
        result_text += f"{class_name}: {prob:.2f}%\n"

    result_label.config(text=result_text)


# =========================
# 建立 GUI 視窗
# =========================
root = tk.Tk()
root.title("CIFAR-10 Image Classifier Test")
root.geometry("500x600")

title_label = tk.Label(
    root,
    text="Lightweight CIFAR-10 Classifier",
    font=("Arial", 16, "bold")
)
title_label.pack(pady=10)

button = tk.Button(
    root,
    text="選擇圖片並預測",
    command=choose_image,
    font=("Arial", 12),
    width=20
)
button.pack(pady=10)

image_label = tk.Label(root)
image_label.pack(pady=10)

result_label = tk.Label(
    root,
    text="請選擇一張圖片進行測試",
    font=("Arial", 12),
    justify="left"
)
result_label.pack(pady=10)

root.mainloop()