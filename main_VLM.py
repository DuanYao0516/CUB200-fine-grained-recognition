## CLIP zero-shot
from transformers import CLIPProcessor, CLIPModel
from PIL import Image
## install packages: torch, transformers==4.23.1


if __name__ == '__main__':
    print('Loading Model, wait for a minute.')
    model_name = "openai/clip-vit-base-patch32"
    model = CLIPModel.from_pretrained(model_name)
    model = model.eval()
    for param in model.parameters():
        param.requires_grad = False
    processor = CLIPProcessor.from_pretrained(model_name)

    image = Image.open(r"Photo path")
    text_labels = ["a photo of a cat", "a photo of a dog", "a photo of a horse"]
    inputs = processor(text=text_labels, images=image, return_tensors="pt", padding=True)
    outputs = model(**inputs)
    logits_per_image = outputs.logits_per_image
    probs = logits_per_image.softmax(dim=1)
    predicted_label_idx = probs.argmax()

    print(predicted_label_idx)
    print(text_labels[predicted_label_idx])


