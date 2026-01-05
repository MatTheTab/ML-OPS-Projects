import bentoml
from PIL import Image
import torch
from transformers import AutoImageProcessor, MobileNetV2ForImageClassification

processor = AutoImageProcessor.from_pretrained("google/mobilenet_v2_1.0_224")
model = MobileNetV2ForImageClassification.from_pretrained("google/mobilenet_v2_1.0_224")


@bentoml.service(name="mobilenet_classifier")
class MobileNetService:
    @bentoml.api
    def classify(self, image: Image.Image) -> dict:
        # image is already a PIL Image thanks to BentoML's auto-conversion
        inputs = processor(image, return_tensors="pt")
        with torch.no_grad():
            logits = model(**inputs).logits

        predicted_label = logits.argmax(-1).item()
        label = model.config.id2label[predicted_label]
        return {"label": label}
