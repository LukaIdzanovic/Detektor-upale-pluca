import os
import cv2
import numpy as np

from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings
from model.visualisation import Visualisator


visualisator = Visualisator()

def home(request):
    context = {}

    if request.method == "POST" and request.FILES.get("image"):
        image = request.FILES["image"]
        fs = FileSystemStorage()

        filename = fs.save(image.name, image)
        name = filename.split(".")[0]
        image_path = fs.path(filename)

        result = visualisator.process_image(image_path)
        prediction = result["label"]
        probability = result["probability"]

        context["prediction"] = prediction
        context["probability"] = round(probability, 4)
        context["image_url"] = fs.url(filename)

        gradcam_filename = f"gradcam_{name}.png"
        gradcam_path = os.path.join(settings.MEDIA_ROOT, gradcam_filename)
        os.makedirs(os.path.dirname(gradcam_path), exist_ok=True)

        gradcam_img = result["gradcam"]

        if gradcam_img.dtype != np.uint8:
            gradcam_img = (gradcam_img * 255).astype(np.uint8)

        gradcam_bgr = cv2.cvtColor(gradcam_img, cv2.COLOR_RGB2BGR)

        cv2.imwrite(gradcam_path, gradcam_bgr)

        context["gradcam_url"] = fs.url(gradcam_filename)

    return render(request, "pneumonia/home.html", context)