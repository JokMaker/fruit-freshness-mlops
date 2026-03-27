from locust import HttpUser, task, between
from PIL import Image
import io
import random

class FruitFreshnessUser(HttpUser):
    wait_time = between(1, 3)

    @task(3)
    def predict(self):
        image_bytes = self._get_test_image()
        self.client.post(
            "/predict",
            files={"file": ("test.jpg", image_bytes, "image/jpeg")},
        )

    @task(1)
    def check_status(self):
        self.client.get("/status")

    @task(1)
    def get_classes(self):
        self.client.get("/classes")

    def _get_test_image(self):
        img = Image.new("RGB", (224, 224), color=(
            random.randint(0, 255),
            random.randint(0, 255),
            random.randint(0, 255)
        ))
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        return buf.getvalue()