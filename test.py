import requests

url = "https://phunggiahuy159--sd-fastapi-fastapi-app-dev.modal.run/txt2img/"
data = {
    "prompt": "a beautiful sunset over mountains",
    "num_inference_steps": 20,
    "guidance_scale": 7.5
}

response = requests.post(url, json=data)
if response.status_code == 200:
    with open("generated_image.png", "wb") as f:
        f.write(response.content)
    print("Image saved successfully!")
else:
    print(f"Error: {response.status_code}")
    print(response.text)

    # curl -X POST "https://phunggiahuy159--sd-fastapi-fastapi-app-dev.modal.run/txt2img/" \
#   -H "Content-Type: application/json" \
#   -d '{"prompt": "a beautiful sunset over mountains"}' \
#   --output test.png