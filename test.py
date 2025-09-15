import requests

# Flask backend URL
url = "http://127.0.0.1:5000/authenticate"

# Path to your test image
IMAGE_PATH = r"C:\Users\Vighnesh\Desktop\1001.jpeg"

def main():
    try:
        # Send POST request with image as multipart/form-data
        with open(IMAGE_PATH, "rb") as f:
            files = {"file": f}
            response = requests.post(url, files=files)

        # Handle response
        if response.status_code == 200:
            print("✅ Response from server:")
            print(response.json())
        else:
            print(f"❌ Request failed [{response.status_code}]:")
            print(response.text)

    except Exception as e:
        print(f"⚠️ Error running test: {e}")

if __name__ == "__main__":
    main()
