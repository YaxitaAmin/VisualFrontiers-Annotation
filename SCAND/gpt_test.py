from openai import OpenAI
import base64, cv2
import numpy as np
import os

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
ok, buf = cv2.imencode(".png", np.zeros((10,10,3), np.uint8))
b64 = base64.b64encode(buf.tobytes()).decode()
try:
    r = client.chat.completions.create(
        model="gpt-5-mini",
        messages=[
            {"role": "user", "content": [
                {"type": "text", "text": "what do you see?"},
                {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}
            ]}
        ]
    )
    print("✅ vision supported")
except Exception as e:
    print("❌ no vision:", e)