from google import genai
import os

API_KEY = os.getenv("GEMINI_API_KEY")

print("Testing API Key...")
try:
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model='gemini-1.5-flash',
        contents='Reply with the word "SUCCESS" if you receive this.'
    )
    print("\n✅ GOLDEN KEY! Google replied:", response.text)
except Exception as e:
    print("\n❌ FAILED. Google rejected the key. Details:\n", e)