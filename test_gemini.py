from google import genai

# PASTE YOUR BRAND NEW KEY EXACTLY BETWEEN THE QUOTES
API_KEY = "AIzaSyCdJGKCEJYcSpDa6qopTV-epI9hFwffA64"

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