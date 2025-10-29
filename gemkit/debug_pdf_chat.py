import os
import pathlib
from google import genai

# Initialize client
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

# Load PDF
pdf_path = pathlib.Path("book.pdf")
print(f"Uploading {pdf_path.name}...")
uploaded_file = client.files.upload(file=pdf_path, config=dict(mime_type='application/pdf'))
print(f"✓ PDF uploaded: {uploaded_file.name}")

# Test query
prompt = "What is this book about?"
print(f"\nSending prompt: {prompt}")

response = client.models.generate_content(
    model="gemini-2.5-flash",
    contents=[uploaded_file, prompt]
)

print(f"\nResponse object type: {type(response)}")
print(f"Response object: {response}")
print(f"Has text attr: {hasattr(response, 'text')}")
if hasattr(response, 'text'):
    print(f"Response.text: {response.text}")
    print(f"Response.text type: {type(response.text)}")
print(f"\nResponse.__dict__: {response.__dict__ if hasattr(response, '__dict__') else 'N/A'}")

# Clean up
client.files.delete(name=uploaded_file.name)
