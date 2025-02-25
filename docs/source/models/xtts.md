pip install pypdf2 tts numpy soundfile
import PyPDF2
import os
from TTS.api import TTS

# Path to the uploaded PDF file
pdf_path = "Document from Vijay Shankar V.pdf"
output_folder = "audiobook_output"

# Ensure output folder exists
os.makedirs(output_folder, exist_ok=True)

# Extract text from PDF
def extract_text_from_pdf(pdf_path):
    text = ""
    with open(pdf_path, "rb") as file:
        pdf_reader = PyPDF2.PdfReader(file)
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
    return text

# Split text into chunks for better TTS processing
def split_text(text, max_words=500):
    words = text.split()
    chunks = [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]
    return chunks

# Convert text to speech using Coqui TTS
def text_to_speech(text_chunks):
    tts_model = "tts_models/en/ljspeech/tacotron2-DDC"  # Change model if needed
    tts = TTS(tts_model).to("cpu")  # Use CPU (change to "cuda" if using GPU)

    for i, chunk in enumerate(text_chunks):
        output_file = os.path.join(output_folder, f"part_{i+1}.wav")
        tts.tts_to_file(text=chunk, file_path=output_file)
        print(f"Generated: {output_file}")

if __name__ == "__main__":
    extracted_text = extract_text_from_pdf(pdf_path)
    text_chunks = split_text(extracted_text, max_words=500)  # Adjust word limit if needed
    text_to_speech(text_chunks)
