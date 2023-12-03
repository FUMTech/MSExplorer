import os
from google.cloud import texttospeech
from PyPDF2 import PdfFileReader
import io

# Path to your Google Cloud service account key file
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = "path/to/your/service-account-file.json"

# Initialize the Text-to-Speech client
client = texttospeech.TextToSpeechClient()

# Function to convert text to speech
def text_to_speech(text, output_filename):
    synthesis_input = texttospeech.SynthesisInput(text=text)

    # Choose the type of voice
    voice = texttospeech.VoiceSelectionParams(
        language_code="en-US",  # Language
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL, # Voice gender
        name="en-US-Wavenet-D"  # You can choose a specific voice model here
    )

    # Select the type of audio file you want
    audio_config = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    # Perform the text-to-speech request
    response = client.synthesize_speech(
        input=synthesis_input, voice=voice, audio_config=audio_config
    )

    # Write the response to an MP3 file
    with open(output_filename, "wb") as out:
        out.write(response.audio_content)
        print(f"Audio content written to file {output_filename}")

# Function to read text from PDF
def extract_text_from_pdf(pdf_path):
    with open(pdf_path, "rb") as file:
        reader = PdfFileReader(file)
        text = ""
        for page in range(reader.numPages):
            text += reader.getPage(page).extractText()
        return text

# Path to your PDF file
pdf_path = "/home/amir/Downloads/Telegram Desktop/Zhenti 2022.pdf"

# Extract text from PDF
pdf_text = extract_text_from_pdf(pdf_path)

# Convert extracted text to speech
text_to_speech(pdf_text, "output.mp3")
