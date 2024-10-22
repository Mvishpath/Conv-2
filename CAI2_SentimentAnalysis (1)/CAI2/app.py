from flask import Flask, render_template, request, jsonify, send_from_directory
from google.cloud import language_v1  # For Sentiment Analysis
from google.cloud import speech, texttospeech
from flask_cors import CORS
import subprocess
import os
import uuid

import os
os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = "/Users/sai/Documents/high-science-436713-p2-c70347159483.json"



app = Flask(__name__)
CORS(app)

# Create directories for saving recordings and transcripts if they don't exist
if not os.path.exists('recordings'):
    os.makedirs('recordings')
if not os.path.exists('transcripts'):
    os.makedirs('transcripts')

@app.route('/')
def home():
    # List all saved transcripts and recordings
    transcripts = os.listdir('transcripts')
    recordings = os.listdir('recordings')
    return render_template('index.html', transcripts=transcripts, recordings=recordings)


def analyze_sentiment(text_content):
    client = language_v1.LanguageServiceClient()
    document = language_v1.Document(content=text_content, type_=language_v1.Document.Type.PLAIN_TEXT)
    sentiment = client.analyze_sentiment(document=document).document_sentiment
    if sentiment.score > 0:
        return "Positive" , sentiment.magnitude
    elif sentiment.score < 0:
        return "Negative" , sentiment.magnitude
    else:
        return "Neutral", sentiment.magnitude

@app.route('/audio-to-text', methods=['POST'])
def convert_audio_to_text():
    audio_data = request.data
    transcript_id = str(uuid.uuid4())  # Unique ID for each transcription

    # Save the uploaded audio to a temporary file
    audio_file_path = f'recordings/{transcript_id}.aac'
    with open(audio_file_path, 'wb') as audio_file:
        audio_file.write(audio_data)

    # Convert AAC to WAV using FFmpeg
    wav_file_path = f'recordings/{transcript_id}.wav'
    try:
        subprocess.run(['ffmpeg', '-i', audio_file_path, '-acodec', 'pcm_s16le', '-ar', '48000', wav_file_path], check=True)
    except subprocess.CalledProcessError as error:
        print(f"FFmpeg error: {error}")
        return jsonify({"error": "Audio conversion failed."}), 500

    speech_client = speech.SpeechClient()

    with open(wav_file_path, 'rb') as wav_file:
        audio_content = wav_file.read()

    audio = speech.RecognitionAudio(content=audio_content)
    config = speech.RecognitionConfig(
        encoding=speech.RecognitionConfig.AudioEncoding.LINEAR16,
        sample_rate_hertz=48000,
        language_code="en-US"
    )

    response = speech_client.recognize(config=config, audio=audio)

    full_transcript = ""
    if response.results:
        transcripts = []
        for result in response.results:
            transcripts.append(result.alternatives[0].transcript)
        full_transcript = "\n".join(transcripts)

    # Analyze sentiment of the transcription
    sentiment_score, sentiment_magnitude = analyze_sentiment(full_transcript)

    # Save the transcript and sentiment to a file
    transcript_file_path = f'transcripts/{transcript_id}.txt'
    with open(transcript_file_path, 'w') as transcript_file:
        transcript_file.write(f"Transcript:\n{full_transcript or 'No transcription available.'}\n\n")
        transcript_file.write(f"Sentiment Score: {sentiment_score}\n")
        transcript_file.write(f"Sentiment Magnitude: {sentiment_magnitude}\n")

    # Clean up temporary WAV file
    os.remove(wav_file_path)

    return jsonify({
        "transcription": full_transcript,
        "sentiment_score": sentiment_score,
        "sentiment_magnitude": sentiment_magnitude,
        "id": transcript_id
    })

@app.route('/text-to-speech', methods=['POST'])
def convert_text_to_speech():
    request_data = request.get_json()
    text_content = request_data.get('text')

    tts_client = texttospeech.TextToSpeechClient()

    synthesis_input = texttospeech.SynthesisInput(text=text_content)
    voice_params = texttospeech.VoiceSelectionParams(
        language_code="en-US",
        ssml_gender=texttospeech.SsmlVoiceGender.NEUTRAL
    )
    audio_settings = texttospeech.AudioConfig(
        audio_encoding=texttospeech.AudioEncoding.MP3
    )

    response = tts_client.synthesize_speech(input=synthesis_input, voice=voice_params, audio_config=audio_settings)

    audio_filename = f'output_audio_{uuid.uuid4()}.mp3'  # Generate unique filename
    with open(f'recordings/{audio_filename}', 'wb') as output_file:
        output_file.write(response.audio_content)

    return jsonify({"audio_url": f"/recordings/{audio_filename}"})

@app.route('/recordings/<filename>')
def send_recording(filename):
    return send_from_directory('recordings', filename)

@app.route('/transcripts/<filename>')
def send_transcript(filename):
    return send_from_directory('transcripts', filename)





if __name__ == '__main__':
    app.run(host='0.0.0.0', port=int(os.environ.get('PORT', 8080)))


