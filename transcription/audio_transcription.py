import speech_recognition as sr
import shelve

def transcribe_audio_files(uploaded_audio_files):
    recognizer = sr.Recognizer()
    transcriptions = {}

    for uploaded_audio in uploaded_audio_files:
        audio_path = uploaded_audio.name
        with open(audio_path, "wb") as f:
            f.write(uploaded_audio.getvalue())

        with sr.AudioFile(audio_path) as source:
            audio_data = recognizer.record(source)
            try:
                transcribed_text = recognizer.recognize_google(audio_data)
                print(f"Transcription: {transcribed_text}")
            except sr.UnknownValueError:
                transcribed_text = "<Unable to transcribe>"
                print("Google Speech Recognition could not understand audio.")
            except sr.RequestError as e:
                transcribed_text = "<Error in transcription service>"
                print(f"Could not request results from Google Speech Recognition service; {e}")

            transcriptions[uploaded_audio.name] = transcribed_text

    # Store transcriptions in database
    with shelve.open("transcriptions") as db:
        for filename, transcription in transcriptions.items():
            db[filename] = transcription

    return transcriptions
