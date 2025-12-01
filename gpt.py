import speech_recognition as sr
from openai import OpenAI
from pathlib import Path
from pydub import AudioSegment
import os


client = OpenAI()

def text_to_speech(text):
    speech_file_path = Path(__file__).parent / "speech.mp3"
    response = client.audio.speech.create(
    model="tts-1",
    voice="alloy",
    input=text
    )

    response.stream_to_file(speech_file_path)
    audio = AudioSegment.from_mp3("speech.mp3")
    audio.export("speech.wav", format="wav")
    cmdline = 'aplay ' + " speech.wav" 
    os.system(cmdline)



# Initialize recognizer
recognizer = sr.Recognizer()
microphone = sr.Microphone()

# Define the wake word
WAKE_WORD = "hi"

def listen_for_wake_word():
    with microphone as source:
        recognizer.adjust_for_ambient_noise(source, duration=0.5)
        print("Listening for wake word...")
        
        while True:
            audio = recognizer.listen(source)
            # audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
            try:
                text = recognizer.recognize_google(audio).lower()
                if WAKE_WORD in text:
                    print(f"Wake word '{WAKE_WORD}' detected.")
                    text_to_speech("hi,what can i do for you?")
                    return True
            except sr.UnknownValueError:
                continue
            except sr.RequestError as e:
                print(f"Could not request results; {e}")
                continue

i=0
def listen_for_command():
    global i
    with microphone as source:
        print("Listening for command...")
        # audio = recognizer.listen(source)
        audio = recognizer.listen(source, timeout=5, phrase_time_limit=5)
        try:
            command = recognizer.recognize_google(audio)
            print(f"You said: {command}")
            i=0
            return command
        except sr.UnknownValueError:
            print("Could not understand the audio")
            i = i+1
        except sr.RequestError as e:
            print(f"Could not request results; {e}")
            i = i+1


def get_gpt_response(prompt):
    completion = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=[
        {"role": "system", "content": "Your name is speaker, you can answer all kinds of questions for me"},
        {"role": "user", "content": prompt}
    ]
    )

    content_string = completion.choices[0].message.content
    paragraphs = content_string.split('\n\n')
    combined_content = ' '.join(paragraphs)
    return combined_content




def main():
    global i
    while 1:
        flag = listen_for_wake_word()
        while flag == True:
            user_input = listen_for_command()
            if i==3:
                flag = False
                i = 0
            if user_input:
                gpt_response = get_gpt_response(user_input)
                print(f"GPT says: {gpt_response}")
                text_to_speech(gpt_response)
                

if __name__ == "__main__":
    main()