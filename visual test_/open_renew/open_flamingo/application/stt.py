import speech_recognition as sr

def recognize_speech(timeout=3):
    # 음성 입력을 받는 부분
    recognizer = sr.Recognizer()
    with sr.Microphone() as source:
        print("Say something!")
        try:
            # timeout 매개변수로 타임아웃 설정
            audio = recognizer.listen(source, timeout=timeout)
            print("Recognizing...")
            # 인식된 음성을 텍스트로 변환하여 반환
            return recognizer.recognize_google(audio)
        except sr.WaitTimeoutError:
            print("Timeout occurred, processing as zero shot.")
            