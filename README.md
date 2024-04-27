# Visual_assist
Create an end-to-end multimodal (visual language model) process to assist individuals with visual impairments.

## 👨‍💻0. 실행 

choose_mode.py를 실행하면 키를 입력받게 된다.

이때 키 값으로 f 또는 d 키를 입력 받는데, f 키의 경우 fast mode d 키의 경우 detail mode로 각각 요구되는 상황에 대해 실행할 수 있도록 하였다.

fast mode의 경우 base model로 microsoft/git-base model를 사용하였으며, detail 모드에서는 Open-Flamingo 모델을 사용, 내부적으로 vision_encoder로는 ViT-L-14, lang_encoder로는 MPT - 1B를 사용하였다.

* Fast mode
  * Purpose : 신호 판단, 횡단보도 여부, 보행자 앞 자동차 여부 정보등과 같이 순간 판단이 정보에 대해 **빠르고** 정확하게 정보를  제공하여 시각장애인에게 도움
  * Measurement (CPU 사용 기준)
    * Model loading time: 2.41 seconds
    * Average caption generation time: 1.6138930819755377 seconds
    * Num of parameters : 176619066 (약 1억 7천만개)


* Detail mode
  * Purpose : 특정 task가 아닌 범용적인 상황에서의 사용, 실시간성보다는 **정확성**이 요구되는 상황
  * Measurement (CPU 사용 기준)
    * Model loading time: 10.96 seconds
    * Average caption generation time: 4.0784940261107225 seconds
    * Num of parameters : 1046992944 약 (10억개)

## 📝1. 프로세스 
End-to-End Process를 구축하여 실제 하드웨어에서 작동하는 것을 고려하며 설계하였다. 프로세스는 다음과 같다.

(사진) 



**캡처를 위해 한 번의 입력을 더 받게 되는데, q 키와 s 키이다. q 키와 s 키는 실제 하드웨어(안경 카메라)의 버튼에 대응되며, 현재는 키보드 입력으로 임시 구현**

  * S key
    * s키의 경우 shot 키로, shot을 줄 때에만 사용. **shot 이란** 모델에 제공하는 **hint**와 같은 것으로 캡처한 순간에 대한 (Text, Image) 쌍을 넘겨주어야 함.
   
    * 
		따라서, **detail mode의 zero-shot, fast mode**에는 **사용 x**

      반면, **detail mode의 few shot**의 경우 s키를 사용하여 모델에 정보를 제공해야 함. 이때, 캡처에 대한 **description text**를 넘겨주어야 하는데, 시각장애인임을 고려하여 음성 입력을 받아(stt) 넘겨주도록 구현

      
