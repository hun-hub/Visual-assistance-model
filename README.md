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
  * Q key
    * q키의 경우 query 키로, 물어보고 싶은 순간의 장면을 캡처하는 것. 다시 말해, q키로 이미지를 캡처하면 프로그램(모델)을 통해 Image에 맞는 Captioning(설명)을 생성하여 넘겨주게 된다. 
		이때 제공되는 설명의 경우, 시각장애인임을 고려하여 음성을 출력하는(tts) 형태로 출력
## 🔍2. 성능평가 & 파인튜닝 

* **Open-Flamingo (Detail mode base model)**
  파인튜닝을 하지 않았을 때에도, 다양한 도메인 영역에서 Bert Score, CIDEr, BLEU와 같은 평가 지표에서 괜찮은 성능과 인간이 판단했을 때에도(Human-in-the-loop) 합리적 


	논문에서는 1,3,9,16 shot과 같이 shot을 늘릴수록 성능이 선형적으로 증가한다고 나와있지만, Viz-Wiz (시각 장애인이 찍은 데이터 셋)와 같은 데이터 셋으로 실제 평가를 진행하였을 때, 우리의 task의 경우 간단해서인지 shot의 수를 zero 혹은 one으로도 충분했으며, 오히려 shot이 높아질수록 논문과 다르게 attention을 수월하게 하지 못함을 보임( 너무 detail 한 정보에 집중하는 것으로 예상)

	*  Zero shot  
**large context**에서 어느 정보에 집중할지 모르는 것과 같은 캡셔닝을 하였다.
이는 시각장애인 task에서 **Critical** 함. 예를 들어 단일 횡단보도 이미지에 대해서 'An image of a zebra crossing in China.' 와 같이 횡단보도를 zebra로 인식하거나, 신호등 사진에 대해 'An image of an Australian traffic light.' 와 같이 국적과 같은 주요 정보가 아닌 정보를 제공하고 중요한 정보인 신호등의 색은 제외하는 등의 문제를 보였다. 
   

   	*  One shot

        shot이 hint와 같은 역할을 하기에 비교적 안정적 성능을 보였지만, shot을 주기 위해서 이미지와 그에 대한 text를 모델에 제공해주어야 하는데, 시각장애인의 특성을 고려했을 때 쉽지 않으며 순간적인 판단이 필요한 task에서 이는 더 **critical** 할 것으로 예상된다.


* **Microsoft/git-base  (Fast mode base model)**

	* Fine-tuned

        Open-Flamingo의 zero shot, one shot 약점 보완을 위해 Fine-tuning 진행하였다. Fast mode 일때 Open-Flamingo를 base model로 하기에는 너무 무겁다고 판단되어 다른 base model을 이용하여 Fine-tuning 하였다. 
   현재는 보행 관련 task에 대해서만 판단하도록 파인튜닝을 하였으며, 실시간성이 요구되는 task에 대해서 main.py를 통해 확장 가능하도록 구축하였다.
보행 관련 task에 대해서는 현재까지 뛰어난 성능과 실시간성을 보여주고 있다. 
   
     
    
## 🎁3. 데이터 셋 

* 성능평가
	* Viz-Wiz (Visual impaired) : 시각 장애인이 찍은 사진 데이터이다. 시각 장애인은 시야가 부분적으로 제한되기에, 그들이 찍은 사진에는 Obscured(occlusion), blur, rotation을 포함한다. 따라서, 모델이 이에도 강건하게 대응할 수 있는지 평가하기 위해 사용
 	* 안경 카메라(직접) : 시중에 판매하는 안경 카메라를 사용해 실제 보행 관련 장면을 녹화하였다. 저가 초소형 카메라이기에 화질이 좋지 않았음에도, 모델이 잘 판단함을 확인할 수 있었다.   

* Fine - tuning
	* WOTR(Walk On The Road) Dataset : 보행 관련 이미지를 담고 있는 데이터 셋으로, 파인 튜닝에 이용하였다.   	



   



