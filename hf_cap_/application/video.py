# 동영상 프로세싱
import cv2
import datetime

def process_video_frames(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("비디오를 열 수 없습니다. 파일 경로를 확인하세요.")
        exit()
    
    img_counter = 0

    while (cap.isOpened):

        ret, frame = cap.read()

        if ret == False:
            break

        cv2.imshow("VideoFrame", frame)

        key = cv2.waitKey(33) & 0xFF  


        if key == 27:  # esc 종료

            break

        elif key == ord("q"):  # "n" 키를 누르면 이미지 저장
            capture_img_save = "C:/Users/USER/Desktop/hf_cap/capture_img_save/img1.png"
            cv2.imwrite(capture_img_save, frame)

            img_name = "C:/Users/USER/Downloads/open_renew/img/query/img.png"
            cv2.imwrite(img_name, frame)
            print("{} 저장됨".format(img_name))
            break




    cap.release()

    cv2.destroyAllWindows()