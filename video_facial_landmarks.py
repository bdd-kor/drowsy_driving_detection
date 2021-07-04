# 실행 : python video_facial_landmarks.py --shape-predictor shape_predictor_68_face_landmarks.dat --picamera 1
 
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import datetime
import argparse
import imutils
import time
import dlib
import cv2
import RPi.GPIO as GPIO
GPIO.setmode(GPIO.BOARD)
LED = 11

GPIO.setup(LED, GPIO.OUT, initial=GPIO.LOW)

# 왼쪽 눈과 오른쪽 눈의 랜드마크 좌표를 잡음
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
 
EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 16

# 프레임 카운터 초기화
COUNTER = 0
def euclidean_dist(ptA, ptB):
    # 두 점 사이의 유클리드 거리를 계산
    return np.linalg.norm(ptA - ptB)


def eye_aspect_ratio(eye):
    # 수직 눈 랜드마크(x, y) 좌표 사이의 유클리드 거리를 계산
    A = euclidean_dist(eye[1], eye[5])
    B = euclidean_dist(eye[2], eye[4])

    # 수평 눈 랜드마크(x, y) 좌표 사이의 유클리드 거리를 계산
    C = euclidean_dist(eye[0], eye[3])

    # 눈의 가로 세로 비율을 계산
    ear = (A + B) / (2.0 * C)

    return ear


# 얼굴 랜드마크 predictor의 경로와 파이카메라 사용 유무 체크
ap = argparse.ArgumentParser()
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-r", "--picamera", type=int, default=-1,
    help="whether or not the Raspberry Pi camera should be used")
args = vars(ap.parse_args())
 
# dlib의 face detector와 랜드마크 predictor 생성
print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(args["shape_predictor"])
 
# 비디오스트림 초기화 및 카메라 센서 준비
print("[INFO] camera sensor warming up...")
vs = VideoStream(usePiCamera=args["picamera"] > 0).start()
time.sleep(2.0)
 
# 비디오 스트림
while True:
    # 비디오 스트림의 다음 프레임을 잡고 이 프레임을 400픽셀의 너비로 크기조정 후
    # 그레이스케일로 변환
    frame = vs.read()
    frame = imutils.resize(frame, width=400)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
 
    # 그레이스케일 프레임에서 얼굴 감지
    rects = detector(gray, 0)
 
    # 얼굴감지
    for rect in rects:
        # 얼굴 랜드마크를 정한 후 랜드마크 좌표를 Numpy 배열로 변환
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
 
        # 얼굴 랜드마크의 좌표를 그림
        for (x, y) in shape:
            cv2.circle(frame, (x, y), 1, (0, 0, 255), -1)
        
        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = eye_aspect_ratio(leftEye)
        rightEAR = eye_aspect_ratio(rightEye)

        ear = (leftEAR + rightEAR) / 2.0
        
        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        if ear < EYE_AR_THRESH:
            COUNTER += 1

            # 눈이 감기면 경고등 켜짐
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                
                GPIO.output(LED, GPIO.HIGH)
                time.sleep(0.5)
                GPIO.output(LED, GPIO.LOW)
                time.sleep(0.5)
                # 경고문구를 화면에 표시
                cv2.putText(frame, "WAKE UP!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

        # 카운터 초기화
        else:
            COUNTER = 0

        # 이 바로 아래 코드는 EAR값을 화면에 출력해주는 코드로 처음에 삽입하였다가 너무 느려지길래 주석처리함
        # cv2.putText(frame, "EAR: {:.3f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
  
    # 프레임 보여주기
    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1) & 0xFF
 
    # q키를 누르면 루프에서 나옴
    if key == ord("q"):
        break
 
# 클린업
GPIO.cleanup()
cv2.destroyAllWindows()
vs.stop()