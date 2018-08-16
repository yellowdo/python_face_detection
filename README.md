# Face & Eye Detection

---

###  python dlib & opencv

- Face & Facial landmark detector implemented inside dlib

- Produces 68 (x,y)-coordinates (specific facial structures)

- These 68 points mappings were obtained by training a shape predictor on the labeled iBUG300-W dataset

![](/image/facial68.png)

### Facial Landmark Detect

- import

```
from imutils.video import VideoStream
from imutils import face_utils
import imutils, dlib, cv2
```

- Hog + Linear SVM (dlib.get_frontal_face_detector())

 - 영상에서 얼굴 검출에 사용 될 객체

 `detector = dlib.get_frontal_face_detector()`

 - 검출 된 얼굴 영상에서 랜드마크 추출

`predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')`
- 웹캠 연결

 `vs = VideoStream(0).start()`

- 캠 영상에서 얼굴 랜드마크 표시

```
while True:
    # 영상 사이즈 조절
    frame = imutils.resize(vs.read(), width=450)
    # 그레이 스케일로 변환
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    # 영상에서 얼굴 검출
    for rect in detector(gray, 0):
        # 추출 된 랜드마크들의 x, y 좌표로 받아 원본 프레임에 circle 표시
        [cv2.circle(frame, (x, y), 1, (255, 255, 255), -1) for (x, y) in face_utils.shape_to_np(predictor(gray, rect))]
    # 영상 출력
    cv2.imshow("Frame", frame)
    # 키보드 q 입력 시 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
```

- 결과 [facial_landmark_detect.py](/facial_landmark_detect.py)

![](/image/landmark.jpg)

### Facial Landmark Detect

- Drowsiness Detection based on Face & Eye Detection using landmark

 - Using EAR(Eye Aspect Ratio)
 
![](/image/ear.jpg)

```
def eye_aspect_ratio(eye):
    return (dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])) / (2.0 * dist.euclidean(eye[0], eye[3]))
```

- Import

```
from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound, imutils
import dlib, cv2
```

- 왼쪽눈, 오른쪽눈 랜드마크 인덱스

`(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]`
`(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
`
- eye aspect ratio 값이 48프레임동안 0.3 이하 일 경우 눈 감는 것으로 간주

`EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES = 0.3, 48`

- 랜드마크에서 눈 좌표를 추출하여 눈이 감는지 체크

```
    for rect in detector(gray, 0):
        # 랜드마크 추출
        shape = face_utils.shape_to_np(predictor(gray, rect))
        # 양쪽눈의 좌표
        leftEye, rightEye = shape[lStart:lEnd], shape[rStart:rEnd]
        # 양쪽 눈의 평균 aspect ratio
        ear = (eye_aspect_ratio(leftEye) + eye_aspect_ratio(rightEye)) / 2.0
        # 양쪽 눈의 최외곽 점들만 추출하여 선으로 표시
        cv2.drawContours(frame, [cv2.convexHull(leftEye)], -1, (0, 200, 0), 1)
        cv2.drawContours(frame, [cv2.convexHull(rightEye)], -1, (200, 0, 0), 1)
        # 양쪽눈의 평균 aspect ratio 표시
        cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 10, 10), 2)
        # aspect ratio가 0.3 이하 일 경우 눈 감는 것으로 간주
        if ear < EYE_AR_THRESH:
            # 프레임이 증가할 때마다 COUNTER 1씩 증가, 48프레임 이상 계속 눈 감으면 경고 표시 및 알람 울림
            COUNTER += 1
            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # 경고문 표시
                cv2.putText(frame, "DROWSINESS ALERT!", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                if not ALARM_ON:
                    # 알람은 한번 울린다.
                    ALARM_ON = True
                    playsound_alarm()
```

- 결과 [detect_drowsiness.py](/detect_drowsiness.py)

![](/image/drowsiness.jpg)

### 환경 설정

- [Windows Anaconda 설치 및 설정](/env.md)

- [pycharm 설치 및 기본 설정](/pycharm.md)

