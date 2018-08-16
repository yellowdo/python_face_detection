from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
import playsound
import imutils
import dlib, cv2

def eye_aspect_ratio(eye):
    return (dist.euclidean(eye[1], eye[5]) + dist.euclidean(eye[2], eye[4])) / (2.0 * dist.euclidean(eye[0], eye[3]))

def playsound_alarm():
    t = Thread(target=playsound.playsound, args=("dingdong.wav",))
    t.deamon = True
    t.start()

# 영상에서 얼굴 검출에 사용 될 객체
detector = dlib.get_frontal_face_detector()
# 검출 된 얼굴 영상에서 랜드마크 추출
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# 왼쪽눈, 오른쪽눈 랜드마크 인덱스
(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]
# eye aspect ratio 값이 48프레임동안 0.3 이하 일 경우 눈 감는 것으로 간주
EYE_AR_THRESH, EYE_AR_CONSEC_FRAMES = 0.3, 48
COUNTER, ALARM_ON = 0, False
# 웹캠 연결
vs = VideoStream(0).start()
while True:
    frame = imutils.resize(vs.read(), width=450) # 영상 사이즈 조절
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) # 그레이 스케일로 변환
    # 영상에서 얼굴 검출
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
        else:
            COUNTER = 0
            ALARM_ON = False

    cv2.imshow("Frame", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break
cv2.destroyAllWindows()
vs.stop()