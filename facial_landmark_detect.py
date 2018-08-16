from imutils.video import VideoStream
from imutils import face_utils
import imutils, dlib, cv2

# 영상에서 얼굴 검출에 사용 될 객체
detector = dlib.get_frontal_face_detector()
# 검출 된 얼굴 영상에서 랜드마크 추출
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
# 웹캠 연결
vs = VideoStream(0).start()
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
cv2.destroyAllWindows()
vs.stop()
