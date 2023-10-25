import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 肩と腰、腰と足首の距離を計算する関数
def calculate_distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)**0.5

# 画像の読み込み
image_path = '/Users/nao0121/BohPJ/oc/OCC/publicdomainq-0013770nsv.jpeg'
image = cv2.imread(image_path)

with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    # BGR画像をRGBに変換して、結果を描画
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)

    if results.pose_landmarks:
        # 上半身と下半身の距離を計算
        upper_body_length = calculate_distance(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])

        lower_body_length = calculate_distance(
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
            results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE])

        # 上半身と下半身の比率を計算
        upper_ratio = (10*upper_body_length)/(upper_body_length+lower_body_length)
        lower_ratio = 10 - upper_ratio

        print(f"上半身の比率：{upper_ratio}")

        # 姿勢のランドマークを描画
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    cv2.imshow('MediaPipe Pose', image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()