import cv2
import mediapipe as mp

mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# 肩と腰、腰と足首の距離を計算する関数
def calculate_distance(landmark1, landmark2):
    return ((landmark1.x - landmark2.x)**2 + (landmark1.y - landmark2.y)**2)**0.5

# 骨格スタイルを判定する関数
def determine_style_from_average_ratio(average_upper_ratio):
    style_ranges = {
        "ウェーブ": (3.8, 4.5),
        "ストレート": (3.8, 4.5),
        "ナチュラル": (3.8, 4.5)
    }

    for style, (lower_bound, upper_bound) in style_ranges.items():
        if lower_bound <= average_upper_ratio <= upper_bound:
            return style
    return "不明"

def main():
    cap = cv2.VideoCapture(0)

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        session_upper_ratio = 0
        session_frame_count = 0

        try:
            while cap.isOpened():
                ret, frame = cap.read()

                image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image.flags.writeable = False
                results = pose.process(image)

                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

                if results.pose_landmarks:
                    upper_body_length = calculate_distance(
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP])

                    lower_body_length = calculate_distance(
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_HIP],
                        results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_ANKLE])

                    upper_ratio = (10 * upper_body_length) / (upper_body_length + lower_body_length)

                    print(f"上半身の比率：{upper_ratio}")

                    session_upper_ratio += upper_ratio
                    session_frame_count += 1

                    mp_drawing.draw_landmarks(
                        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

                cv2.imshow('MediaPipe Pose', image)
                if cv2.waitKey(10) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            pass

        if session_frame_count > 0:
            session_average_upper_ratio = session_upper_ratio / session_frame_count
            print(f"セッション内の平均上半身の比率：{session_average_upper_ratio}")

            detected_style = determine_style_from_average_ratio(session_average_upper_ratio)
            print(f"あなたは骨格は「{detected_style}」に類似しています。")

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
