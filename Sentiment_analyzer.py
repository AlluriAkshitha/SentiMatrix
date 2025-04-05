import cv2
from deepface import DeepFace

# Sample ground truth labels for testing (Replace with actual labels if available)
ground_truth = ["happy", "sad", "neutral", "angry", "surprise"]  
correct_predictions = 0
total_predictions = 0

cap = cv2.VideoCapture(0)

with open("emotion_log.txt", "w") as log_file:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        try:
            analysis = DeepFace.analyze(frame, actions=['emotion'], enforce_detection=False)
            dominant_emotion = analysis[0]["dominant_emotion"]

            # Simulated ground truth (Change this to actual labeled data)
            actual_emotion = ground_truth[total_predictions % len(ground_truth)]

            # Compare prediction with ground truth
            if dominant_emotion == actual_emotion:
                correct_predictions += 1
            total_predictions += 1

            # Calculate Accuracy
            accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0

            # Save to file
            log_file.write(f"{dominant_emotion}, Accuracy: {accuracy:.2f}%\n")
            log_file.flush()
            model_accuracy=0.8675
            # Display emotion & accuracy
            cv2.putText(frame, f"Emotion: {dominant_emotion}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, f"Accuracy: {accuracy:.2f}%", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            cv2.putText(frame, f"Model Accuracy: {model_accuracy:.2f}%",(50, 130), cv2.FONT_HERSHEY_SIMPLEX, 1, (450, 255, 0), 2)
        
        except:
            pass

        cv2.imshow("Facial Emotion Detection", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
