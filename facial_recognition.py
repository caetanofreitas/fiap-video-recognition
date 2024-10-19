import cv2
from deepface import DeepFace
import mediapipe as mp
import torch
from tqdm import tqdm

def main():
  cap = cv2.VideoCapture('./video_facial_recognition.mp4')
  capture_actions(cap, './output-video_facial_recognition.mp4')


TEXT_SCALE = 0.6
COLOR = (88, 255, 5)
THICKNESS = 1
CONFIDENCE_INDICATOR = 0.5

yolo_model = torch.hub.load('ultralytics/yolov5', 'yolov5s')
yolo_model.classes=[0]
mp_pose = mp.solutions.pose  
pose = mp_pose.Pose(
  static_image_mode=True,
  min_detection_confidence=CONFIDENCE_INDICATOR,
  min_tracking_confidence=CONFIDENCE_INDICATOR)
mp_drawing = mp.solutions.drawing_utils
line_spec = mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=THICKNESS)
point_spec = mp_drawing.DrawingSpec(color=(220, 100, 0), thickness=-THICKNESS) 

def capture_actions(cap, output_path, isImage = False):
  if not isImage and not cap.isOpened():
    print('Erro ao acessar conteúdo')
    return

  total_frames = 1
  if not isImage:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

  for _ in tqdm(range(total_frames), desc='Processando conteúdo'):
    if isImage:
      frame = cap.copy()
      frame = processFrame(frame)
      cv2.imwrite(output_path, frame)
    else:
      ret, frame = cap.read()
      if not ret:
        break

      frame = processFrame(frame)
      out.write(frame)
  
  if not isImage:
    cap.release()
    out.release()
    cv2.destroyAllWindows()

def processFrame(frame):
  rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
  yoloResult = yolo_model(rgb_frame)
  
  frameMap = {}
  for (xmin, ymin, xmax, ymax, _, _) in yoloResult.xyxy[0].tolist():
    person = frame[int(ymin):int(ymax),int(xmin):int(xmax):].copy()
    person.flags.writeable = False
    resultPoses = pose.process(person)
    resultEmotion = DeepFace.analyze(person,
                                    actions=['emotion'],
                                    enforce_detection=False)
    person.flags.writeable = True
    for face in resultEmotion:
      x, y, w, h = face['region']['x'], face['region']['y'], face['region']['w'], face['region']['h']
      dominant_emotion = face['dominant_emotion']
      cv2.rectangle(person, (x, y), (x+w, y+h), COLOR, THICKNESS)
      cv2.putText(person, dominant_emotion, (x + 2, y+h - 2), cv2.FONT_HERSHEY_SIMPLEX, TEXT_SCALE, COLOR, THICKNESS) 
    
    if resultPoses.pose_landmarks:
      mp_drawing.draw_landmarks(person,
                                landmark_list=resultPoses.pose_landmarks,
                                connections=mp_pose.POSE_CONNECTIONS,
                                connection_drawing_spec=line_spec,
                                landmark_drawing_spec=point_spec)
    frameMap[int(ymin):int(ymax),int(xmin):int(xmax):] = person

  if len(frameMap) > 0:
    for key in frameMap:
      frame[key] = frameMap[key]
  
  return frame


if __name__ == "__main__":
  main()