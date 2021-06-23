import cv2
import mediapipe as mp
import time


class FindFaceMesh:
    def __init__(self, static_image_mode=False,
                 max_num_faces=2,
                 min_detection_confidence=0.5,
                 min_tracking_confidence=0.5):
        self.num_faces = max_num_faces

        self.mpface_mesh = mp.solutions.face_mesh
        self.mpDraw = mp.solutions.drawing_utils
        self.face_mesh = self.mpface_mesh.FaceMesh(max_num_faces=self.num_faces)
        self.drawspec = self.mpDraw.DrawingSpec(thickness=2, circle_radius=2)
    def find_Face_Mesh(self, img,Draw = True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.face_mesh.process(imgRGB)
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                if Draw:
                    self.mpDraw.draw_landmarks(img, facelms, self.mpface_mesh.FACE_CONNECTIONS,self.drawspec,self.drawspec)

    def find_pos(self,img,Draw=False):
        lmlit = []
        if self.results.multi_face_landmarks:
            for facelms in self.results.multi_face_landmarks:
                for id, facelm in enumerate(facelms.landmark):
                    h, w, c = img.shape
                    cx = int(facelm.x * w)
                    cy = int(facelm.y * h)
                    lmlit.append([id,cx,cy])
                    if Draw:
                        cv2.putText(img, str(id), (cx, cy), cv2.FONT_HERSHEY_PLAIN, 0.7, (255, 0, 255), 1)
        return lmlit



def main():
    finder = FindFaceMesh(max_num_faces=2)
    cap = cv2.VideoCapture('faceM3.mp4')
    ptime = 0
    lmlist = []
    while True:
        success, img = cap.read()
        finder.find_Face_Mesh(img)
        lmlist = finder.find_pos(img)
        if len(lmlist) != 0:
            print(lmlist)

        ctime = time.time()
        fps = 1 / (ctime - ptime)
        ptime = ctime
        cv2.putText(img, 'FPS=' + str(int(fps)), (50, 20), cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 255), 2)
        img = cv2.resize(img, (720, 720))
        cv2.imshow('img', img)
        if cv2.waitKey(1) == ord('q'):
            break


if __name__ == '__main__':
    main()
