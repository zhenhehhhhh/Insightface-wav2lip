import os

import cv2
import insightface
import numpy as np
from sklearn import preprocessing


class FaceRecognition:
    def __init__(self, gpu_id=0, face_db='face_db', threshold=1.24, det_thresh=0.50, det_size=(640, 640)):
        """
        人脸识别工具类
        :param gpu_id: 正数为GPU的ID, 负数为使用CPU
        :param face_db: 人脸库文件夹
        :param threshold: 人脸识别阈值
        :param det_thresh: 检测阈值
        :param det_size: 检测模型图片大小
        """
        self.gpu_id = gpu_id
        self.face_db = face_db
        self.threshold = threshold
        self.det_thresh = det_thresh
        self.det_size = det_size
        self.people_size = 100

        # 加载人脸识别模型，当allowed_modules=['detection', 'recognition']时，只单纯检测和识别
        self.model = insightface.app.FaceAnalysis(root='./',
                                                  allowed_modules=None,
                                                  providers=['CPUExecutionProvider'])
        self.model.prepare(ctx_id=self.gpu_id, det_thresh=self.det_thresh, det_size=self.det_size)
        # 人脸库的人脸特征
        self.faces_embedding = list()

    # 加载人脸库中的人脸
    def load_faces(self, face_db_path):
        if not os.path.exists(face_db_path):
            os.makedirs(face_db_path)
        for root, dirs, files in os.walk(face_db_path):
            for file in files:
                input_image = cv2.imdecode(np.fromfile(os.path.join(root, file), dtype=np.uint8), 1)
                user_name = file.split(".")[0]
                face = self.model.get(input_image)[0]
                embedding = np.array(face.embedding).reshape((1, -1))
                embedding = preprocessing.normalize(embedding)
                self.faces_embedding.append({
                    "user_name": user_name,
                    "feature": embedding
                })

    # 人脸识别
    def recognition(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            # 开始人脸识别
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            user_name = "unknown"
            for com_face in self.faces_embedding:
                r = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r:
                    user_name = com_face["user_name"]
            results.append(user_name)
        return results

    @staticmethod
    def feature_compare(feature1, feature2, threshold):
        diff = np.subtract(feature1, feature2)
        dist = np.sum(np.square(diff), 1)
        if dist < threshold:
            return True
        else:
            return False

    def register(self, image):
        faces = self.model.get(image)
        if len(faces) == 0:
            return '图片检测不到人脸'
        
        result_reg = []
        # 判断人脸是否存在
        for i in range(len(faces)):
            embedding = np.array(faces[i].embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            is_exits = False
            for com_face in self.faces_embedding:
                r = self.feature_compare(embedding, com_face["feature"], self.threshold)
                if r:
                    is_exits = True
            if is_exits:
                result_reg.append('该用户已存在')
                continue
            # 符合注册条件保存图片，同时把特征添加到人脸特征库中
            user_name = '{}'.format(len(self.faces_embedding))
            cv2.imencode('.png', image)[1].tofile(os.path.join(self.face_db, '%s.png' % user_name))
            self.faces_embedding.append({
                "user_name": user_name,
                "feature": embedding
            })
            result_reg.append('success')
        if 'success' not in result_reg:
            return '用户已存在'
        else:
            return 'success'

    # 检测人脸
    def detect(self, image):
        faces = self.model.get(image)
        results = list()
        for face in faces:
            result = dict()
            # 获取人脸属性
            result["bbox"] = np.array(face.bbox).astype(np.int32).tolist()
            result["pose"] = np.array(face.pose).astype(np.int32).tolist()
            embedding = np.array(face.embedding).reshape((1, -1))
            embedding = preprocessing.normalize(embedding)
            result["embedding"] = embedding
            results.append(result)
        return results
