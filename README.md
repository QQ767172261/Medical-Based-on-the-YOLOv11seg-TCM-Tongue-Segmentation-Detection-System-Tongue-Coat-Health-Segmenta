舌苔健康状态分割数据集，医学类-基于 YOLOv11-seg 中医舌象分割检测系统 舌苔健康状态分割数据集，![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/ba7ed7546b9441028d6ca6a89ee6b3dc.png)
2361张，yolo，voc，coco三种标注方式![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/fab7b4538f324da19083a9b6eaa81f3a.png)

图像尺寸:640*640
类别数量:6类![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/17bdac35c4194be5af501de714f5eb54.png)

训练集图像数量:1653; 验证集图像数量:471； 测试集图像数量:237
类别名称: 每一类图像数 ，每一类标注数
black tongue coating - 黑色舌苔：402, 402
map tongue coating - 地图舌苔：80, 80
purple tongue coating - 紫色舌苔：341, 341
red tongue yellow fur thick greasy fur - 红舌黄腻厚苔：752, 752
The red tongue is thick and greasy - 红舌厚腻：549, 549
The white tongue is thick and greasy - 白舌厚腻：237, 237
image num: 2361
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/63d6b7bf8bc04ebbb2971ce650b53163.png)


模型代码：模型训练使用yolov11n-seg训练，60个epoch训练结果，map如描述图所示。![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/b6aa0d97626a4177b8e7af96ea503e73.png)

qt界面：运行界面采用pyqt编写，本项目已经训练好模型，配置好环境后可直接使用，运行效果见描述图像![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/45ef296ff99c41c2b20d45e13de5a5d2.png)
# 基于YOLOv11-seg 中医舌象分割检测系统
## 1、数据集配置文件 tongue.yaml
```yaml
path: "./tongue_dataset"
train: images/train
val: images/val
test: images/test

names:
  0: black tongue coating        #黑苔
  1: map tongue coating         #地图舌
  2: purple tongue coating      #紫舌
  3: red tongue yellow fur thick greasy fur #红舌黄厚腻苔
  4: The red tongue is thick and greasy #红舌厚腻
  5: The white tongue is thick and greasy #白舌厚腻
```

## 2、训练脚本 train_tongue.py![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/f5537b67cebd4ed3965d4d9989b60437.png)

```python
from ultralytics import YOLO

if __name__ == '__main__':
    #加载YOLOv11分割预训练权重
    model = YOLO("yolo11n-seg.pt")
    results = model.train(
        data="tongue.yaml",
        epochs=100,
        imgsz=640,
        batch=8,
        device=0,
        workers=2,
        project="tongue_seg_result",
        name="yolo11_tongue",
        amp=True,
        mosaic=0.8,
        conf=0.25
    )
```

## 3、推理脚本 predict_tongue.py
```python
from ultralytics import YOLO
import cv2

model = YOLO("./tongue_seg_result/yolo11_tongue/weights/best.pt")

def tongue_detect(img_path, save_path="tongue_out.jpg"):
    res = model(img_path)[0]
    draw_img = res.plot()
    cv2.imwrite(save_path, draw_img)
    return draw_img, res.boxes, res.masks

if __name__ == "__main__":
    tongue_detect("test_tongue.jpg")
```

## 4、PyQt5可视化界面 tongue_gui.py
```python
import sys
import cv2
from PyQt5.QtWidgets import (QApplication, QMainWindow, QPushButton, QLabel, QFileDialog,
                             QTableWidget, QTableWidgetItem, QGroupBox, QVBoxLayout, QHBoxLayout, QWidget)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from ultralytics import YOLO

class TongueSegUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("YOLOv11-seg 舌苔健康状况分割系统")
        self.resize(1550, 900)
        self.model = YOLO("./tongue_seg_result/yolo11_tongue/weights/best.pt")
        self.init_ui()

    def init_ui(self):
        main_widget = QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QHBoxLayout(main_widget)

        # 左侧：原图+分割图展示
        left_layout = QVBoxLayout()
        img_layout = QHBoxLayout()
        self.label_origin = QLabel("原始图像")
        self.label_origin.setFixedSize(720, 600)
        self.label_seg = QLabel("分割结果")
        self.label_seg.setFixedSize(720, 600)
        img_layout.addWidget(self.label_origin)
        img_layout.addWidget(self.label_seg)
        left_layout.addLayout(img_layout)

        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["序号", "类别", "置信度", "坐标", "掩码"])
        left_layout.addWidget(self.table)

        # 右侧功能面板
        right_layout = QVBoxLayout()
        group_file = QGroupBox("检测操作")
        self.btn_img = QPushButton("选择图片")
        self.btn_img.clicked.connect(self.load_img)
        self.btn_video = QPushButton("选择视频")
        self.btn_cam = QPushButton("摄像头实时检测")
        self.btn_save = QPushButton("保存结果")
        v1 = QVBoxLayout()
        v1.addWidget(self.btn_img)
        v1.addWidget(self.btn_video)
        v1.addWidget(self.btn_cam)
        v1.addWidget(self.btn_save)
        group_file.setLayout(v1)

        group_info = QGroupBox("检测结果信息")
        self.label_time = QLabel("推理用时：0 s | 目标数量：0")
        v2 = QVBoxLayout()
        v2.addWidget(self.label_time)
        group_info.setLayout(v2)
        right_layout.addWidget(group_file)
        right_layout.addWidget(group_info)

        main_layout.addLayout(left_layout)
        main_layout.addLayout(right_layout)

    def load_img(self):
        fpath, _ = QFileDialog.getOpenFileName(self, "打开舌象图片", ".", "图片(*.jpg *.png *.jpeg)")
        if not fpath:
            return
        img_origin = cv2.imread(fpath)
        pred = self.model(fpath, conf=0.25)[0]
        seg_img = pred.plot()

        self.show_img(img_origin, self.label_origin)
        self.show_img(seg_img, self.label_seg)

        self.table.setRowCount(0)
        for idx, box in enumerate(pred.boxes):
            cls_name = pred.names[int(box.cls)]
            conf = float(box.conf)
            xyxy = box.xyxy[0].cpu().numpy()
            self.table.insertRow(idx)
            self.table.setItem(idx, 0, QTableWidgetItem(str(idx+1)))
            self.table.setItem(idx,1,QTableWidgetItem(cls_name))
            self.table.setItem(idx,2,QTableWidgetItem(f"{conf:.2f}"))
            self.table.setItem(idx,3,QTableWidgetItem(f"{xyxy.round(0)}"))
            self.table.setItem(idx,4,QTableWidgetItem("存在掩码"))
        self.label_time.setText(f"推理用时：{pred.speed['inference']/1000:.3f} s | 目标数目：{len(pred.boxes)}")

    def show_img(self,cv_img,label):
        rgb_img = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)
        h,w,c = rgb_img.shape
        qimg = QImage(rgb_img.data, w, h, w*c, QImage.Format_RGB888)
        label.setPixmap(QPixmap.fromImage(qimg).scaled(label.size(),Qt.KeepAspectRatio))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = TongueSegUI()
    win.show()
    sys.exit(app.exec_())
```

## 环境安装
```bash
pip install ultralytics torch opencv-python pyqt5
```
## 关键词标签
中医舌象分割、舌苔检测、YOLOv11-seg、舌体实例分割、舌色苔色识别、PyQt5可视化界面
## 应用场景
1. 中医辅助面诊，自动分割舌体区域，识别黑苔、地图舌、紫舌、黄厚腻苔等舌象类型
2. 移动端/PC端舌象拍照智能分析，输出分割掩码、类别与置信度
3. 舌象体检筛查，可接入摄像头实时采集舌图并自动检测
4. 中医教学数据集标注、舌象统计分析


