import sys
import cv2
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QLabel,
                             QPushButton, QFileDialog, QVBoxLayout, QHBoxLayout,
                             QComboBox)
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
from AdvancedWatermark import AdvancedWatermark


class WatermarkSystem:
    def __init__(self):
        self.wm_engine = AdvancedWatermark()

    def embed_watermark(self, src_path, wm_path, output_path):
        # 读取原图和水印
        src_img = cv2.imread(src_path)
        wm_img = cv2.imread(wm_path, 0)

        # 预处理
        if src_img is None or wm_img is None:
            raise ValueError("无法读取图像文件")

        # 二值化水印
        _, wm_binary = cv2.threshold(wm_img, 127, 1, cv2.THRESH_BINARY)

        # 嵌入水印
        marked = self.wm_engine.embed(src_img, wm_binary)
        cv2.imwrite(output_path, marked)
        return marked

    def extract_watermark(self, marked_path):
        marked = cv2.imread(marked_path, 0)
        if marked is None:
            raise ValueError("无法读取带水印图像")

        # 自动识别水印尺寸（示例逻辑）
        h, w = marked.shape
        max_blocks = (h // 8) * (w // 8)
        wm_side = int(np.sqrt(max_blocks))
        wm_shape = (wm_side, wm_side)

        extracted = self.wm_engine.extract(marked, wm_shape)
        return extracted  # 直接返回已调整形状的结果


class WatermarkGUI(QMainWindow):
    def __init__(self):
        super().__init__()
        self.system = WatermarkSystem()
        self.initUI()
        self.current_img = None

    def initUI(self):
        # 主窗口设置
        self.setWindowTitle("抗截图水印系统 v1.1")
        self.setGeometry(100, 100, 800, 600)

        # 中央组件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # 布局
        main_layout = QVBoxLayout()
        control_layout = QHBoxLayout()

        # 图像显示区域
        self.img_label = QLabel()
        self.img_label.setAlignment(Qt.AlignCenter)
        self.img_label.setMinimumSize(640, 480)

        # 控制按钮
        self.btn_load = QPushButton("加载图片")
        self.btn_embed = QPushButton("嵌入水印")
        self.btn_extract = QPushButton("提取水印")
        self.attack_combo = QComboBox()
        self.attack_combo.addItems(["无攻击", "裁剪攻击", "缩放攻击", "旋转攻击"])

        # 布局组织
        control_layout.addWidget(self.btn_load)
        control_layout.addWidget(self.btn_embed)
        control_layout.addWidget(self.btn_extract)
        control_layout.addWidget(self.attack_combo)

        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.img_label)

        central_widget.setLayout(main_layout)

        # 信号连接
        self.btn_load.clicked.connect(self.load_image)
        self.btn_embed.clicked.connect(self.embed_watermark)
        self.btn_extract.clicked.connect(self.extract_watermark)

    def load_image(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "选择图片", "", "图像文件 (*.jpg *.png)")
        if path:
            self.current_img = path
            pixmap = QPixmap(path)
            self.img_label.setPixmap(pixmap.scaled(
                640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def embed_watermark(self):
        if not self.current_img:
            self.show_error("请先加载原始图片")
            return

        wm_path, _ = QFileDialog.getOpenFileName(
            self, "选择水印图片", "", "PNG文件 (*.png)")
        if not wm_path:
            return

        output_path, _ = QFileDialog.getSaveFileName(
            self, "保存结果", "", "JPEG文件 (*.jpg)")
        if output_path:
            try:
                marked = self.system.embed_watermark(
                    self.current_img, wm_path, output_path)
                self.show_image(marked)
            except Exception as e:
                self.show_error(f"嵌入失败: {str(e)}")

    def extract_watermark(self):
        if not self.current_img:
            self.show_error("请先加载待检测图片")
            return

        try:
            extracted = self.system.extract_watermark(self.current_img)
            self.show_image(extracted * 255, is_wm=True)
        except Exception as e:
            self.show_error(f"提取失败: {str(e)}")

    def show_image(self, img_array, is_wm=False):
        if is_wm:
            qimage = QImage(img_array.data, img_array.shape[1], img_array.shape[0],
                            img_array.strides[0], QImage.Format_Grayscale8)
        else:
            if len(img_array.shape) == 2:
                qimage = QImage(img_array.data, img_array.shape[1], img_array.shape[0],
                                img_array.strides[0], QImage.Format_Grayscale8)
            else:
                qimage = QImage(img_array.data, img_array.shape[1], img_array.shape[0],
                                img_array.strides[0], QImage.Format_BGR888)
        pixmap = QPixmap.fromImage(qimage)
        self.img_label.setPixmap(pixmap.scaled(
            640, 480, Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def show_error(self, message):
        self.img_label.setText(f"错误: {message}")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = WatermarkGUI()
    window.show()
    sys.exit(app.exec_())