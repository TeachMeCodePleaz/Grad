import cv2
import numpy as np
from sklearn.cluster import KMeans
from pywt import dwt2, idwt2
from skimage.feature import SIFT, ORB  # 改用ORB替代SURF（专利问题）


class AdvancedWatermark:
    def __init__(self, password=1234):
        self.dct_block_size = 8
        self.dwt_level = 2
        self.sift = SIFT()
        self.orb = ORB(n_keypoints=50)  # ORB替代SURF

        self.alpha_dct = 15.0
        self.alpha_spatial = 0.1
        self.keypoint_radius = 16

    def _get_feature_points(self, img):
        """获取稳定的特征点"""
        # SIFT检测
        self.sift.detect(img)
        sift_kps = self.sift.keypoints

        # ORB检测
        self.orb.detect(img)
        orb_kps = self.orb.keypoints

        # 合并特征点
        all_kps = np.vstack([sift_kps, orb_kps])
        return all_kps[:100]  # 取前100个关键点

    def _embed_spatial(self, img, wm, keypoints):
        """空间域嵌入"""
        mask = np.zeros_like(img, dtype=np.uint8)
        for kp in keypoints:
            x, y = map(int, kp)
            cv2.circle(mask, (x, y), self.keypoint_radius, 1, -1)

        img_flat = img.flatten()
        wm_ext = np.tile(wm, img_flat.size // wm.size + 1)[:img_flat.size]
        marked_flat = (img_flat & 0xFE) | (wm_ext & 0x01)
        return marked_flat.reshape(img.shape) * mask + img * (1 - mask)

    def _embed_frequency(self, img, wm):
        # 多级小波分解
        coeffs = []
        current = img.astype(np.float32)
        for _ in range(self.dwt_level):
            ca, (h, v, d) = dwt2(current, 'haar')
            coeffs.append((h, v, d))
            current = ca

        # 动态填充（关键修改）
        pad_h = (self.dct_block_size - current.shape[0] % self.dct_block_size) % self.dct_block_size
        pad_w = (self.dct_block_size - current.shape[1] % self.dct_block_size) % self.dct_block_size
        current_padded = np.pad(current,
                                ((0, pad_h), (0, pad_w)),
                                mode='symmetric')  # 对称填充效果更好

        # 分块处理
        blocks = []
        h_pad, w_pad = current_padded.shape
        wm_idx = 0
        for i in range(0, h_pad, self.dct_block_size):
            for j in range(0, w_pad, self.dct_block_size):
                block = current_padded[i:i + self.dct_block_size, j:j + self.dct_block_size]
                dct_block = cv2.dct(block)
                # 在中高频区域嵌入（提高鲁棒性）
                dct_block[4, 5] += self.alpha_dct * wm[wm_idx % len(wm)]
                blocks.append(cv2.idct(dct_block))
                wm_idx += 1

        # 重构图像（严格尺寸匹配）
        reconstructed = np.zeros_like(current_padded)
        idx = 0
        for i in range(0, h_pad, self.dct_block_size):
            for j in range(0, w_pad, self.dct_block_size):
                reconstructed[i:i + self.dct_block_size, j:j + self.dct_block_size] = blocks[idx]
                idx += 1

        # 去除填充
        reconstructed = reconstructed[:current.shape[0], :current.shape[1]]

        # 逆小波变换
        for i in range(self.dwt_level - 1, -1, -1):
            reconstructed = idwt2((reconstructed, coeffs[i]), 'haar')

        return np.clip(reconstructed, 0, 255).astype(np.uint8)

    def embed(self, img, wm):
        # 新增尺寸调整代码
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 计算适配分块和小波分解的尺寸
        block_size = self.dct_block_size * (2 ** self.dwt_level)
        h, w = img.shape
        new_h = ((h + block_size - 1) // block_size) * block_size
        new_w = ((w + block_size - 1) // block_size) * block_size

        # 边缘填充保持内容连续
        img = cv2.copyMakeBorder(img,
                                 0, new_h - h,
                                 0, new_w - w,
                                 cv2.BORDER_REFLECT)

        # 后续处理保持不变
        keypoints = self._get_feature_points(img)
        wm_binary = (wm > 0.5).astype(np.uint8).flatten()

        spatial_marked = self._embed_spatial(img, wm_binary, keypoints)
        frequency_marked = self._embed_frequency(img, wm_binary)

        return np.clip(0.7 * frequency_marked + 0.3 * spatial_marked, 0, 255).astype(np.uint8)

    def extract(self, marked_img):
        """联合提取入口"""
        # 频域提取
        ca, _ = dwt2(marked_img.astype(np.float32), 'haar')
        wm_freq = []
        h, w = ca.shape
        for i in range(0, h, self.dct_block_size):
            for j in range(0, w, self.dct_block_size):
                block = ca[i:i + self.dct_block_size, j:j + self.dct_block_size]
                if block.shape == (self.dct_block_size, self.dct_block_size):
                    dct_block = cv2.dct(block)
                    wm_freq.append(dct_block[3, 4] / self.alpha_dct)

        # 空间域提取
        keypoints = self._get_feature_points(marked_img)
        wm_spatial = []
        for kp in keypoints[:50]:  # 取前50个关键点
            x, y = map(int, kp)
            region = marked_img[y - self.keypoint_radius:y + self.keypoint_radius,
                     x - self.keypoint_radius:x + self.keypoint_radius]
            if region.size > 0:
                bits = (region.flatten() & 0x01)
                wm_spatial.extend(bits[:len(wm_freq) // 50])  # 均匀采样

        # 信息融合
        min_len = min(len(wm_freq), len(wm_spatial))
        wm_combined = 0.7 * np.array(wm_freq[:min_len]) + 0.3 * np.array(wm_spatial[:min_len])
        return (wm_combined > 0.5).astype(np.uint8)