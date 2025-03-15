import cv2
import numpy as np
from sklearn.cluster import KMeans
from pywt import dwt2, idwt2
from skimage.feature import SIFT, ORB  # 改用ORB替代SURF（专利问题）
import hashlib


class AdvancedWatermark:
    def __init__(self, password=1234):
        self.dct_block_size = 8
        self.dwt_level = 2

        self.alpha_dct = 25.0   # 频域嵌入参数 提高强度
        self.alpha_spatial = 0.2    # 空间域嵌入参数 提高强度
        self.keypoint_radius = 16

        # 初始化密码系统
        self.seed = self._hash_password(password)
        self.rng = np.random.RandomState(self.seed)

        # 初始化特征检测器
        self.sift = SIFT()
        self.orb = ORB(n_keypoints=100)  # ORB替代SURF 提高强度

        self.pos_mask = None  # 新增保存掩码

    def _hash_password(self, password):
        """生成可重复随机种子"""
        return int(hashlib.sha256(str(password).encode()).hexdigest(), 16) % 2 ** 32

    def _get_feature_points(self, img):
        """基于密码的稳定特征点筛选"""
        # 检测特征点
        self.sift.detect(img)
        self.orb.detect(img)

        # 合并并随机筛选
        all_kps = np.vstack([self.sift.keypoints, self.orb.keypoints])
        self.rng.shuffle(all_kps)
        return all_kps[:100]  # 密码控制的筛选

    # 保存特征点信息-确保一致-有点问题需要修改
    # def save_keypoints(self, keypoints, filename):
    #     np.save(filename, keypoints)
    #
    # def load_keypoints(self, filename):
    #     return np.load(filename)
    #
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
        # 生成随机嵌入掩码
        pos_mask = self.rng.random(size=len(wm)) < 0.7  # 70%位置嵌入

        # 确保原图尺寸可被2^dwt_level整除
        original_h, original_w = img.shape
        factor = 2 ** self.dwt_level
        pad_h = (factor - (original_h % factor)) % factor
        pad_w = (factor - (original_w % factor)) % factor
        img_padded = np.pad(img, ((0, pad_h), (0, pad_w)), mode='symmetric')

        # 多级小波分解（*必须使用填充后的图像）
        coeffs = []
        current = img_padded.astype(np.float32)  # 使用填充后图像
        for _ in range(self.dwt_level):
            ca, (h, v, d) = dwt2(current, 'haar')
            # # 检查分解后尺寸的奇偶性（调试用）
            # print(f"Level {level}: CA {ca.shape}, H {h.shape}")
            coeffs.append((h, v, d))  # 存储顺序：第0个元素是第一层细节
            current = ca

        # *记录DCT前的原始低频尺寸
        original_low_freq_shape = current.shape  # 保存分解后的低频尺寸 (95,119)

        # DCT分块填充处理 *确保DCT块尺寸正确
        pad_h_dct = (self.dct_block_size - current.shape[0] % self.dct_block_size) % self.dct_block_size
        pad_w_dct = (self.dct_block_size - current.shape[1] % self.dct_block_size) % self.dct_block_size
        current_padded = np.pad(current, ((0, pad_h_dct), (0, pad_w_dct)), mode='symmetric')

        # 分块处理
        blocks = []
        h_pad, w_pad = current_padded.shape
        wm_idx = 0
        for i in range(0, h_pad, self.dct_block_size):
            for j in range(0, w_pad, self.dct_block_size):
                block = current_padded[i:i + self.dct_block_size, j:j + self.dct_block_size]
                dct_block = cv2.dct(block)
                # 修改后的嵌入部分
                if pos_mask[wm_idx % len(pos_mask)]:  # 根据掩码决定是否嵌入
                    # 将二值水印映射到[-1,1]提高鲁棒性
                    watermark_value = (wm[wm_idx % len(wm)] * 2) - 1
                    dct_block[4, 5] += self.alpha_dct * watermark_value

                blocks.append(cv2.idct(dct_block))
                wm_idx += 1

        # 重构图像-严格尺寸匹配
        reconstructed = np.zeros_like(current_padded)
        idx = 0
        for i in range(0, h_pad, self.dct_block_size):
            for j in range(0, w_pad, self.dct_block_size):
                reconstructed[i:i + self.dct_block_size, j:j + self.dct_block_size] = blocks[idx]
                idx += 1

        # 重构后裁剪回原始低频尺寸
        reconstructed = reconstructed[:original_low_freq_shape[0], :original_low_freq_shape[1]]  # 裁剪到(95,119)

        # 逆小波变换（*正确重建层级关系）
        current_reconstructed = reconstructed  # DCT处理后的低频部分
        for i in reversed(range(self.dwt_level)):  # 必须倒序访问coeffs
            h_level, v_level, d_level = coeffs[i]
            # 打印系数尺寸（调试用）
            print(f"Reconstruct Level {i}: CA {current_reconstructed.shape}, H {h_level.shape}")
            current_reconstructed = idwt2(
                (current_reconstructed, (h_level, v_level, d_level)),
                'haar'
            )

        # 最终裁剪（需考虑两次填充）
        final_reconstructed = current_reconstructed[:original_h, :original_w]  # 裁剪原始填充
        return np.clip(final_reconstructed, 0, 255).astype(np.uint8)


    def embed(self, img, wm):
        # 修改为YUV空间处理
        if len(img.shape) == 3:
            img_yuv = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            y_channel = img_yuv[:, :, 0].copy()  # 仅处理Y通道
        else:
            y_channel = img.copy()

        # 后续处理只在Y通道进行
        keypoints = self._get_feature_points(y_channel)
        wm_binary = (wm > 0.5).astype(np.uint8).flatten()

        spatial_marked = self._embed_spatial(y_channel, wm_binary, keypoints)
        frequency_marked = self._embed_frequency(y_channel, wm_binary)

        # 合并回彩色图像
        if len(img.shape) == 3:
            img_yuv[:, :, 0] = np.clip(0.7 * frequency_marked + 0.3 * spatial_marked, 0, 255)
            return cv2.cvtColor(img_yuv.astype(np.uint8), cv2.COLOR_YUV2BGR)
        else:
            return np.clip(0.7 * frequency_marked + 0.3 * spatial_marked, 0, 255).astype(np.uint8)

    def extract(self, marked_img, wm_shape=(64, 64)):
        """联合提取入口"""
        # 生成与嵌入一致的pos_mask
        pos_mask = self.rng.random(size=np.prod(wm_shape)) < 0.7

        # 频域提取
        ca, _ = dwt2(marked_img.astype(np.float32), 'haar')
        wm_freq = []
        h, w = ca.shape

        # 动态计算最大可提取块数
        max_blocks_h = h // self.dct_block_size
        max_blocks_w = w // self.dct_block_size
        max_blocks = max_blocks_h * max_blocks_w

        # 调整水印尺寸适配实际容量
        actual_wm_size = min(max_blocks, np.prod(wm_shape))
        wm_side = int(np.sqrt(actual_wm_size))  # 计算最大整数边
        actual_wm_size = wm_side ** 2  # 确保是完美平方数
        wm_shape = (wm_side, wm_side)

        # 频域提取  *仅提取被掩码标记的位置
        wm_freq = []
        blocks_extracted = 0
        total_blocks = 0  # 总扫描块数计数器
        for i in range(0, max_blocks_h * self.dct_block_size, self.dct_block_size):
            for j in range(0, max_blocks_w * self.dct_block_size, self.dct_block_size):
                block = ca[i:i + self.dct_block_size, j:j + self.dct_block_size]
                if block.shape != (self.dct_block_size, self.dct_block_size):
                    continue

                # 只处理掩码标记为True的块
                if total_blocks < len(pos_mask) and pos_mask[total_blocks]:
                    dct_block = cv2.dct(block)
                    wm_freq.append(dct_block[4, 5] / self.alpha_dct)
                    blocks_extracted += 1

                total_blocks += 1
                if blocks_extracted >= actual_wm_size:
                    break
            if blocks_extracted >= actual_wm_size:
                break

        # 水印值逆映射
        wm_freq = np.array(wm_freq)
        wm_freq = (wm_freq > 0).astype(np.uint8)  # 将[-1,1]映射回0/1

        # 空间域提取
        keypoints = self._get_feature_points(marked_img)  # 密码控制的特征点
        wm_spatial = []
        required_bits = actual_wm_size
        samples_per_kp = max(1, required_bits // max(1, len(keypoints[:50])))

        total_collected = 0
        for kp in keypoints[:50]:
            if total_collected >= required_bits:
                break
            x, y = map(int, kp)
            region = marked_img[y - self.keypoint_radius:y + self.keypoint_radius,
                     x - self.keypoint_radius:x + self.keypoint_radius]
            if region.size > 0:
                bits = (region.flatten() & 0x01)[:samples_per_kp]
                collect_num = min(samples_per_kp, required_bits - total_collected)
                wm_spatial.extend(bits[:collect_num])
                total_collected += collect_num
        wm_spatial = wm_spatial[:actual_wm_size]

        # 最终长度验证（*）
        min_len = min(len(wm_freq), len(wm_spatial))

        # 强制对齐到最大完美平方数
        wm_side = int(np.sqrt(min_len))  # 取整数边长
        actual_wm_size = wm_side ** 2  # 计算实际可用的数据量
        wm_freq = wm_freq[:actual_wm_size]
        wm_spatial = wm_spatial[:actual_wm_size]
        wm_shape = (wm_side, wm_side)

        # 信息融合
        wm_combined = 0.7 * np.array(wm_freq) + 0.3 * np.array(wm_spatial)
        return (wm_combined > 0.5).astype(np.uint8).reshape(wm_shape)

