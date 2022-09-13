import os, sys
import numpy as np
from numpy import cos, sin, arctan, sqrt
import cv2
import copy
import time

def Homography(image, img_points, world_width, world_height,
               interpolation=cv2.INTER_CUBIC, ratio_width=1.0, ratio_height=1.0):
    _points = np.array(img_points).reshape(-1, 2).astype(np.float32)

    expand_x = int(0.5 * world_width * (ratio_width - 1))
    expand_y = int(0.5 * world_height * (ratio_height - 1))

    pt_lefttop = [expand_x, expand_y]
    pt_righttop = [expand_x + world_width, expand_y]
    pt_leftbottom = [expand_x + world_width, expand_y + world_height]
    pt_rightbottom = [expand_x, expand_y + world_height]

    pts_std = np.float32([pt_lefttop, pt_righttop,
                          pt_leftbottom, pt_rightbottom])

    img_crop_width = int(world_width * ratio_width)
    img_crop_height = int(world_height * ratio_height)

    M = cv2.getPerspectiveTransform(_points, pts_std)

    dst_img = cv2.warpPerspective(
        image,
        M, (img_crop_width, img_crop_height),
        borderMode=cv2.BORDER_CONSTANT,  # BORDER_CONSTANT BORDER_REPLICATE
        flags=interpolation)

    return dst_img

class CurveTextRectifier:
    """
    spatial transformer via monocular vision
    """
    def __init__(self):
        self.get_virtual_camera_parameter()


    def get_virtual_camera_parameter(self):
        vcam_thz = 0
        vcam_thx1 = 180
        vcam_thy = 180
        vcam_thx2 = 0

        vcam_x = 0
        vcam_y = 0
        vcam_z = 100

        radian = np.pi / 180

        angle_z = radian * vcam_thz
        angle_x1 = radian * vcam_thx1
        angle_y = radian * vcam_thy
        angle_x2 = radian * vcam_thx2

        optic_x = vcam_x
        optic_y = vcam_y
        optic_z = vcam_z

        fu = 100
        fv = 100

        matT = np.zeros((4, 4))
        matT[0, 0] = cos(angle_z) * cos(angle_y) - sin(angle_z) * sin(angle_x1) * sin(angle_y)
        matT[0, 1] = cos(angle_z) * sin(angle_y) * sin(angle_x2) - sin(angle_z) * (
                    cos(angle_x1) * cos(angle_x2) - sin(angle_x1) * cos(angle_y) * sin(angle_x2))
        matT[0, 2] = cos(angle_z) * sin(angle_y) * cos(angle_x2) + sin(angle_z) * (
                    cos(angle_x1) * sin(angle_x2) + sin(angle_x1) * cos(angle_y) * cos(angle_x2))
        matT[0, 3] = optic_x
        matT[1, 0] = sin(angle_z) * cos(angle_y) + cos(angle_z) * sin(angle_x1) * sin(angle_y)
        matT[1, 1] = sin(angle_z) * sin(angle_y) * sin(angle_x2) + cos(angle_z) * (
                    cos(angle_x1) * cos(angle_x2) - sin(angle_x1) * cos(angle_y) * sin(angle_x2))
        matT[1, 2] = sin(angle_z) * sin(angle_y) * cos(angle_x2) - cos(angle_z) * (
                    cos(angle_x1) * sin(angle_x2) + sin(angle_x1) * cos(angle_y) * cos(angle_x2))
        matT[1, 3] = optic_y
        matT[2, 0] = -cos(angle_x1) * sin(angle_y)
        matT[2, 1] = cos(angle_x1) * cos(angle_y) * sin(angle_x2) + sin(angle_x1) * cos(angle_x2)
        matT[2, 2] = cos(angle_x1) * cos(angle_y) * cos(angle_x2) - sin(angle_x1) * sin(angle_x2)
        matT[2, 3] = optic_z
        matT[3, 0] = 0
        matT[3, 1] = 0
        matT[3, 2] = 0
        matT[3, 3] = 1

        matS = np.zeros((4, 4))
        matS[2, 3] = 0.5
        matS[3, 2] = 0.5

        self.ifu = 1 / fu
        self.ifv = 1 / fv

        self.matT = matT
        self.matS = matS
        self.K = np.dot(matT.T, matS)
        self.K = np.dot(self.K, matT)


    def vertical_text_process(self, points, org_size):
        """
        change sequence amd process
        :param points:
        :param org_size:
        :return:
        """
        org_w, org_h = org_size
        _points = np.array(points).reshape(-1).tolist()
        _points = np.array(_points[2:] + _points[:2]).reshape(-1, 2)

        # convert to horizontal points
        adjusted_points = np.zeros(_points.shape, dtype=np.float32)
        adjusted_points[:, 0] = _points[:, 1]
        adjusted_points[:, 1] = org_h - _points[:, 0] - 1

        _image_coord, _world_coord, _new_image_size = self.horizontal_text_process(adjusted_points)

        # # convert to vertical points back
        image_coord = _points.reshape(1, -1, 2)
        world_coord = np.zeros(_world_coord.shape, dtype=np.float32)
        world_coord[:, :, 0] = 0 - _world_coord[:, :, 1]
        world_coord[:, :, 1] = _world_coord[:, :, 0]
        world_coord[:, :, 2] = _world_coord[:, :, 2]
        new_image_size = (_new_image_size[1], _new_image_size[0])

        return image_coord, world_coord, new_image_size


    def horizontal_text_process(self, points):
        """
        get image coordinate and world coordinate
        :param points:
        :return:
        """
        poly = np.array(points).reshape(-1)

        dx_list = []
        dy_list = []
        for i in range(1, len(poly) // 2):
            xdx = poly[i * 2] - poly[(i - 1) * 2]
            xdy = poly[i * 2 + 1] - poly[(i - 1) * 2 + 1]
            d = sqrt(xdx ** 2 + xdy ** 2)
            dx_list.append(d)

        for i in range(0, len(poly) // 4):
            ydx = poly[i * 2] - poly[len(poly) - 1 - (i * 2 + 1)]
            ydy = poly[i * 2 + 1] - poly[len(poly) - 1 - (i * 2)]
            d = sqrt(ydx ** 2 + ydy ** 2)
            dy_list.append(d)

        dx_list = [(dx_list[i] + dx_list[len(dx_list) - 1 - i]) / 2 for i in range(len(dx_list) // 2)]

        height = np.around(np.mean(dy_list))

        rect_coord = [0, 0]
        for i in range(0, len(poly) // 4 - 1):
            x = rect_coord[-2]
            x += dx_list[i]
            y = 0
            rect_coord.append(x)
            rect_coord.append(y)

        rect_coord_half = copy.deepcopy(rect_coord)
        for i in range(0, len(poly) // 4):
            x = rect_coord_half[len(rect_coord_half) - 2 * i - 2]
            y = height
            rect_coord.append(x)
            rect_coord.append(y)

        np_rect_coord = np.array(rect_coord).reshape(-1, 2)
        x_min = np.min(np_rect_coord[:, 0])
        y_min = np.min(np_rect_coord[:, 1])
        x_max = np.max(np_rect_coord[:, 0])
        y_max = np.max(np_rect_coord[:, 1])
        new_image_size = (int(x_max - x_min + 0.5), int(y_max - y_min + 0.5))
        x_mean = (x_max - x_min) / 2
        y_mean = (y_max - y_min) / 2
        np_rect_coord[:, 0] -= x_mean
        np_rect_coord[:, 1] -= y_mean
        rect_coord = np_rect_coord.reshape(-1).tolist()

        rect_coord = np.array(rect_coord).reshape(-1, 2)
        world_coord = np.ones((len(rect_coord), 3)) * 0

        world_coord[:, :2] = rect_coord

        image_coord = np.array(poly).reshape(1, -1, 2)
        world_coord = world_coord.reshape(1, -1, 3)

        return image_coord, world_coord, new_image_size


    def horizontal_text_estimate(self, points):
        """
        horizontal or vertical text
        :param points:
        :return:
        """
        pts = np.array(points).reshape(-1, 2)
        x_min = int(np.min(pts[:, 0]))
        y_min = int(np.min(pts[:, 1]))
        x_max = int(np.max(pts[:, 0]))
        y_max = int(np.max(pts[:, 1]))
        x = x_max - x_min
        y = y_max - y_min
        is_horizontal_text = True
        if y / x > 1.5: # vertical text condition
            is_horizontal_text = False
        return is_horizontal_text


    def virtual_camera_to_world(self, size):
        ifu, ifv = self.ifu, self.ifv
        K, matT = self.K, self.matT

        ppu = size[0] / 2 + 1e-6
        ppv = size[1] / 2 + 1e-6

        P = np.zeros((size[1], size[0], 3))

        lu = np.array([i for i in range(size[0])])
        lv = np.array([i for i in range(size[1])])
        u, v = np.meshgrid(lu, lv)

        yp = (v - ppv) * ifv
        xp = (u - ppu) * ifu
        angle_a = arctan(sqrt(xp * xp + yp * yp))
        angle_b = arctan(yp / xp)

        D0 = sin(angle_a) * cos(angle_b)
        D1 = sin(angle_a) * sin(angle_b)
        D2 = cos(angle_a)

        D0[xp <= 0] = -D0[xp <= 0]
        D1[xp <= 0] = -D1[xp <= 0]

        ratio_a = K[0, 0] * D0 * D0 + K[1, 1] * D1 * D1 + K[2, 2] * D2 * D2 + \
                  (K[0, 1] + K[1, 0]) * D0 * D1 + (K[0, 2] + K[2, 0]) * D0 * D2 + (K[1, 2] + K[2, 1]) * D1 * D2
        ratio_b = (K[0, 3] + K[3, 0]) * D0 + (K[1, 3] + K[3, 1]) * D1 + (K[2, 3] + K[3, 2]) * D2
        ratio_c = K[3, 3] * np.ones(ratio_b.shape)

        delta = ratio_b * ratio_b - 4 * ratio_a * ratio_c
        t = np.zeros(delta.shape)
        t[ratio_a == 0] = -ratio_c[ratio_a == 0] / ratio_b[ratio_a == 0]
        t[ratio_a != 0] = (-ratio_b[ratio_a != 0] + sqrt(delta[ratio_a != 0])) / (2 * ratio_a[ratio_a != 0])
        t[delta < 0] = 0

        P[:, :, 0] = matT[0, 3] + t * (matT[0, 0] * D0 + matT[0, 1] * D1 + matT[0, 2] * D2)
        P[:, :, 1] = matT[1, 3] + t * (matT[1, 0] * D0 + matT[1, 1] * D1 + matT[1, 2] * D2)
        P[:, :, 2] = matT[2, 3] + t * (matT[2, 0] * D0 + matT[2, 1] * D1 + matT[2, 2] * D2)

        return P


    def world_to_image(self, image_size, world, intrinsic, distCoeffs, rotation, tvec):
        r11 = rotation[0, 0]
        r12 = rotation[0, 1]
        r13 = rotation[0, 2]
        r21 = rotation[1, 0]
        r22 = rotation[1, 1]
        r23 = rotation[1, 2]
        r31 = rotation[2, 0]
        r32 = rotation[2, 1]
        r33 = rotation[2, 2]

        t1 = tvec[0]
        t2 = tvec[1]
        t3 = tvec[2]

        k1 = distCoeffs[0]
        k2 = distCoeffs[1]
        p1 = distCoeffs[2]
        p2 = distCoeffs[3]
        k3 = distCoeffs[4]
        k4 = distCoeffs[5]
        k5 = distCoeffs[6]
        k6 = distCoeffs[7]

        if len(distCoeffs) > 8:
            s1 = distCoeffs[8]
            s2 = distCoeffs[9]
            s3 = distCoeffs[10]
            s4 = distCoeffs[11]
        else:
            s1 = s2 = s3 = s4 = 0

        if len(distCoeffs) > 12:
            tx = distCoeffs[12]
            ty = distCoeffs[13]
        else:
            tx = ty = 0

        fu = intrinsic[0, 0]
        fv = intrinsic[1, 1]
        ppu = intrinsic[0, 2]
        ppv = intrinsic[1, 2]

        cos_tx = cos(tx)
        cos_ty = cos(ty)
        sin_tx = sin(tx)
        sin_ty = sin(ty)

        tao11 = cos_ty * cos_tx * cos_ty + sin_ty * cos_tx * sin_ty
        tao12 = cos_ty * cos_tx * sin_ty * sin_tx - sin_ty * cos_tx * cos_ty * sin_tx
        tao13 = -cos_ty * cos_tx * sin_ty * cos_tx + sin_ty * cos_tx * cos_ty * cos_tx
        tao21 = -sin_tx * sin_ty
        tao22 = cos_ty * cos_tx * cos_tx + sin_tx * cos_ty * sin_tx
        tao23 = cos_ty * cos_tx * sin_tx - sin_tx * cos_ty * cos_tx

        P = np.zeros((image_size[1], image_size[0], 2))

        c3 = r31 * world[:, :, 0] + r32 * world[:, :, 1] + r33 * world[:, :, 2] + t3
        c1 = r11 * world[:, :, 0] + r12 * world[:, :, 1] + r13 * world[:, :, 2] + t1
        c2 = r21 * world[:, :, 0] + r22 * world[:, :, 1] + r23 * world[:, :, 2] + t2

        x1 = c1 / c3
        y1 = c2 / c3
        x12 = x1 * x1
        y12 = y1 * y1
        x1y1 = 2 * x1 * y1
        r2 = x12 + y12
        r4 = r2 * r2
        r6 = r2 * r4

        radial_distortion = (1 + k1 * r2 + k2 * r4 + k3 * r6) / (1 + k4 * r2 + k5 * r4 + k6 * r6)
        x2 = x1 * radial_distortion + p1 * x1y1 + p2 * (r2 + 2 * x12) + s1 * r2 + s2 * r4
        y2 = y1 * radial_distortion + p2 * x1y1 + p1 * (r2 + 2 * y12) + s3 * r2 + s4 * r4

        x3 = tao11 * x2 + tao12 * y2 + tao13
        y3 = tao21 * x2 + tao22 * y2 + tao23

        P[:, :, 0] = fu * x3 + ppu
        P[:, :, 1] = fv * y3 + ppv
        P[c3 <= 0] = 0

        return P


    def spatial_transform(self, image_data, new_image_size, mtx, dist, rvecs, tvecs, interpolation):
        rotation, _ = cv2.Rodrigues(rvecs)
        world_map = self.virtual_camera_to_world(new_image_size)
        image_map = self.world_to_image(new_image_size, world_map, mtx, dist, rotation, tvecs)
        image_map = image_map.astype(np.float32)
        dst = cv2.remap(image_data, image_map[:, :, 0], image_map[:, :, 1], interpolation)
        return dst


    def calibrate(self, org_size, image_coord, world_coord):
        """
        calibration
        :param org_size:
        :param image_coord:
        :param world_coord:
        :return:
        """
        # flag = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_TILTED_MODEL  | cv2.CALIB_THIN_PRISM_MODEL
        flag = cv2.CALIB_RATIONAL_MODEL
        flag2 = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_TILTED_MODEL
        flag3 = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_THIN_PRISM_MODEL
        flag4 = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_ASPECT_RATIO
        flag5 = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_TILTED_MODEL | cv2.CALIB_ZERO_TANGENT_DIST
        flag6 = cv2.CALIB_RATIONAL_MODEL | cv2.CALIB_FIX_ASPECT_RATIO
        flag_list = [flag2, flag3, flag4, flag5, flag6]

        ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(world_coord.astype(np.float32),
                                                                image_coord.astype(np.float32),
                                                                org_size,
                                                                None,
                                                                None,
                                                                flags=flag)
        if ret > 2:
            # strategies
            min_ret = ret
            for i, flag in enumerate(flag_list):
                _ret, _mtx, _dist, _rvecs, _tvecs = cv2.calibrateCamera(world_coord.astype(np.float32),
                                                                   image_coord.astype(np.float32),
                                                                   org_size,
                                                                   None,
                                                                   None,
                                                                   flags=flag)
                if _ret < min_ret:
                    min_ret = _ret
                    ret, mtx, dist, rvecs, tvecs = _ret, _mtx, _dist, _rvecs, _tvecs

        return ret, mtx, dist, rvecs, tvecs


    def dc_homo(self, img, img_points, obj_points, is_horizontal_text, interpolation=cv2.INTER_LINEAR,
                ratio_width=1.0, ratio_height=1.0):
        """
        divide and conquer: homography
        # ratio_width and ratio_height must be 1.0 here
        """
        _img_points = img_points.reshape(-1, 2)
        _obj_points = obj_points.reshape(-1, 3)

        homo_img_list = []
        width_list = []
        height_list = []
        # divide and conquer
        for i in range(len(_img_points) // 2 - 1):
            new_img_points = np.zeros((4, 2)).astype(np.float32)
            new_obj_points = np.zeros((4, 2)).astype(np.float32)

            new_img_points[0:2, :] = _img_points[i:(i + 2), :2]
            new_img_points[2:4, :] = _img_points[::-1, :][i:(i + 2), :2][::-1, :]

            new_obj_points[0:2, :] = _obj_points[i:(i + 2), :2]
            new_obj_points[2:4, :] = _obj_points[::-1, :][i:(i + 2), :2][::-1, :]

            if is_horizontal_text:
                world_width = np.abs(new_obj_points[1, 0] - new_obj_points[0, 0])
                world_height = np.abs(new_obj_points[3, 1] - new_obj_points[0, 1])
            else:
                world_width = np.abs(new_obj_points[1, 1] - new_obj_points[0, 1])
                world_height = np.abs(new_obj_points[3, 0] - new_obj_points[0, 0])

            homo_img = Homography(img, new_img_points, world_width, world_height,
                                              interpolation=interpolation,
                                              ratio_width=ratio_width, ratio_height=ratio_height)

            homo_img_list.append(homo_img)
            _h, _w = homo_img.shape[:2]
            width_list.append(_w)
            height_list.append(_h)

        # stitching
        rectified_image = np.zeros((np.max(height_list), sum(width_list), 3)).astype(np.uint8)

        st = 0
        for (homo_img, w, h) in zip(homo_img_list, width_list, height_list):
            rectified_image[:h, st:st + w, :] = homo_img
            st += w

        if not is_horizontal_text:
            # vertical rotation
            rectified_image = np.rot90(rectified_image, 3)

        return rectified_image


    def __call__(self, image_data, points, interpolation=cv2.INTER_LINEAR, ratio_width=1.0, ratio_height=1.0, mode='calibration'):
        """
        spatial transform for a poly text
        :param image_data:
        :param points: [x1,y1,x2,y2,x3,y3,...], clockwise order, (x1,y1) must be the top-left of first char.
        :param interpolation: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        :param ratio_width:  roi_image width expansion. It should not be smaller than 1.0
        :param ratio_height: roi_image height expansion. It should not be smaller than 1.0
        :param mode: 'calibration' or 'homography'. when homography, ratio_width and ratio_height must be 1.0
        :return:
        """
        org_h, org_w = image_data.shape[:2]
        org_size = (org_w, org_h)
        self.image = image_data

        is_horizontal_text = self.horizontal_text_estimate(points)
        if is_horizontal_text:
            image_coord, world_coord, new_image_size = self.horizontal_text_process(points)
        else:
            image_coord, world_coord, new_image_size = self.vertical_text_process(points, org_size)

        if mode.lower() == 'calibration':
            ret, mtx, dist, rvecs, tvecs = self.calibrate(org_size, image_coord, world_coord)

            st_size = (int(new_image_size[0]*ratio_width), int(new_image_size[1]*ratio_height))
            dst = self.spatial_transform(image_data, st_size, mtx, dist[0], rvecs[0], tvecs[0], interpolation)
        elif mode.lower() == 'homography':
            # ratio_width and ratio_height must be 1.0 here and ret set to 0.01 without loss manually
            ret = 0.01
            dst = self.dc_homo(image_data, image_coord, world_coord, is_horizontal_text,
                               interpolation=interpolation, ratio_width=1.0, ratio_height=1.0)
        else:
            raise ValueError('mode must be ["calibration", "homography"], but got {}'.format(mode))

        return dst, ret


class PlanB:
    def __call__(self, image, points, curveTextRectifier, interpolation=cv2.INTER_LINEAR,
                 ratio_width=1.0, ratio_height=1.0, loss_thresh=5.0, square=False):
        """
        Plan B using sub-image when it failed in original image
        :param image:
        :param points:
        :param curveTextRectifier: CurveTextRectifier
        :param interpolation: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        :param ratio_width:  roi_image width expansion. It should not be smaller than 1.0
        :param ratio_height: roi_image height expansion. It should not be smaller than 1.0
        :param loss_thresh: if loss greater than loss_thresh --> get_rotate_crop_image
        :param square: crop square image or not. True or False. The default is False
        :return:
        """
        h, w = image.shape[:2]
        _points = np.array(points).reshape(-1, 2).astype(np.float32)
        x_min = int(np.min(_points[:, 0]))
        y_min = int(np.min(_points[:, 1]))
        x_max = int(np.max(_points[:, 0]))
        y_max = int(np.max(_points[:, 1]))
        dx = x_max - x_min
        dy = y_max - y_min
        max_d = max(dx, dy)
        mean_pt = np.mean(_points, 0)

        expand_x = (ratio_width - 1.0) * 0.5 * max_d
        expand_y = (ratio_height - 1.0) * 0.5 * max_d

        if square:
            x_min = np.clip(int(mean_pt[0] - max_d - expand_x), 0, w - 1)
            y_min = np.clip(int(mean_pt[1] - max_d - expand_y), 0, h - 1)
            x_max = np.clip(int(mean_pt[0] + max_d + expand_x), 0, w - 1)
            y_max = np.clip(int(mean_pt[1] + max_d + expand_y), 0, h - 1)
        else:
            x_min = np.clip(int(x_min - expand_x), 0, w - 1)
            y_min = np.clip(int(y_min - expand_y), 0, h - 1)
            x_max = np.clip(int(x_max + expand_x), 0, w - 1)
            y_max = np.clip(int(y_max + expand_y), 0, h - 1)

        new_image = image[y_min:y_max, x_min:x_max, :].copy()
        new_points = _points.copy()
        new_points[:, 0] -= x_min
        new_points[:, 1] -= y_min

        dst_img, loss = curveTextRectifier(new_image, new_points, interpolation, ratio_width, ratio_height, mode='calibration')

        return dst_img, loss


class AutoRectifier:
    def __init__(self):
        self.npoints = 10
        self.curveTextRectifier = CurveTextRectifier()

    @staticmethod
    def get_rotate_crop_image(img, points, interpolation=cv2.INTER_CUBIC, ratio_width=1.0, ratio_height=1.0):
        """
        crop or homography
        :param img:
        :param points:
        :param interpolation:
        :param ratio_width:
        :param ratio_height:
        :return:
        """
        h, w = img.shape[:2]
        _points = np.array(points).reshape(-1, 2).astype(np.float32)

        if len(_points) != 4:
            x_min = int(np.min(_points[:, 0]))
            y_min = int(np.min(_points[:, 1]))
            x_max = int(np.max(_points[:, 0]))
            y_max = int(np.max(_points[:, 1]))
            dx = x_max - x_min
            dy = y_max - y_min
            expand_x = int(0.5 * dx * (ratio_width - 1))
            expand_y = int(0.5 * dy * (ratio_height - 1))
            x_min = np.clip(int(x_min - expand_x), 0, w - 1)
            y_min = np.clip(int(y_min - expand_y), 0, h - 1)
            x_max = np.clip(int(x_max + expand_x), 0, w - 1)
            y_max = np.clip(int(y_max + expand_y), 0, h - 1)

            dst_img = img[y_min:y_max, x_min:x_max, :].copy()
        else:
            img_crop_width = int(
                max(
                    np.linalg.norm(_points[0] - _points[1]),
                    np.linalg.norm(_points[2] - _points[3])))
            img_crop_height = int(
                max(
                    np.linalg.norm(_points[0] - _points[3]),
                    np.linalg.norm(_points[1] - _points[2])))

            dst_img = Homography(img, _points, img_crop_width, img_crop_height, interpolation, ratio_width, ratio_height)

        return dst_img


    def visualize(self, image_data, points_list):
        visualization = image_data.copy()

        for box in points_list:
            box = np.array(box).reshape(-1, 2).astype(np.int32)
            cv2.drawContours(visualization, [np.array(box).reshape((-1, 1, 2))], -1, (0, 0, 255), 2)
            for i, p in enumerate(box):
                if i != 0:
                    cv2.circle(visualization, tuple(p), radius=1, color=(255, 0, 0), thickness=2)
                else:
                    cv2.circle(visualization, tuple(p), radius=1, color=(255, 255, 0), thickness=2)
        return visualization


    def __call__(self, image_data, points, interpolation=cv2.INTER_LINEAR,
                 ratio_width=1.0, ratio_height=1.0, loss_thresh=5.0, mode='calibration'):
        """
        rectification in strategies for a poly text
        :param image_data:
        :param points: [x1,y1,x2,y2,x3,y3,...], clockwise order, (x1,y1) must be the top-left of first char.
        :param interpolation: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        :param ratio_width:  roi_image width expansion. It should not be smaller than 1.0
        :param ratio_height: roi_image height expansion. It should not be smaller than 1.0
        :param loss_thresh: if loss greater than loss_thresh --> get_rotate_crop_image
        :param mode: 'calibration' or 'homography'. when homography, ratio_width and ratio_height must be 1.0
        :return:
        """
        _points = np.array(points).reshape(-1,2)
        if len(_points) >= self.npoints and len(_points) % 2 == 0:
            try:
                curveTextRectifier = CurveTextRectifier()

                dst_img, loss = curveTextRectifier(image_data, points, interpolation, ratio_width, ratio_height, mode)
                if loss >= 2:
                    # for robust
                    # large loss means it cannot be reconstruct correctly, we must find other way to reconstruct
                    img_list, loss_list = [dst_img], [loss]
                    _dst_img, _loss = PlanB()(image_data, points, curveTextRectifier,
                                              interpolation, ratio_width, ratio_height,
                                              loss_thresh=loss_thresh,
                                              square=True)
                    img_list += [_dst_img]
                    loss_list += [_loss]

                    _dst_img, _loss = PlanB()(image_data, points, curveTextRectifier,
                                              interpolation, ratio_width, ratio_height,
                                              loss_thresh=loss_thresh, square=False)
                    img_list += [_dst_img]
                    loss_list += [_loss]

                    min_loss = min(loss_list)
                    dst_img = img_list[loss_list.index(min_loss)]

                    if min_loss >= loss_thresh:
                        print('calibration loss: {} is too large for spatial transformer. It is failed. Using get_rotate_crop_image'.format(loss))
                        dst_img = self.get_rotate_crop_image(image_data, points, interpolation, ratio_width, ratio_height)
                        print('here')
            except Exception as e:
                print(e)
                dst_img = self.get_rotate_crop_image(image_data, points, interpolation, ratio_width, ratio_height)
        else:
            dst_img = self.get_rotate_crop_image(image_data, _points, interpolation, ratio_width, ratio_height)

        return dst_img


    def run(self, image_data, points_list, interpolation=cv2.INTER_LINEAR,
            ratio_width=1.0, ratio_height=1.0, loss_thresh=5.0, mode='calibration'):
        """
        run for texts in an image
        :param image_data: numpy.ndarray. The shape is [h, w, 3]
        :param points_list: [[x1,y1,x2,y2,x3,y3,...], [x1,y1,x2,y2,x3,y3,...], ...], clockwise order, (x1,y1) must be the top-left of first char.
        :param interpolation: cv2.INTER_NEAREST, cv2.INTER_LINEAR, cv2.INTER_AREA, cv2.INTER_CUBIC, cv2.INTER_LANCZOS4
        :param ratio_width:  roi_image width expansion. It should not be smaller than 1.0
        :param ratio_height: roi_image height expansion. It should not be smaller than 1.0
        :param loss_thresh: if loss greater than loss_thresh --> get_rotate_crop_image
        :param mode: 'calibration' or 'homography'. when homography, ratio_width and ratio_height must be 1.0
        :return: res: roi-image list, visualized_image: draw polys in original image
        """
        if image_data is None:
            raise ValueError
        if not isinstance(points_list, list):
            raise ValueError
        for points in points_list:
            if not isinstance(points, list):
                raise ValueError

        if ratio_width < 1.0 or ratio_height < 1.0:
            raise ValueError('ratio_width and ratio_height cannot be smaller than 1, but got {}', (ratio_width, ratio_height))

        if mode.lower() != 'calibration' and mode.lower() != 'homography':
            raise ValueError('mode must be ["calibration", "homography"], but got {}'.format(mode))

        if mode.lower() == 'homography' and ratio_width != 1.0 and ratio_height != 1.0:
            raise ValueError('ratio_width and ratio_height must be 1.0 when mode is homography, but got mode:{}, ratio:({},{})'.format(mode, ratio_width, ratio_height))

        res = []
        for points in points_list:
            rectified_img = self(image_data, points, interpolation, ratio_width, ratio_height,
                                 loss_thresh=loss_thresh, mode=mode)
            res.append(rectified_img)

        # visualize
        visualized_image = self.visualize(image_data, points_list)

        return res, visualized_image



if __name__ == '__main__':
    # test for a poly text
    print('begin')

    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='Assign the image path')
    parser.add_argument('--txt', type=str, help='Assign the path of .txt to get points',
                        default=None)
    parser.add_argument('--mode', type=str, help='Assign the mode: calibration or homography',
                        default='calibration')
    args = parser.parse_args()

    # test_points = [417,171,415,171,410,170,409,170,400,169,364,170,361,171,358,171,353,171,347,173,342,173,336,175,336,175,331,176,323,177,323,177,320,178,315,178,313,179,310,179,309,179,305,182,302,184,297,184,295,185,291,186,290,186,287,186,284,184,282,183,281,180,280,177,279,174,279,170,279,167,280,165,281,162,284,160,286,158,287,158,290,156,291,156,291,156,294,154,296,153,300,153,301,154,301,153,304,152,306,151,313,149,316,149,318,148,322,147,323,147,324,147,328,147,334,147,340,146,341,146,346,145,357,142,358,142,368,141,370,141,381,142,381,142,385,141,430,142,431,142,446,143,464,143,477,146,479,147,484,147,487,147,489,148,495,148,499,149,501,151,502,152,504,154,505,156,505,161,505,166,504,168,504,169,504,171,504,174,502,177,501,178,500,179,498,181,495,182,490,182,487,181,485,180,472,178,469,177,469,177,451,174,439,173,437,173,431,171,420,171]
    test_points = [162 , 99 ,163, 100 ,166 ,102, 214 , 95, 174, 102, 186, 100, 169, 103, 209,  96 ,192 , 99 ,170 ,103, 204 , 97 ,188 ,100, 190, 100 ,185 ,101 ,207 , 97, 230,  94 ,228,  95 ,242 , 94 ,244 , 95 ,246 , 94 ,254 , 94 ,256,  95 ,257 , 95, 266,  96, 265,  95, 269,  96, 272, 96 ,278,  98 ,279 , 98, 293 ,100, 291,  99, 318, 107, 300, 101,330, 110, 327, 109, 321, 107, 321, 107, 331, 110, 308, 103, 298, 100, 315, 105, 302 ,101 ,333 ,110, 316 ,105, 306, 102,336, 109, 340, 104, 341, 101, 343 , 94 ,343 , 93 ,343 , 91, 343, 89, 342 , 87, 341 , 84, 339,  83, 336,  81, 334,  81, 333 , 80 ,330 , 79, 324,  78, 320 , 77 ,317,  76,316 , 76 ,312 , 75, 311,  75, 299 , 74 ,299,  73, 287 , 71, 282  ,70 ,279,  70, 273,  70, 270 , 69, 267  ,69 ,264 , 69 ,260,  69 ,260 , 69 ,253 , 68, 252 , 67, 242,  67, 239,  67, 229 , 67 ,228,  67,224  ,68, 224,  68, 219,  68 ,217 , 68 ,208  ,70, 208 , 70 ,203,  70 ,201 , 70, 186 , 71 ,185  ,72 ,166,  76 ,164,  76, 162  ,78, 160  ,80, 158 , 82, 157 , 86, 158 , 89, 158,  91 ,159,  93 ,160 , 97]
    # test_points = [330,79,333,80,334,  81 ,336 , 81, 339 , 83, 341 , 84 ,342 , 87 ,343 , 89, 343 , 91,343  ,93 ,343  ,94, 341, 101 ,340 ,104 ,336 ,109, 333, 110, 331, 110, 330, 110, 327 ,109,321, 107 ,321, 107, 318 ,107, 316, 105, 315, 105 ,308, 103, 306, 102, 302, 101 ,300 ,101,298 ,100, 293, 100, 291,  99 ,279 , 98 ,278,  98 ,272 , 96, 269,  96, 266,  96, 265 , 95,257,  95, 256,  95, 254,  94, 246,  94 ,244,  95, 242 , 94, 230,  94 ,228 , 95, 214,  95,209 , 96, 207,  97, 204 , 97, 192,  99, 190, 100 ,188, 100, 186, 100, 185, 101, 174, 102,170, 103, 169, 103 ,166, 102, 163 ,100, 162 , 99 ,160 , 97, 159,  93, 158,  91, 158 , 89,157,  86 ,158,  82, 160 , 80, 162  ,78, 164 , 76, 166,  76, 185,  72, 186 , 71 ,201,  70,203 , 70, 208 , 70, 208, 70 ,217 , 68, 219,  68 ,224 , 68, 224,  68 ,228,  67, 229,  67,239,  67 ,242,  67, 252,  67, 253,  68 ,260,  69, 260 , 69, 264,  69 ,267 , 69 ,270,  69,273 , 70 ,279 , 70, 282, 70 ,287,  71, 299,  74, 299,  73, 311,  75, 312,  75, 316 , 76,317,  76 ,320 , 77 ,324,  78]
    # test_points = [314 ,226, 313, 226, 312, 226, 312 ,226 ,310 ,227, 308 ,228 ,308 ,228 ,306 ,228 ,305 ,229,304 ,229 ,304, 229 ,303 ,229, 301 ,230 ,300 ,231 ,299 ,231 ,298, 231, 297, 231, 296 ,232,295, 232 ,295 ,232 ,294 ,232 ,293, 232 ,292, 232, 291 ,233 ,290 ,233, 289, 233, 288 ,233,287 ,234 ,286 ,234, 285, 234, 284 ,234 ,283, 235, 281 ,235, 280, 235, 279, 235, 278 ,236,276, 236 ,275, 236, 273, 236 ,272, 237 ,270 ,237 ,270, 237, 267, 237 ,266 ,237 ,264, 237,263, 237 ,262, 237, 261 ,235, 260 ,234, 259 ,232, 259 ,231 ,260 ,229 ,260 ,228 ,261 ,227,263 ,226 ,263 ,226, 264 ,226 ,265 ,226 ,266 ,225, 269 ,225 ,270 ,225 ,272 ,225 ,273 ,224,276 ,224 ,277, 224, 277, 224, 278, 223, 280, 223 ,281 ,223 ,282 ,223 ,283 ,223 ,284, 223,285 ,222, 286 ,222 ,287, 222 ,288 ,222 ,289 ,221, 290, 221, 291 ,221 ,293 ,221 ,293 ,220,294, 220 ,295, 219, 296 ,219 ,297 ,219, 298, 219 ,299 ,219 ,300 ,219, 300 ,218 ,302 ,218,302 ,218 ,304, 217 ,304, 217 ,305, 216 ,306 ,216, 307 ,216, 308, 215, 309, 215, 309 ,215,310 ,215, 312 ,214 ,312 ,214 ,313 ,214 ,314, 214 ,315, 213, 317, 213 ,319 ,214 ,320 ,215,320 ,217 ,320, 221, 320 ,223 ,319 ,224 ,317 ,224 ,315 ,225]
    image_path = 'demo1.jpg'

    if not os.path.exists(image_path):
        raise FileNotFoundError

    with open(image_path, "rb") as f:
        image = np.array(bytearray(f.read()), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)

    if image is None:
        raise ValueError('the image of {} is None'.format(image_path))

    autoRectifier = AutoRectifier()
    res = autoRectifier(image, test_points, interpolation=cv2.INTER_LINEAR,
                        ratio_width=1.0, ratio_height=1.0,
                        loss_thresh=5.0, mode=args.mode)

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    cv2.imwrite(os.path.join(save_path, '111OneBoxImageResult.jpg'), res)
    print('{} is saved.'.format(os.path.join(save_path, '111OneBoxImageResult.jpg')))
    print('done!')