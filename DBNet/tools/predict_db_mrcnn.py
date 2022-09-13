import os
import cv2
import torch
from torchvision import transforms as T

from maskrcnn_benchmark.modeling.detector import build_detection_model
from maskrcnn_benchmark.utils.checkpoint import DetectronCheckpointer
from maskrcnn_benchmark.structures.image_list import to_image_list
from maskrcnn_benchmark.config import cfg
from maskrcnn_benchmark.utils.chars import getstr_grid, get_tight_rect

from PIL import Image
import numpy as np
import argparse
from shapely.geometry import Point
from shapely.geometry import Polygon
import time

class Pytorch_model:
    def __init__(self, model_path, post_p_thre=0.7, gpu_id=None):
        '''
        初始化pytorch模型
        :param model_path: 模型地址(可以是模型的参数或者参数和计算图一起保存的文件)
        :param gpu_id: 在哪一块gpu上运行
        '''
        self.gpu_id = gpu_id

        if self.gpu_id is not None and isinstance(self.gpu_id, int) and torch.cuda.is_available():
            self.device = torch.device("cuda:%s" % self.gpu_id)
        else:
            self.device = torch.device("cpu")
        print('device:', self.device)
        checkpoint = torch.load(model_path, map_location=self.device)

        config = checkpoint['config']
        config['arch']['backbone']['pretrained'] = False

        self.post_process.box_thresh = post_p_thre
        self.img_mode = config['dataset']['train']['dataset']['args']['img_mode']
        self.model.load_state_dict(checkpoint['state_dict'])
        self.model.to(self.device)
        self.model.eval()

        self.transform = []
        for t in config['dataset']['train']['dataset']['args']['transforms']:
            if t['type'] in ['ToTensor', 'Normalize']:
                self.transform.append(t)

    def predict(self, img_path: str, is_output_polygon=False, short_size: int = 1024):
        '''
        对传入的图像进行预测，支持图像地址,opecv 读取图片，偏慢
        :param img_path: 图像地址
        :param is_numpy:
        :return:
        '''
        assert os.path.exists(img_path), 'file is not exists'
        img = cv2.imread(img_path, 1 if self.img_mode != 'GRAY' else 0)
        if self.img_mode == 'RGB':
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        # 将图片由(w,h)变为(1,img_channel,h,w)
        tensor = self.transform(img)
        tensor = tensor.unsqueeze_(0)

        tensor = tensor.to(self.device)
        batch = {'shape': [(h, w)]}
        with torch.no_grad():
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            start = time.time()
            preds = self.model(tensor)
            if str(self.device).__contains__('cuda'):
                torch.cuda.synchronize(self.device)
            box_list, score_list = self.post_process(batch, preds, is_output_polygon=is_output_polygon)
            box_list, score_list = box_list[0], score_list[0]
            if len(box_list) > 0:
                if is_output_polygon:
                    idx = [x.sum() > 0 for x in box_list]
                    box_list = [box_list[i] for i, v in enumerate(idx) if v]
                    score_list = [score_list[i] for i, v in enumerate(idx) if v]
                else:
                    idx = box_list.reshape(box_list.shape[0], -1).sum(axis=1) > 0  # 去掉全为0的框
                    box_list, score_list = box_list[idx], score_list[idx]
            else:
                box_list, score_list = [], []
            t = time.time() - start
        return preds[0, 0, :, :].detach().cpu().numpy(), box_list, score_list, t


def save_depoly(model, input, save_path):
    traced_script_model = torch.jit.trace(model, input)
    traced_script_model.save(save_path)


def init_args():
    import argparse
    parser = argparse.ArgumentParser(description='DBNet.pytorch')
    parser.add_argument('--model_path', default=r'model_best.pth', type=str)
    parser.add_argument('--input_folder', default='./test/input', type=str, help='img path for predict')
    parser.add_argument('--output_folder', default='./test/output', type=str, help='img path for output')
    parser.add_argument('--thre', default=0.3,type=float, help='the thresh of post_processing')
    parser.add_argument('--polygon', action='store_true', help='output polygon or box')
    parser.add_argument('--show', action='store_true', help='show result')
    parser.add_argument('--save_resut', action='store_true', help='save box and score to txt file')
    args = parser.parse_args()
    return args

class TextDemo(object):
    def __init__(
        self,
        cfg,
        confidence_threshold=0.7,
        min_image_size=224,
        output_polygon=True
    ):
        self.cfg = cfg.clone()
        self.model = build_detection_model(cfg)
        self.model.eval()
        self.device = torch.device(cfg.MODEL.DEVICE)
        self.model.to(self.device)
        self.min_image_size = min_image_size

        checkpointer = DetectronCheckpointer(cfg, self.model)
        _ = checkpointer.load(cfg.MODEL.WEIGHT)

        self.transforms = self.build_transform()
        self.cpu_device = torch.device("cpu")
        self.confidence_threshold = confidence_threshold
        self.output_polygon = output_polygon

    def build_transform(self):
        """
        Creates a basic transformation that was used to train the models
        """
        cfg = self.cfg
        # we are loading images with OpenCV, so we don't need to convert them
        # to BGR, they are already! So all we need to do is to normalize
        # by 255 if we want to convert to BGR255 format, or flip the channels
        # if we want it to be in RGB in [0-1] range.
        if cfg.INPUT.TO_BGR255:
            to_bgr_transform = T.Lambda(lambda x: x * 255)
        else:
            to_bgr_transform = T.Lambda(lambda x: x[[2, 1, 0]])

        normalize_transform = T.Normalize(
            mean=cfg.INPUT.PIXEL_MEAN, std=cfg.INPUT.PIXEL_STD
        )

        transform = T.Compose(
            [
                T.ToPILImage(),
                T.Resize(self.min_image_size),
                T.ToTensor(),
                to_bgr_transform,
                normalize_transform,
            ]
        )
        return transform

    def run_on_opencv_image(self, image):
        """
        Arguments:
            image (np.ndarray): an image as returned by OpenCV
        Returns:
            result_polygons (list): detection results
            result_words (list): recognition results
        """
        result_polygons, result_words = self.compute_prediction(image)

        return result_polygons, result_words

    def compute_prediction(self, original_image):
        # apply pre-processing to image
        image = self.transforms(original_image)
        # convert to an ImageList, padded so that it is divisible by
        # cfg.DATALOADER.SIZE_DIVISIBILITY
        image_list = to_image_list(image, self.cfg.DATALOADER.SIZE_DIVISIBILITY)
        image_list = image_list.to(self.device)
        # compute predictions
        with torch.no_grad():
            predictions = self.model(image_list)
        global_predictions = predictions[0]
        char_predictions = predictions[1]
        char_mask = char_predictions['char_mask']
        char_boxes = char_predictions['boxes']
        words, rec_scores = self.process_char_mask(char_mask, char_boxes)
        seq_words = char_predictions['seq_outputs']
        seq_scores = char_predictions['seq_scores']
        global_predictions = [o.to(self.cpu_device) for o in global_predictions]

        # always single image is passed at a time
        global_prediction = global_predictions[0]

        # reshape prediction (a BoxList) into the original image size
        height, width = original_image.shape[:-1]
        global_prediction = global_prediction.resize((width, height))
        boxes = global_prediction.bbox.tolist()
        scores = global_prediction.get_field("scores").tolist()
        masks = global_prediction.get_field("mask").cpu().numpy()

        result_polygons = []
        result_words = []
        for k, box in enumerate(boxes):
            score = scores[k]
            if score < self.confidence_threshold:
                continue
            box = list(map(int, box))
            mask = masks[k,0,:,:]
            polygon = self.mask2polygon(mask, box, original_image.shape, threshold=0.5, output_polygon=self.output_polygon)
            if polygon is None:
                polygon = [box[0], box[1], box[2], box[1], box[2], box[3], box[0], box[3]]
            result_polygons.append(polygon)
            word = words[k]
            rec_score = rec_scores[k]
            seq_word = seq_words[k]
            seq_char_scores = seq_scores[k]
            seq_score = sum(seq_char_scores) / float(len(seq_char_scores))
            if seq_score > rec_score:
                result_words.append(seq_word)
            else:
                result_words.append(word)
        return result_polygons, result_words

    def process_char_mask(self, char_masks, boxes, threshold=192):
        texts, rec_scores = [], []
        for index in range(char_masks.shape[0]):
            box = list(boxes[index])
            box = list(map(int, box))
            text, rec_score, _, _ = getstr_grid(char_masks[index,:,:,:].copy(), box, threshold=threshold)
            texts.append(text)
            rec_scores.append(rec_score)
        return texts, rec_scores

    def mask2polygon(self, mask, box, im_size, threshold=0.5, output_polygon=True):
        # mask 32*128
        image_width, image_height = im_size[1], im_size[0]
        box_h = box[3] - box[1]
        box_w = box[2] - box[0]
        cls_polys = (mask*255).astype(np.uint8)
        poly_map = np.array(Image.fromarray(cls_polys).resize((box_w, box_h)))
        poly_map = poly_map.astype(np.float32) / 255
        poly_map=cv2.GaussianBlur(poly_map,(3,3),sigmaX=3)
        ret, poly_map = cv2.threshold(poly_map,0.5,1,cv2.THRESH_BINARY)
        if output_polygon:
            SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            poly_map = cv2.erode(poly_map,SE1) 
            poly_map = cv2.dilate(poly_map,SE1);
            poly_map = cv2.morphologyEx(poly_map,cv2.MORPH_CLOSE,SE1)
            try:
                _, contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            except:
                contours, _ = cv2.findContours((poly_map * 255).astype(np.uint8), cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)
            if len(contours)==0:
                print(contours)
                print(len(contours))
                return None
            max_area=0
            max_cnt = contours[0]
            for cnt in contours:
                area=cv2.contourArea(cnt)
                if area > max_area:
                    max_area = area
                    max_cnt = cnt
            perimeter = cv2.arcLength(max_cnt,True)
            epsilon = 0.01*cv2.arcLength(max_cnt,True)
            approx = cv2.approxPolyDP(max_cnt,epsilon,True)
            pts = approx.reshape((-1,2))
            pts[:,0] = pts[:,0] + box[0]
            pts[:,1] = pts[:,1] + box[1]
            polygon = list(pts.reshape((-1,)))
            polygon = list(map(int, polygon))
            if len(polygon)<6:
                return None     
        else:      
            SE1=cv2.getStructuringElement(cv2.MORPH_RECT,(3,3))
            poly_map = cv2.erode(poly_map,SE1) 
            poly_map = cv2.dilate(poly_map,SE1);
            poly_map = cv2.morphologyEx(poly_map,cv2.MORPH_CLOSE,SE1)
            idy,idx=np.where(poly_map == 1)
            xy=np.vstack((idx,idy))
            xy=np.transpose(xy)
            hull = cv2.convexHull(xy, clockwise=True)
            #reverse order of points.
            if  hull is None:
                return None
            hull=hull[::-1]
            #find minimum area bounding box.
            rect = cv2.minAreaRect(hull)
            corners = cv2.boxPoints(rect)
            corners = np.array(corners, dtype="int")
            pts = get_tight_rect(corners, box[0], box[1], image_height, image_width, 1)
            polygon = [x * 1.0 for x in pts]
            polygon = list(map(int, polygon))
        return polygon

    def visualization(self, image, polygons, words):
        for polygon, word in zip(polygons, words):
            pts = np.array(polygon, np.int32)

            pts = pts.reshape((-1,1,2))


            xmin = min(pts[:,0,0])
            ymin = min(pts[:,0,1])
            cv2.polylines(image,[pts],True,(0,0,255))
            cv2.putText(image, word, (xmin, ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,255), 2)
            print(pts.reshape(1, -1))


    def crop(self, image, polygons):

        pixels = np.array(image)
        im_copy = np.array(image)
        for index_num,polygon in enumerate(polygons):
            pixels = np.array(image)
            im_copy = np.array(image)
            pts = np.array(polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))

            for i in pts:
                temp = i[0][1]
                i[0][1] = i[0][0]
                i[0][0] = temp
            # print(pts.shape)
            # print(type(pts))
            # pts = np.array(pts)
            coor = []
            for j in pts:
                coor.append(tuple(j[0]))
            # print(coor)
            region = Polygon(coor)
            for index, pixel in np.ndenumerate(pixels):
              # Unpack the index.
              row, col, channel = index
              # We only need to look at spatial pixel data for one of the four channels.
              if channel != 0:
                continue
              point = Point(row, col)
              if not region.contains(point):
                im_copy[(row, col, 0)] = 255
                im_copy[(row, col, 1)] = 255
                im_copy[(row, col, 2)] = 0
                # im_copy[(row, col, 3)] = 0
            cut_image = Image.fromarray(im_copy)
            cut_image.save(str(index_num)+ '_crop' +'.png')

def main(args):
    # update the config options with the config file
    cfg.merge_from_file(args.config_file)
    # manual override some options
    # cfg.merge_from_list(["MODEL.DEVICE", "cpu"])

    text_demo = TextDemo(
        cfg,
        min_image_size=800,
        confidence_threshold=0.7,
        output_polygon=True
    )
    # load image and then run prediction
    
    image = cv2.imread(args.image_path)
    result_polygons, result_words = text_demo.run_on_opencv_image(image)
    text_demo.visualization(image, result_polygons, result_words)
    cv2.imwrite(args.visu_path, image)
    text_demo.crop(image, result_polygons)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='parameters for demo')
    parser.add_argument("--config-file", type=str, default='../configs/finetune.yaml')
    parser.add_argument("--image_path", type=str, default='../demo_images/demo.jpg')
    parser.add_argument("--visu_path", type=str, default='../demo_images/demo_results.jpg')
    args = parser.parse_args()
    main(args)
