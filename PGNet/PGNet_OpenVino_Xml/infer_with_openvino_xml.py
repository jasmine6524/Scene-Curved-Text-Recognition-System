from email.mime import audio
import gradio as gr
import cv2

import os
import numpy as np
import time
import sys

import tools.infer.utility as utility
# from ppocr.utils.logging import get_logger
# from ppocr.utils.utility import get_image_file_list, check_and_read_gif
from ppocr.data import create_operators, transform
from ppocr.postprocess import build_post_process

from w2v import *

class TextE2E(object):
    def __init__(self, args):
        self.args = args
        # self.e2e_algorithm = "PGNet"
        self.use_onnx = args.use_onnx
        # OpenVINO Support here
        self.use_openvino = args.use_openvino
        pre_process_list = [{
            'E2EResizeForTest': {}
        }, {
            'NormalizeImage': {
                'std': [0.229, 0.224, 0.225],
                'mean': [0.485, 0.456, 0.406],
                'scale': '1./255.',
                'order': 'hwc'
            }
        }, {
            'ToCHWImage': None
        }, {
            'KeepKeys': {
                'keep_keys': ['image', 'shape']
            }
        }]
        postprocess_params = {}
        
        pre_process_list[0] = {
            'E2EResizeForTest': {
                'max_side_len': args.e2e_limit_side_len,
                'valid_set': 'totaltext'
            }
        }
        postprocess_params['name'] = 'PGPostProcess'
        postprocess_params["score_thresh"] = args.e2e_pgnet_score_thresh
        postprocess_params["character_dict_path"] = args.e2e_char_dict_path
        postprocess_params["valid_set"] = args.e2e_pgnet_valid_set
        postprocess_params["mode"] = args.e2e_pgnet_mode
        

        self.preprocess_op = create_operators(pre_process_list)
        self.postprocess_op = build_post_process(postprocess_params)
        self.predictor, self.input_tensor, self.output_tensors, _ = utility.create_predictor(
            args, 'e2e')  # paddle.jit.load(args.det_model_dir)
        # self.predictor.eval()

    def clip_det_res(self, points, img_height, img_width):
        for pno in range(points.shape[0]):
            points[pno, 0] = int(min(max(points[pno, 0], 0), img_width - 1))
            points[pno, 1] = int(min(max(points[pno, 1], 0), img_height - 1))
        return points

    def filter_tag_det_res_only_clip(self, dt_boxes, image_shape):
        img_height, img_width = image_shape[0:2]
        dt_boxes_new = []
        for box in dt_boxes:
            box = self.clip_det_res(box, img_height, img_width)
            dt_boxes_new.append(box)
        dt_boxes = np.array(dt_boxes_new)
        return dt_boxes

    def __call__(self, img):

        ori_im = img.copy()
        data = {'image': img}
        data = transform(data, self.preprocess_op)
        img, shape_list = data
        if img is None:
            return None, 0
        img = np.expand_dims(img, axis=0)
        shape_list = np.expand_dims(shape_list, axis=0)
        img = img.copy()
        starttime = time.time()

        if self.use_onnx:
            input_dict = {}
            input_dict[self.input_tensor.name] = img
            outputs = self.predictor.run(self.output_tensors, input_dict)
            preds = {}
            preds['f_border'] = outputs[0]
            preds['f_char'] = outputs[1]
            preds['f_direction'] = outputs[2]
            preds['f_score'] = outputs[3]
        # OpenVINO Support Here
        elif self.use_openvino:
            outputs = self.predictor([img])
            out_layers = self.predictor.output
            preds = {}
            preds['f_border'] = outputs[out_layers(0)]
            preds['f_char'] = outputs[out_layers(1)]
            preds['f_direction'] = outputs[out_layers(2)]
            preds['f_score'] = outputs[out_layers(3)]
        else:
            self.input_tensor.copy_from_cpu(img)
            self.predictor.run()
            outputs = []
            for output_tensor in self.output_tensors:
                output = output_tensor.copy_to_cpu()
                outputs.append(output)
            preds = {}
            preds['f_border'] = outputs[0]
            preds['f_char'] = outputs[1]
            preds['f_direction'] = outputs[2]
            preds['f_score'] = outputs[3]

        post_result = self.postprocess_op(preds, shape_list)
        points, strs = post_result['points'], post_result['texts']
        dt_boxes = self.filter_tag_det_res_only_clip(points, ori_im.shape)
        elapse = time.time() - starttime
        return dt_boxes, strs, elapse

def test_net(img):
    args = utility.parse_args()
    text_detector = TextE2E(args)
    draw_img_save = "./results/res.jpg"
    if img is None:
        print("error in loading image")
    points, strs, elapse = text_detector(img)
    image = img.copy()
    src_im = utility.draw_e2e_res(points, strs, image)
    cv2.imwrite(draw_img_save, src_im)
    # return {"img":src_im, "str":strs}
    # print(strs)
    words = ''
    with open('./results/res.txt','w', encoding='utf-8') as f:
        for s in strs:
            f.write(s+'\n')
            words += s+' '
    # str2Talk(words)
    # audio_path = './results/res.mp3'
    # return src_im, words, audio_path
    return src_im, words, elapse


if __name__ == '__main__':
    # interface = gr.Interface(fn=test_net, inputs="image", outputs=['image', 'text', 'audio'])
    # interface.launch()
    import os
    import cv2 as cv
    imgs = os.listdir('./imgs')
    time_ = 0
    for img in imgs:
        img_path = os.path.join('./imgs', img)
        image = cv.imread(img_path)
        _, w, t = test_net(image)
        print(img, t)
        time_ += t
    print('Avg Time:{:.4f}'.format(time_/len(imgs)))
    pass