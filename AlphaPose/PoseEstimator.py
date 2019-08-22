import numpy as np
from torch import nn
import torch
import cv2
import skimage
from yolo.darknet import Darknet
from yolo.util import dynamic_write_results
from AlphaPose.dataloader import Mscoco, DataWriter, crop_from_dets
from SPPE.src.main_fast_inference import InferenNet, InferenNet_fast
from SPPE.src.utils.eval import getPrediction, getMultiPeakPrediction
from yolo.preprocess import prep_image, prep_image_array, prep_frame, inp_to_image, letterbox_image
from SPPE.src.utils.img import load_image, cropBox, im_to_torch
from AlphaPose.pPose_nms import pose_nms, write_json

class PoseEstimator(nn.Module):
    def __init__(self):
        super(PoseEstimator, self).__init__()
        # Det model
        self.det_model = Darknet("models/yolo/yolov3-spp.cfg").cuda()
        self.det_model.load_weights('models/yolo/yolov3-spp.weights')
        self.inp_dim = 608 
        self.det_model.net_info['height'] = self.inp_dim
        self.det_inp_dim = int(self.det_model.net_info['height'])
        self.det_model.eval()
        # Pose model
        self.pose_dataset = Mscoco()
        self.pose_model = InferenNet_fast(4 * 1 + 1, self.pose_dataset)
        self.pose_model.cuda()
        self.pose_model.eval()
        
    def forward(self, image_list):
        images = image_list
        im_dim_list = torch.tensor(
                [[im.shape[1], im.shape[0],
                    im.shape[1], im.shape[0]] for im in image_list]).float()
        img_batch = torch.cat([prep_image_array(image, 608)[0] for image in images]).cuda()

        #From dataloader
        with torch.no_grad():
            prediction = self.det_model(img_batch, CUDA=True)
            dets = dynamic_write_results(prediction, confidence=0.05,
                                num_classes=80, nms=True, nms_conf=0.6)
            if isinstance(dets, int) or dets.shape[0] == 0:
                #TODO: nothing detected.
                return [None] * len(image_list)
                # print(orig_img, im_name, None, None, None, None, None)
            dets = dets.cpu()
            im_dim_list = torch.index_select(im_dim_list,0, dets[:, 0].long())
            scaling_factor = torch.min(self.det_inp_dim / im_dim_list, 1)[0].view(-1, 1)

            # coordinate transfer
            dets[:, [1, 3]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 0].view(-1, 1)) / 2
            dets[:, [2, 4]] -= (self.det_inp_dim - scaling_factor * im_dim_list[:, 1].view(-1, 1)) / 2
            
            dets[:, 1:5] /= scaling_factor
            for j in range(dets.shape[0]):
                dets[j, [1, 3]] = torch.clamp(dets[j, [1, 3]], 0.0, im_dim_list[j, 0])
                dets[j, [2, 4]] = torch.clamp(dets[j, [2, 4]], 0.0, im_dim_list[j, 1])
            boxes = dets[:, 1:5]
            scores = dets[:, 5:6]

        inps_combined = []
        pt1_combined = []
        pt2_combined = []
        for k in range(len(image_list)):
            boxes_k = boxes[dets[:,0]==k]
            if isinstance(boxes_k, int) or boxes_k.shape[0] == 0:
                #nothing detected.
                continue
            inputResH = 320
            inputResW = 256
            inps = torch.zeros(boxes_k.size(0), 3, inputResH, inputResW)
            pt1 = torch.zeros(boxes_k.size(0), 2)
            pt2 = torch.zeros(boxes_k.size(0), 2)

            inp = im_to_torch(cv2.cvtColor(images[k], cv2.COLOR_BGR2RGB))
            inps, pt1, pt2 = crop_from_dets(inp, boxes_k, inps, pt1, pt2)
            inps_combined.append(inps)
            pt1_combined.append(pt1)
            pt2_combined.append(pt2)

        inps = torch.cat(inps_combined)
        pt1 = torch.cat(pt1_combined)
        pt2 = torch.cat(pt2_combined)

        with torch.no_grad():
            # TODO: General batch size
            batchSize = 1
            datalen = inps.size(0)
            leftover = 0
            if (datalen) % batchSize:
                leftover = 1
            num_batches = datalen // batchSize + leftover
            hm = []
            # print("num_batches: {}".format(num_batches))
            for j in range(num_batches):
                inps_j = inps[j*batchSize:min((j +  1)*batchSize, datalen)].cuda()
                hm_j = self.pose_model(inps_j)
                hm.append(hm_j)
            hm = torch.cat(hm)
            hm = hm.cpu()

        outputResH=80
        outputResW=64
        preds_hm, preds_img, preds_scores = getPrediction(
            hm, pt1, pt2, inputResH, inputResW, outputResH, outputResW)

        result = []
        for i in range(len(image_list)):
            image_pos = dets[:,0].numpy()
            indices = np.where(image_pos==i)
            if len(indices[0])==0:
                result.append(None)
                continue
            image_result = pose_nms(boxes[indices], scores[indices],
                preds_img[indices], preds_scores[indices])
            if len(image_result):
                result.append(image_result[0])
            else:
                result.append(None)
        return result 
