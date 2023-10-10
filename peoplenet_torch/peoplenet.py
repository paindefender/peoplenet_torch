import os
import time
from typing import Any
from urllib.request import urlretrieve

import onnx
import torch
import torchvision
from onnx2torch import convert


class PeopleNet:
    def __init__(self, device=None) -> None:
        """Initialize a PeopleNet model

        Args:
            device (str, optional): specify the pytorch device.
        """
        if device is None:
            if torch.cuda.is_available():
                device = "cuda"
            else:
                device = "cpu"
        model_filename = "resnet34_peoplenet_int8.onnx"
        jit_filename = "peoplenet.jit.pt"
        if not os.path.exists(jit_filename):
            if not os.path.exists(model_filename):
                urlretrieve(
                    "https://api.ngc.nvidia.com/v2/models/org/nvidia/team/tao/peoplenet/pruned_quantized_decrypted_v2.3.3/files?redirect=true&path=resnet34_peoplenet_int8.onnx",
                    model_filename,
                )
            onnx_model = onnx.load(model_filename)
            onnx_model = convert(onnx_model).to(device)
            self.model = torch.jit.trace(
                onnx_model.to(device), torch.randn((1, 3, 544, 960)).to("cuda")
            ).eval()
            torch.jit.save(self.model, jit_filename)
        else:
            self.model = torch.jit.load(jit_filename)
        self.device = device
        self.box_norm = 35.0

        self.centers = (
            torch.stack(
                torch.meshgrid(
                    torch.linspace(0, 33, 34), torch.linspace(0, 59, 60), indexing="ij"
                )
            )
            * 16.0
            + 0.5
        ).flip((0)).to(device) / self.box_norm

        self.people_boxes = torch.empty((4, 34, 60)).to(device)

    def __call__(
        self, input, conf=0.5, nms_threshold=0.7, xywh=False, verbose=True
    ) -> list:
        """_summary_

        Args:
            input (matrix): a uint8 960x544 image
            conf (float, optional): minimum confidence for a valid bounding box. Defaults to 0.5.
            nms_threshold (float, optional): threshold for the non-max suppression algorithm. Defaults to 0.7.
            xywh (bool, optional): return boxes in xywh format, by default they're xyxy. Defaults to False.
            verbose (bool, optional): show more detail. Defaults to True.

        Returns:
            list: people's bounding boxes
        """
        prepre = time.time()
        input = self.preprocess(input)
        postpre = time.time()

        with torch.no_grad():
            preinf = time.time()
            confs, boxes = self.model(input)
            postinf = time.time()

            prepost = time.time()
            detections = self.postprocess(confs, boxes, conf, nms_threshold, xywh)
            postpost = time.time()

        if verbose:
            print(f"detected {detections.shape[0]} people")
            print(
                f"pre-processing: {(postpre-prepre)*1000:.2f}ms, inference: {(postinf-preinf)*1000:.2f}ms, post-processing: {(postpost-prepost)*1000:.2f}ms"
            )
        return detections

    def preprocess(self, input):
        input = (
            torch.tensor(input).permute((2, 0, 1)).unsqueeze(0).to(self.device) / 255.0
        )
        return input

    def postprocess(self, confs, boxes, conf, nms_threshold, xywh):
        boxes_ppl = boxes[0][:4]
        confs_ppl = confs[0][0].reshape(-1)
        self.people_boxes[:2] = (boxes_ppl[:2] - self.centers) * (-self.box_norm)
        self.people_boxes[2:] = (boxes_ppl[2:] + self.centers) * self.box_norm
        ppl_box_flat = self.people_boxes.permute((1, 2, 0)).reshape(-1, 4)[
            confs_ppl > conf
        ]
        boxes_nmsed = torchvision.ops.nms(
            ppl_box_flat, confs_ppl[confs_ppl > conf], nms_threshold
        )
        result = ppl_box_flat[boxes_nmsed]
        if xywh:
            xywh_result = torch.empty_like(result)
            xywh_result[:, 0] = (result[:, 0] + result[:, 2]) / 2
            xywh_result[:, 1] = (result[:, 1] + result[:, 3]) / 2
            xywh_result[:, 2] = result[:, 2] - result[:, 0]
            xywh_result[:, 3] = result[:, 3] - result[:, 1]
            return xywh_result.int().cpu().numpy()
        else:
            return result.int().cpu().numpy()
