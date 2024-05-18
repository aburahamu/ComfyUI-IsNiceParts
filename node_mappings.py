try:
    from .dwpose import *
except ImportError:
    print(f"node dwpose is import failed.")

from ultralytics import YOLO
import numpy as np
import torch
import cv2
from PIL import Image, ImageOps, ImageSequence

import os
import datetime
import statistics
import folder_paths

OUTPUTS_DIRECTORY = 'IsNiceParts'
MODELS_DIRECTORY = 'models'
MODEL_NAME_YOLO = 'hand_yolov8n.pt'

# create ditector.
pose = DWposeDetector()

# create output directory.
outputs_dir = os.path.join(folder_paths.get_output_directory(), OUTPUTS_DIRECTORY)
if not os.path.exists(outputs_dir):
    os.makedirs(outputs_dir)

# set YOLO model.
current_path = os.path.dirname(os.path.realpath(__file__))
models_path = os.path.join(current_path, MODELS_DIRECTORY)
yolo_file_path = os.path.join(models_path, MODEL_NAME_YOLO)
yolo_model = YOLO(yolo_file_path)

# JudgeHand ***********************************************************************************
class NiceHand:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": ("IMAGE", {}),
                "filename_prefix": ("STRING", {"default": "IsNiceParts"}),
                "confidence": ("FLOAT", {"default": 0.9, "min":0, "max": 1, "step":0.01}),
                },
            }

    FUNCTION = "run"
    OUTPUT_NODE = True
    RETURN_TYPES = ( "BOOLEAN","IMAGE",)
    RETURN_NAMES = ( "bool","image",)

    CATEGORY = "IsNiceParts"

    # return nice image.
    def run(self, image, confidence, filename_prefix="IsNiceParts"):
        # create output list.
        result_images = list()
        hand_dict = dict(
            origin_width = 0,
            origin_height = 0,
            hand_regions = None,
            bone_regions = None,
            confidence = confidence
        )
        
        # get file info.
        subfolder = os.path.dirname(os.path.normpath(filename_prefix))

        # save original image.
        save_info = self.save_images(subfolder = subfolder, prefix = filename_prefix, images = image, image_type = "origin_image")
        origin_img_path = save_info["save_path"]
        result_images.append({"filename": save_info["file_name"], "subfolder": subfolder, "type": "output",})
        
        # load original image.
        output_image, output_mask = self.load_image(origin_img_path)
        hand_dict["origin_width"] = output_image.shape[2]
        hand_dict["origin_height"] = output_image.shape[1]

        # detect hand_images and saving.
        if len(image) > 0:
            hand_images, hand_xywhns = self.getHandImages_v2(model = yolo_model, image_path = origin_img_path)
            hand_regions = self.getHandRegions(xywhns = hand_xywhns)
            hand_dict["hand_regions"] = hand_regions
            save_info = self.save_images(subfolder = subfolder, prefix = filename_prefix, images = hand_images, image_type = "hand_images")
            result_images.append({"filename": save_info["file_name"], "subfolder": subfolder, "type": "output",})

        # detect bone_image and saving.
        if len(hand_images) > 0:
            bone_image, bone_regions = self.detectBone_v4(image_path = origin_img_path)
            hand_dict["bone_regions"] = bone_regions
            save_info = self.save_images(subfolder = subfolder, prefix = filename_prefix, images = bone_image, image_type = "bone_image")
            result_images.append({"filename": save_info["file_name"], "subfolder": subfolder, "type": "output",})
        
        # judge isNice.
        if hand_dict["origin_width"] > 0 and hand_dict["origin_height"] > 0:
            isNice, judge_scores = self.judgeNice(hand_dict)
            for i in range(len(judge_scores)):
                print(f"score[{i}] = {judge_scores[i]}")
        else:
            isNice = False

        print(f"isNice = {isNice}. hand = {len(hand_images)}, bone = {len(bone_regions)}.")
        return {"ui": {"images": result_images}, "result": (isNice,output_image, output_mask, )}

    # return isNice(True or False).
    def judgeNice(self, hand_dict):
        # read dict.
        W = hand_dict["origin_width"]
        H = hand_dict["origin_height"]
        conf = hand_dict["confidence"]

        # degree of deviation.
        dd_dict = []
        for i in range(len(hand_dict["hand_regions"])):
            isNice = True
            h_region = hand_dict["hand_regions"][i]
            dd_candidate = []
            for j in range(len(hand_dict["bone_regions"])):
                b_region = hand_dict["bone_regions"][j]
                # calc param.
                dd_center_x = abs(h_region["x_center"] - b_region["x_center"]) / W
                dd_center_y = abs(h_region["y_center"] - b_region["y_center"]) / H
                dd_width = abs(h_region["width"] - b_region["width"]) / W
                dd_height = abs(h_region["height"] - b_region["height"]) / H
                dd_average = 1 - statistics.mean([dd_center_x, dd_center_y, dd_width, dd_height])
                
                # create dict_temp.
                dd_temp = dict(
                    dd_average = int(dd_average * 1000) / 1000,
                    dd_center_x = int(dd_center_x * 1000) / 1000,
                    dd_center_y = int(dd_center_y * 1000) / 1000,
                    dd_width = int(dd_width * 1000) / 1000,
                    dd_height = int(dd_height * 1000) / 1000
                    )
                dd_candidate.append(dd_temp)
            # add dd_dict
            try:
                # Take the best score.
                dd_dict.append(dd_candidate[0])
                for k in range(len(dd_candidate)):
                    if dd_dict[i]["dd_average"] < dd_candidate[k]["dd_average"]:
                        dd_dict[i] = dd_candidate[k]
            except Exception as e:
                isNice = False        
        # return isNice.
        for dd in dd_dict:
            dd_score = dd["dd_average"]
            if dd_score < conf:
                isNice = False
        
        return isNice, dd_dict

    # return hand regions.
    def getHandRegions(self, xywhns):
        hand_regions = []
        for xywhn in xywhns:
            region = torch.round(xywhn).int()
            center_x = region[0].item()
            center_y = region[1].item()
            width = region[2].item()
            height = region[3].item()
            
            # append parameter.
            hand_regions.append(dict(
                x_min = center_x - int(width * 0.5),
                x_max = center_x + int(width * 0.5),
                x_center = int(center_x),
                y_min = center_y - int(height * 0.5),
                y_max = center_y + int(height * 0.5),
                y_center = int(center_y),
                width = width,
                height = height
            ))
        return hand_regions

    # return cropped image.
    def crop_image(self, image_path, region):
        # read image.
        img = cv2.imread(f"{image_path}")
        h, w = img.shape[:2]

        # get region.
        x_center, y_center, width, height = region
        x_center *= w
        y_center *= h
        width *= w
        height *= h

        # proc region.
        top_left = (int(x_center - width / 2), int(y_center - height / 2))
        bottom_right = (int(x_center + width / 2), int(y_center + height / 2))

        # cropping hand.
        cropped_img = img[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]

        return cropped_img

    # return hand images with yolo.
    def getHandImages_v2(self, model, image_path):
        image = cv2.imread(f"{image_path}")
        image_tensor = torch.clamp(torch.from_numpy(image).float() / 255., 0, 1)
        image_tensor = image_tensor[..., [2, 1, 0]]

        # Run inference on image.
        image_bchw = image
        results_yolo = model.predict(image_bchw, imgsz=640, conf=0.5, max_det=4)

        # cropping hands.
        hand_images = []
        hand_regions = []
        for r in results_yolo:
            for xywhn in r.boxes.xywhn:
                # Add HandImage.
                crop_img = self.crop_image(image_path = image_path, region = xywhn)
                # crop_imgはRGBなのでそのまま渡す
                hand_images.append(crop_img)
                hand_regions.append(xywhn)

        return hand_images, hand_regions
    
    # detectBone_v4.
    def detectBone_v4(self, image_path):
        # read image.
        orig_image = cv2.imread(image_path)
        # 骨格推定
        bone_image, poses = pose(orig_image)

        # handsのX座標とY座標を取得
        hands = poses['hands']
        bone_regions = []
        for hand in hands:
            # 大きさを取得
            H, W, C = orig_image.shape

            # 手の情報を取得
            x_coords = hand[:, 0]
            y_coords = hand[:, 1]

            # 見えない部位（-1）を除外
            visible_x_coords = x_coords[x_coords != -1]
            visible_y_coords = y_coords[y_coords != -1]

            # 領域のパラメータを初期化
            min_x = max_x = width = 0
            min_y =  max_y = height = 0

            # visible_x_coordsとvisible_y_coordsが空でないことを確認
            if visible_x_coords.size > 0 and visible_y_coords.size > 0:
                # X座標の最大と最小、Y座標の最大と最小を計算
                min_x, max_x = visible_x_coords.min(), visible_x_coords.max()
                min_y, max_y = visible_y_coords.min(), visible_y_coords.max()
                width = max_x - min_x
                height = max_y - min_y
                center_x = (min_x + max_x) * 0.5
                center_y = (min_y + max_y) * 0.5

                bone_regions.append(dict(
                    x_min = int(min_x * W), 
                    x_max = int(max_x * W),
                    x_center = int(center_x * W),
                    y_min = int(min_y * H), 
                    y_max = int(max_y * H), 
                    y_center = int(center_y * H),
                    width = int(width * W),
                    height = int(height * H)
                    ))

        return bone_image, bone_regions

    # 画像を読み込んで返す関数(LoadImageをコピペした)
    def load_image(self, image_path):
        img = Image.open(image_path)
        output_images = []
        output_masks = []
        for i in ImageSequence.Iterator(img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            if 'A' in i.getbands():
                mask = np.array(i.getchannel('A')).astype(np.float32) / 255.0
                mask = 1. - torch.from_numpy(mask)
            else:
                mask = torch.zeros((64,64), dtype=torch.float32, device="cpu")
            output_images.append(image)
            output_masks.append(mask.unsqueeze(0))

        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
            output_mask = torch.cat(output_masks, dim=0)
        else:
            output_image = output_images[0]
            output_mask = output_masks[0]
            
        return (output_image, output_mask)

    # 受け取った画像を保存し辞書を返す関数
    def save_images(self, subfolder, prefix, images, image_type):
        # get save info.
        output_dir = folder_paths.get_output_directory()
        full_output_folder = os.path.join(output_dir, subfolder)
        formatted_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

        # save original image.
        if image_type == "origin_image":
            file_name = f"{prefix}_{formatted_now}.png"
            save_path = os.path.join(full_output_folder, file_name)
            for (batch_number, image) in enumerate(images):
                # i = 255. * image.cpu().numpy()
                image = image * 255
                image = image.numpy().astype(np.uint8)
                pil_img = Image.fromarray(image)
                # pil_img = Image.fromarray(np.clip(image.numpy(), 0, 255).astype(np.uint8))
                pil_img.save(save_path, **{'compress_level': 4}, pnginfo=None)

        # save hand image.
        elif image_type == "hand_images":
            for i in range(len(images)):
                file_name = f"{prefix}_{formatted_now}_hand{i}.png"
                save_path = os.path.join(full_output_folder, file_name)
                img = cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB)
                pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
                pil_img.save(save_path, **{'compress_level': 4}, pnginfo=None)

        # save bone image.
        elif image_type == "bone_image":
            file_name = f"{prefix}_{formatted_now}_bone.png"
            save_path = os.path.join(full_output_folder, file_name)
            img = images
            pil_img = Image.fromarray(np.clip(img, 0, 255).astype(np.uint8))
            pil_img.save(save_path, **{'compress_level': 4}, pnginfo=None)
        else:
            pass
        
        # create result from image.
        result = dict(save_path = save_path, file_name = file_name)
        return result
    
# Mappings ***********************************************************************************
NODE_CLASS_MAPPINGS = {
    "NiceHand": NiceHand,
    }

NODE_DISPLAY_NAME_MAPPINGS = {
    "NiceHand": "NiceHand",
    }