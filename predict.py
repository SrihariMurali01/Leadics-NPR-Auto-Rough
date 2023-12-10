import re
import hydra
import torch
import cv2
import pytesseract
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box

# Set the path to the pytesseract executable (change this to your actual path)
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class DetectionPredictor(BasePredictor):
    def __init__(self, config=DEFAULT_CONFIG, overrides=None):
        super().__init__(config, overrides)
        self.txt_path = None

    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string

        # Print bounding box coordinates
        log_string += "Box Coordinates: "
        for det_entry in reversed(det):
            if len(det_entry) == 6:
                xyxy, conf, cls = det_entry[:4], det_entry[4], det_entry[5]
            elif len(det_entry) == 7:
                xyxy, conf, cls = det_entry[1:5], det_entry[5], det_entry[6]
            else:
                raise ValueError(f"Unexpected length of det_entry: {len(det_entry)}")

            log_string += f"({int(xyxy[0])},{int(xyxy[1])}), ({int(xyxy[2])},{int(xyxy[3])}) | "

        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "

        # Perform OCR on the number plate region if it exists
        number_plate_detected = False
        for det_entry in reversed(det):
            if len(det_entry) == 6:
                xyxy, conf, cls = det_entry[:4], det_entry[4], det_entry[5]
            elif len(det_entry) == 7:
                xyxy, conf, cls = det_entry[1:5], det_entry[5], det_entry[6]
            else:
                raise ValueError(f"Unexpected length of det_entry: {len(det_entry)}")

            # Assuming class index for number plate is 0, change this if needed
            if int(cls) == 0:
                number_plate_detected = True
                x1, y1, x2, y2 = map(int, xyxy)
                number_plate_roi = im0[y1:y2, x1:x2]

                # Perform OCR on the number plate region
                text = pytesseract.image_to_string(number_plate_roi)
                text = re.sub(r'[^A-Z0-9]', '', text)
                # Print the extracted text
                log_string += f"Detected Number Plate: {text}"
                with open('predict.log', 'a') as p:
                    p.write(f"{log_string}\n")
                p.close()
                print("Detected Number Plate: ", text)

                # Display the image with the detected region
                cv2.imshow('Number Plate', number_plate_roi)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

        if not number_plate_detected:
            print("No number plate detected in the image.")

        return log_string


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    predict()
