import os
import time
from datetime import datetime
import cv2
import numpy as np
from PIL import Image
from orbbec_Utils import frame_to_bgr_image
from pyorbbecsdk import *
import torch
import torch.nn.functional as F
import transforms as transforms
from models import VGG

OBConfig = Config

class_names = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
cut_size = 44

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

transform_test = transforms.Compose([
    transforms.TenCrop(cut_size),
    transforms.Lambda(lambda crops: torch.stack([transforms.ToTensor()(crop) for crop in crops])),
])

_model_public = None
_model_private = None
_inference_count = 0
_total_inference_time = 0.0


def _load_model(checkpoint_name):
    model = VGG('VGG19')
    checkpoint_path = os.path.join('FER2013_VGG19', checkpoint_name)
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f'Model checkpoint not found: {checkpoint_path}')
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['net'])
    model.to(device)
    model.eval()
    return model


def _preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    resized = cv2.resize(gray, (48, 48), interpolation=cv2.INTER_LINEAR)
    img = resized[:, :, None]
    img = np.concatenate((img, img, img), axis=2)
    pil_img = Image.fromarray(img)
    return transform_test(pil_img)


def fer_by_vgg19(frame):
    """Run inference on both models and return structured stats.

    Returns a dict with keys:
      - results: {model_name: {'pred': int, 'score': float, 'scores': np.array}}
      - times: {model_name: float_seconds}
      - avg_fps: running average fps
    """
    global _model_public, _model_private, _inference_count, _total_inference_time

    if _model_public is None:
        _model_public = _load_model('PublicTest_model.pth')
    if _model_private is None:
        _model_private = _load_model('PrivateTest_model.pth')

    inputs = _preprocess_frame(frame)
    ncrops, c, h, w = inputs.shape
    inputs = inputs.view(-1, c, h, w).to(device)

    results = {}
    model_times = {}
    start_total = time.perf_counter()

    with torch.no_grad():
        for model_name, model in [('Public', _model_public), ('Private', _model_private)]:
            t0 = time.perf_counter()
            outputs = model(inputs)
            t1 = time.perf_counter()
            outputs_avg = outputs.view(ncrops, -1).mean(0)
            score = F.softmax(outputs_avg, dim=0)
            scores_np = score.cpu().numpy()
            pred = int(score.argmax().item())
            results[model_name] = {'pred': pred, 'score': float(score[pred].item()), 'scores': scores_np}
            model_times[model_name] = t1 - t0

    elapsed_total = time.perf_counter() - start_total
    _inference_count += 1
    _total_inference_time += elapsed_total
    avg_fps = _inference_count / _total_inference_time if _total_inference_time > 0 else 0.0

    return {
        'results': results,
        'times': model_times,
        'avg_fps': avg_fps,
    }


def main():
    ctx = Context()
    device_list = ctx.query_devices()

    if device_list.get_count() == 0:
        print("Error: No Orbbec device found")
        raise SystemExit(1)

    print(f"Found {device_list.get_count()} Orbbec device(s). Using the first one.")
    orbb_device = device_list.get_device_by_index(0)
    pipeline = Pipeline(orbb_device)
    profiles = pipeline.get_stream_profile_list(OBSensorType.COLOR_SENSOR)
    p = profiles.get_default_video_stream_profile()
    print(
            f"{p.get_width()}x{p.get_height()} " f"@ {p.get_fps()} fps  format={p.get_format()}"
        )
    pipeline.start()

    window_name = "FER Live Demo"

    # recording state
    recording = False
    writer = None
    video_fps = 60.0
    out_dir = os.path.join(os.path.dirname(__file__), 'records')
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)

    while True:
        frames = pipeline.wait_for_frames(1000)
        if frames is None:
            continue

        color_frame = frames.get_color_frame()
        if color_frame is None:
            continue

        color_image = frame_to_bgr_image(color_frame)
        if color_image is None:
            continue

        info = fer_by_vgg19(color_image)
        # draw overlay on the frame
        try:
            pub = info['results']['Public']
            pri = info['results']['Private']
            t_pub = info['times']['Public'] * 1000.0
            t_pri = info['times']['Private'] * 1000.0
            fps = info['avg_fps']
            header1 = f"Public: {class_names[pub['pred']]} {pub['score']:.2f} time={t_pub:.1f}ms"
            header2 = f"Private: {class_names[pri['pred']]} {pri['score']:.2f} time={t_pri:.1f}ms"
            header3 = f"FPS(avg): {fps:.1f}"
            cv2.putText(color_image, header1, (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, header2, (10, 45), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            cv2.putText(color_image, header3, (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

            # Draw confidence bars for both models on right side
            h_img, w_img = color_image.shape[:2]
            bar_w = 180
            bar_h = 12
            gap = 6
            base_x = w_img - bar_w - 10
            base_y = 20
            # Public model bars
            cv2.putText(color_image, 'Public scores', (base_x, base_y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            for i, cname in enumerate(class_names):
                sc = float(info['results']['Public']['scores'][i])
                bx = base_x
                by = base_y + i * (bar_h + gap)
                fill_w = int(sc * bar_w)
                cv2.rectangle(color_image, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
                cv2.rectangle(color_image, (bx, by), (bx + fill_w, by + bar_h), (0, 180, 0), -1)
                cv2.putText(color_image, f"{cname}:{sc:.2f}", (bx - 120, by + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)

            # Private model bars (below public)
            offset = len(class_names) * (bar_h + gap) + 10
            cv2.putText(color_image, 'Private scores', (base_x, base_y + offset - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)
            for i, cname in enumerate(class_names):
                sc = float(info['results']['Private']['scores'][i])
                bx = base_x
                by = base_y + offset + i * (bar_h + gap)
                fill_w = int(sc * bar_w)
                cv2.rectangle(color_image, (bx, by), (bx + bar_w, by + bar_h), (50, 50, 50), -1)
                cv2.rectangle(color_image, (bx, by), (bx + fill_w, by + bar_h), (200, 100, 0), -1)
                cv2.putText(color_image, f"{cname}:{sc:.2f}", (bx - 120, by + bar_h - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (220, 220, 220), 1)
        except Exception:
            cv2.putText(color_image, str(info), (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

        cv2.imshow(window_name, color_image)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            # toggle recording
            if not recording:
                h_img, w_img = color_image.shape[:2]
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                out_path = os.path.join(out_dir, f'record_{timestamp}.mp4')
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(out_path, fourcc, video_fps, (w_img, h_img))
                if not writer.isOpened():
                    writer = None
                    print('Failed to open video writer')
                else:
                    recording = True
                    print(f'Started recording to {out_path} at {video_fps} FPS')
            else:
                # stop recording
                recording = False
                if writer is not None:
                    writer.release()
                    print('Stopped recording')
                    writer = None
        if recording and writer is not None:
            # ensure frame is BGR and correct size
            writer.write(color_image)

        if key == ord('q'):
            if recording and writer is not None:
                writer.release()
                writer = None
                recording = False
            break

    cv2.destroyAllWindows()
    pipeline.stop()


if __name__ == "__main__":
    main()
