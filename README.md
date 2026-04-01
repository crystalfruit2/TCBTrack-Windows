# TCBTrack-Windows: RTX 50-Series & Windows 11 Compatible Fork

This repository is a heavily modified, Windows-compatible fork of the original multi-object tracking architecture, adapted for modern hardware workflows and experimental setups like edge-optimized drone target tracking.

The original academic codebase was designed for Linux environments and older CUDA architectures (circa 2022). This fork provides the surgical patches and environment configurations necessary to run the TCBTrack cross-correlation head and YOLOX backbone on modern Windows systems utilizing next-generation NVIDIA hardware (specifically tested on the RTX 5060 Ti / `sm_120` Blackwell architecture).

## 📜 Acknowledgements & Original Citations
This repository is built upon the foundational research of the following teams. All credit for the core architectures, mathematical models, and baseline training weights goes to the original authors. 

* **TCBTrack (Original Architecture):** [hustvl/TCBTrack](https://github.com/hustvl/TCBTrack)
* **ByteTrack (Base Tracking Logic):** [ifzhang/ByteTrack](https://github.com/ifzhang/ByteTrack)
* **YOLOX (Object Detection Backbone):** [Megvii-BaseDetection/YOLOX](https://github.com/Megvii-BaseDetection/YOLOX)

If you utilize this codebase for academic or research purposes, please cite the original TCBTrack paper:
> *Zhang, Y., Liang, C., Gao, J., Zhang, Z., Hu, W., Maybank, S., Zhou, X., & Li, L. "Temporal Correlation Meets Embedding: Towards a 2nd Generation of JDE-based Real-Time Multi-Object Tracking." arXiv preprint arXiv:2407.14086 (2024).*

---

## 🛠️ Key Patches & Modifications
* **Dependency Resolution:** Downgraded specific libraries (`protobuf`, `numpy`) to maintain compatibility with older ONNX and TensorBoard integrations.
* **Windows File System Fixes:** Bypassed Linux-specific multiprocessing deadlocks and `.sh` dataset splitting scripts.
* **RTX 50-Series Compatibility (Blackwell):** * Disabled `--fp16` mixed-precision acceleration to prevent Tensor Core deadlocks on unsupported hardware architectures.
  * **Surgical NMS Patch:** Intercepted the `torchvision::nms` Non-Maximum Suppression filter (which lacks compiled C++ operators for the newest CUDA backends) and rerouted it to the CPU, preventing a hard crash at the end of every inference step.

---

## 1. Environment Setup

It is highly recommended to isolate this installation using Miniconda.

```bash
# Initialize the environment
conda create -n tcb_env python=3.8
conda activate tcb_env

# Install PyTorch (Adjust CUDA version to match your system if necessary)
pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cu118](https://download.pytorch.org/whl/cu118)

# Install dependencies and the tracker
pip install -r requirements.txt
python setup.py develop

# Fix 2022 dependency deprecations
pip install "protobuf<4.0.0"
pip install "numpy<1.24"
```

## 2. Dataset Preparation (MOT17)

TCBTrack requires the [MOT17 Dataset](https://motchallenge.net/). The raw dataset provides ground-truth annotations in `.txt` format, but the YOLOX backbone strictly requires COCO-formatted `.json` dictionaries.

1. Download the MOT17 dataset and place it in `TCB/datasets/mot/`.
2. Ensure you have the target directory created: `TCB/datasets/mot/annotations/`.
3. Convert the annotations:
```bash
python tools/convert_mot17_to_coco.py
```
*Note for Windows users: If you manually partitioned your validation data, the converter may bundle everything into a single `train.json` file. To satisfy the dataloader for validation, simply copy `train.json` and rename the copy to `val_half.json`.*

## 3. Model Weights

Due to regional download restrictions on the original Baidu Drive links, this setup utilizes the compatible ByteTrack "donor brain".

1. Download `bytetrack_x_mot17.pth.tar` from the [ByteTrack GitHub Releases (v1.0.0)](https://github.com/ifzhang/ByteTrack/releases/tag/v1.0.0).
2. Create a `models` folder: `TCB/models/`.
3. Place the downloaded file inside and rename it exactly to `mot17.pth.tar`.
*(Windows Warning: Ensure "File name extensions" is enabled in View settings to prevent the file from secretly being named `mot17.pth.tar.tar`).*

## 4. The Surgical NMS Patch (For RTX 40/50-Series)

If you are running an NVIDIA GPU architecture newer than Ampere, the legacy `torchvision` NMS operator will throw a `NotImplementedError` on the CUDA backend.

To patch this, open `TCB/yolox/evaluators/mot_evaluator_dancetrack.py` and navigate to **line 300**.
Modify the tensor to route through the CPU before post-processing:

**Change this:**
```python
outputs = postprocess(outputs,self.num_classes,self.confthre,self.nmsthre)
```
**To this:**
```python
outputs = postprocess(outputs.cpu(),self.num_classes,self.confthre,self.nmsthre)
```

## 5. Execution

To run the tracker on the MOT17 validation split, use the following command. 
*(Note: Do not append the `--fp16` flag if utilizing Blackwell architecture, as it will cause a silent multiprocessing deadlock).*

```bash
python tools/track2.py -f exps/example/mot/mot17.py -c models/mot17.pth.tar -b 1 -d 1 --fuse
```

### Expected Output
The tracker will output pure inference data. The results will be saved as `.txt` files inside:
`TCB/YOLOX_outputs/mot17/track_results/`

Each text file contains the frame-by-frame tracked bounding box coordinates for the target objects.