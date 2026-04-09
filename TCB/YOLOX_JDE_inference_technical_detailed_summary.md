# TCBTrack / YOLOX-JDE Inference Detailed Technical Guide

This document explains the YOLOX-JDE inference pipeline in `TCBTrack` in detail from start to finish. Each function, data structure, and control flow will be examined step by step.

---

## SECTION 1: Inference Entry Point

### 1.1 `TCB/tools/track2.py` - Main Entry Point

```bash
python tools/track2.py -f exps/example/mot/mot17.py -c models/mot17.pth.tar -b 1 -d 1 --fuse
```

Within this command:

```python
# track2.py -> main()
def main(exp, args, num_gpu):
    # ... setup code ...
    
    # 1. Model loading
    model = exp.get_model2()  # <- YOLOX2 + YOLOXHead2 is created
    model.cuda(rank)
    model.eval()  # <- Test mode
    
    # 2. Checkpoint loading
    ckpt = torch.load(ckpt_file, map_location=loc)
    if "head.reid_classifier.weight" in ckpt["model"]:
        ckpt["model"].pop("head.reid_classifier.weight")  # <- ReID classifier used during training is removed
    if "head.reid_classifier.bias" in ckpt["model"]:
        ckpt["model"].pop("head.reid_classifier.bias")
    model.load_state_dict(ckpt["model"], strict=False)
    
    # 3. Start tracking
    *_, summary = evaluator.evaluate_TCB(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
```

**Important points:**
- `get_model2()` → Model using `YOLOXHead2` (ReID embedding-based)
- `model.eval()` → Batch norms are frozen, dropout is disabled
- ReID classifier (`nID` class) used during training is removed (only embedding is needed)

---

### 1.2 `TCB/yolox/evaluators/mot_evaluator_mot17.py` - evaluate_TCB()

```python
def evaluate_TCB(self, model, distributed=False, half=False, ...):
    model = model.eval()
    if half:
        model = model.half()
    
    tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    
    tracker = BYTETracker(self.args)  # Initialize tracker
    
    # Loop over dataloader (for each frame/batch)
    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
        with torch.no_grad():  # <- Gradient calculation disabled (inference mode)
            frame_id = info_imgs[2].item()
            video_id = info_imgs[3].item()
            
            # If frame number is 1, create new tracker
            if frame_id == 1:
                tracker = BYTETracker(self.args, params)
            
            # Convert image to appropriate tensor type
            imgs = imgs.type(tensor_type)
            
            # ===== MODEL FORWARD PASS =====
            outputs = model(imgs)  # <- YOLOX2.forward() → (Backbone + Head)
            
            # ===== POSTPROCESS =====
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
            # ===== TRACKER UPDATE =====
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                # then write results to file...
```

**Control flow:**
1. Image is loaded: `imgs` → [batch, 3, H, W]
2. Model is run: `outputs = model(imgs)`
3. Postprocess: threshold + NMS applied
4. Tracker is updated: new detections and old tracks are matched

---

## SECTION 2: Model Forward Pass

### 2.1 `YOLOX2.forward()` - Backbone + Head Combination

```python
# yolox2.py
class YOLOX2(nn.Module):
    def forward(self, x, targets=None, last_reid=None):
        if not self.training:  # <- Inference mode
            # x shape: [batch, 3, H, W] e.g. [1, 3, 800, 1440]
            
            # Step 1: Backbone (Feature Pyramid Network)
            fpn_outs = self.backbone(x)
            # fpn_outs: 3-level feature map list
            # [
            #   [1, 256, 100, 180],  stride=8
            #   [1, 512, 50, 90],    stride=16
            #   [1, 1024, 25, 45]    stride=32
            # ]
            
            # Step 2: Head (Detection + ReID Embedding)
            outputs = self.head(xin=fpn_outs, imgs=x, last_reid=last_reid)
            # outputs: shape [batch, 23625, 134]
            # 23625 = 100*180 + 50*90 + 25*45 (total anchor count across all FPN levels)
            # 134 = 4 (bbox) + 1 (obj_conf) + 80 (class_probs) + 512 (embedding)
            
            return outputs
        else:  # Training mode
            # Handled differently during training
            # (not important for inference)
            pass
```

**What is Backbone?**
- `YOLOPAFPN` → CSPNet + PANet (Feature Pyramid Network)
- Produces multi-scale features: stride 8, 16, 32
- Different resolutions per level: 100x180, 50x90, 25x45

---

### 2.2 `YOLOXHead2.forward()` - Two-Headed Inference

```python
# yolo_head2.py
class YOLOXHead2(nn.Module):
    def __init__(self, num_classes, ...):
        self.emb_dim = 512  # <- Embedding dimension
        self.num_classes = 80  # e.g. for COCO
        
        # Parallel branches for each FPN level
        self.cls_convs = nn.ModuleList()   # Class prediction convs
        self.reg_convs = nn.ModuleList()   # Bbox regression convs
        self.reid_convs = nn.ModuleList()  # ReID embedding convs
        
        # ...
    
    def forward(self, xin, labels=None, imgs=None, last_reid=None, feature_id=None):
        # xin: three FPN levels [[1, 256, 100, 180], [1, 512, 50, 90], [1, 1024, 25, 45]]
        
        outputs = []
        
        # For each FPN level
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            # x: e.g. [1, 256, 100, 180]
            
            # Stem: channel reduction
            x = self.stems[k](x)  # [1, 256, 100, 180]
            
            # === DETECTION BRANCH ===
            cls_x = x
            reg_x = x
            reid_x = x
            
            # Classification head
            cls_feat = cls_conv(cls_x)              # 2x Conv
            cls_output = self.cls_preds[k](cls_feat) # 1x Conv
            # cls_output shape: [1, 80, 100, 180]
            
            # Bounding box regression
            reg_feat = reg_conv(reg_x)              # 2x Conv
            reg_output = self.reg_preds[k](reg_feat) # 1x Conv
            # reg_output shape: [1, 4, 100, 180]
            
            obj_output = self.obj_preds[k](reg_feat) # 1x Conv
            # obj_output shape: [1, 1, 100, 180]
            
            # === REID EMBEDDING BRANCH ===
            reid_feat = self.reid_convs[k](reid_x)    # 2x Conv
            reid_output = self.reid_preds[k](reid_feat) # 1x Conv
            # reid_output shape: [1, 512, 100, 180]
            
            # === CONCATENATE ===
            output = torch.cat([
                reg_output,           # 4 channels
                obj_output.sigmoid(), # 1 channel
                cls_output.sigmoid(), # 80 channels
                reid_output           # 512 channels
            ], 1)
            # output shape: [1, 134, 100, 180]
            
            outputs.append(output)
        
        # outputs now for 3 levels: [
        #   [1, 134, 100, 180],
        #   [1, 134, 50, 90],
        #   [1, 134, 25, 45]
        # ]
        
        if not self.training:  # <- Inference mode
            # decode_outputs() flattens all heads and adds grid coordinates
            self.hw = [x.shape[-2:] for x in outputs]
            yolo_outputs = self.decode_outputs(outputs, dtype=xin[0].type())
            
            # final output: combine all levels
            return torch.cat([
                x.view(1, -1, 1+4+self.num_classes+self.emb_dim) 
                for x in yolo_outputs
            ], dim=1)
            # return shape: [1, 23625, 134]
```

**Animation (conceptual):**

```
Input: [1, 3, 800, 1440]
           ↓
        BACKBONE (YOLOPAFPN)
           ↓
    3 FPN levels:
    ├─ [1, 256, 100, 180]  ← stride 8
    ├─ [1, 512, 50, 90]    ← stride 16
    └─ [1, 1024, 25, 45]   ← stride 32
           ↓
    HEAD (YOLOXHead2) - Parallel branches per level ↓
    ├─ Detection: [bbox(4) + obj(1) + cls(80)]
    ├─ Embedding: [reID(512)]
    └─ Concat: 134 channels
           ↓
    After concat and decode:
    [1, 100*180 + 50*90 + 25*45, 134]
    [1, 23625, 134]
```

---

### 2.3 `decode_outputs()` - Output Decoding

```python
def decode_outputs(self, outputs, dtype):
    # outputs: 3 levels [
    #   [1, 134, 100, 180],
    #   [1, 134, 50, 90],
    #   [1, 134, 25, 45]
    # ]
    
    out = []
    
    for (hsize, wsize), stride, output in zip(self.hw, self.strides, outputs):
        # stride: 8, 16, or 32
        # hsize, wsize: image level dimensions
        
        # Step 1: Flatten spatial dimensions
        output = output.flatten(start_dim=2).permute(0,2,1)
        # [1, 134, 100, 180] → [1, 100*180, 134] → [1, 18000, 134]
        
        # Step 2: Create grid
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1,-1,2).type(dtype)
        # grid shape: [1, 18000, 2] containing (x_grid, y_grid) for each anchor
        # e.g.: grid[0, 0] = [0, 0], grid[0, 1] = [1, 0], etc.
        
        # Step 3: Decode coordinates
        output[...,:2] = ((output[...,:2] + grid) * stride).type(dtype)
        # raw x,y offsets → grid-relative → stride-scaled
        # x_decoded = (x_offset + x_grid) * stride
        # this way coordinates are cleverly transformed to original image space
        
        output[...,2:4] = (torch.exp(output[...,2:4]) * stride).type(dtype)
        # w,h: log-scale → exponential scale → stride-scaled
        
        # Step 4: Reshape back
        output = output.view(-1, hsize, wsize, 4+1+self.num_classes+self.emb_dim)
        # [1, 18000, 134] → [1, 100, 180, 134]
        
        out.append(output)
    
    return out
    # return: [
    #   [1, 100, 180, 134],
    #   [1, 50, 90, 134],
    #   [1, 25, 45, 134]
    # ]
```

**Coordinate Transformation Explanation:**

```
For each FPN level:
- Raw model output: [-0.5, 0.3, ...] (center offset relative to grid cell)
- Grid: [[0,0], [1,0], [2,0], ..., [179, 99]] (all grid cells)
- Stride: 8 (original 800x1440 → 100x180)

For grid cell (50, 40):
- x_decoded = (-0.5 + 50) * 8 = 396 pixels (in original image)
- y_decoded = (0.3 + 40) * 8 = 325.4 pixels

Thus, model output is directly converted to original image coordinates.
```

---

## SECTION 3: Postprocess (NMS and Threshold)

### 3.1 `postprocess()` Function - Filter Detections

```python
# boxes.py
def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45):
    # prediction: [batch, 23625, 134]
    #   [:, :, 0:4] = bbox (x1, y1, x2, y2)
    #   [:, :, 4] = obj_confidence
    #   [:, :, 5:85] = class probabilities (num_classes = 80)
    #   [:, :, 85:] = embedding (512 dim)
    
    # Step 1: Convert center coordinates to corner coordinates
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2  # x1 = cx - w/2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2  # y1 = cy - h/2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2  # x2 = cx + w/2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2  # y2 = cy + h/2
    prediction[:, :, :4] = box_corner[:, :, :4]
    # Now prediction[:, :, :4] is in [x1, y1, x2, y2] format
    
    output = [None for _ in range(len(prediction))]  # Results per batch
    
    # Step 2: Process each image in batch
    for i, image_pred in enumerate(prediction):  # image_pred: [23625, 134]
        
        if not image_pred.size(0):
            continue
        
        # Get class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5+num_classes],  # all class probabilities
            1,                                # max along class dimension
            keepdim=True
        )
        # class_conf: [23625, 1] - highest logit confidence
        # class_pred: [23625, 1] - class index (0-79)
        
        # Step 3: Confidence filtering
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # obj_confidence * class_confidence >= threshold
        # So: obj_score × class_score ≥ 0.7 (default)
        # FALSE: low confidence detections → filtered out
        
        # Step 4: Concatenate detection info
        detections = torch.cat((
            image_pred[:, :5],                    # [x1, y1, x2, y2, obj_conf]
            class_conf,                           # class_confidence
            class_pred.float(),                   # class_index
            image_pred[:, 1+4+num_classes:]      # embedding vectors (512 dim)
        ), 1)
        # detections shape: [23625, 7+512] = [23625, 519]
        #   [:, 0:4] = bbox
        #   [:, 4] = obj_conf
        #   [:, 5] = class_conf
        #   [:, 6] = class_id
        #   [:, 7:519] = embedding
        
        # Step 5: Apply confidence mask
        detections = detections[conf_mask]  # [N, 519] where N < 23625
        # Now only detections with confidence > 0.7 remain
        
        if not detections.size(0):
            continue  # If no detections remain, go to next image
        
        # Step 6: NMS (Non-Maximum Suppression)
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],              # bboxes
            detections[:, 4] * detections[:, 5],  # confidence scores (obj × class)
            detections[:, 6],               # class indices
            nms_thre                        # IoU threshold (0.45)
        )
        # nms_out_index: indices to keep
        # NMS: Removes low confidence boxes with high IoU in same class
        
        detections = detections[nms_out_index]  # [M, 519] where M <= N
        
        # Step 7: Store output
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    
    # output: [tensor1, None, tensor2, ...]
    # each tensor shape: [num_dets_after_nms, 519]
    #   num_dets_after_nms: typically 50-500
    return output
```

**Postprocess Pipeline:**

```
Input: [1, 23625, 134]
   ↓
Confidence filtering (threshold=0.7):
   obj_score × class_score ≥ 0.7
   ↓ (typically 500-2000 detections remain)
   ↓
NMS (IoU threshold=0.45):
   Filter overlapping boxes in same class
   ↓ (typically 50-300 detections remain)
   ↓
Output: [num_dets, 519]
   [:, :4] = bbox [x1, y1, x2, y2]
   [:, 4] = obj_confidence
   [:, 5] = class_confidence
   [:, 6] = class_id
   [:, 7:519] = embedding (512 dim)
```

---

## SECTION 4: Tracker Initialization and Initial Setup

### 4.1 BYTETracker.__init__()

```python
# byte_tracker.py (my_byte_tracker_mot17_kal.py)
from yolox.tracker.my_byte_tracker_mot17_kal import BYTETracker

class BYTETracker(object):
    def __init__(self, args, params=[-100, 0.9, 100, -100, -100]):
        self.tracked_stracks = []   # Currently actively tracked objects
        self.lost_stracks = []      # Lost objects (wait for several frames)
        self.removed_stracks = []   # Permanently removed objects
        
        self.frame_id = 0           # Current frame number
        self.args = args            # Command-line arguments
        self.det_thresh = args.track_thresh + 0.1  # Threshold for new tracks
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        # buffer_size: how long to wait for lost tracks
        # Typical: 30 frames
        
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()  # Initialize Kalman filter
        
        self.params = params  # [match_thresh_stage1, iou_thresh_gate, ...]
```

**Data Structures:**

```python
# STrack objects represent tracks
class STrack:
    def __init__(self, tlwh, score, temp_feat, buffer_size=60):
        self._tlwh = tlwh  # Top-left-width-height
        self.score = score  # Detection confidence
        
        self.curr_feat = None  # Current embedding feature [512,]
        self.update_features(temp_feat)  # Set initial feature
        self.alpha = 0.1  # EMA update coefficient
        
        self.kalman_filter = None  # Will be set when activated
        self.mean, self.covariance = None, None  # Kalman state
        self.is_activated = False  # Have we confirmed this track yet?
        self.track_id = -1  # Track ID (negative = not assigned yet)
        self.frame_id = None  # Frame where this track was first seen
        self.state = TrackState.NEW  # Can be: NEW, TRACKED, LOST, REMOVED
```

**Track State Management:**

- **NEW**: Recently detected, not yet confirmed as stable track
- **TRACKED**: Actively being tracked with recent detections
- **LOST**: No matching detection in recent frames, but still recoverable
- **REMOVED**: Permanently removed from tracking (too long lost or low confidence)

**Kalman Filter State:**
- **mean**: [x, y, w, h, vx, vy, vw, vh] - position and velocity estimates
- **covariance**: Uncertainty matrix for state estimation
- **8-dimensional state**: 4 position + 4 velocity components

---

## SECTION 5: Tracker.update() - Main Tracking Loop

### 5.1 tracker.update() Structure

```python
def update(self, output_results, img_info, img_size, id_feature=None):
    """
    Args:
        output_results: [num_dets, 519]
           [:, :4] = bbox, [:, 4] = obj_conf, [:, 5] = cls_conf,
           [:, 6] = cls_id, [:, 7:] = embedding
        
        img_info: [H, W, 1, 1, path_string]
        img_size: (input_H, input_W) e.g. (800, 1440)
    
    Returns:
        output_stracks: List of STrack objects (only activated ones)
    """
    
    self.frame_id += 1
    
    # ===== STEP 0: Parse detections =====
    if output_results.shape[1] == 5:  # Old format (bbox + conf only)
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
    else:  # New format (bbox + conf + class + embedding)
        output_results = output_results.cpu().numpy()
        scores = output_results[:, 4] * output_results[:, 5]  # obj_conf × class_conf
        bboxes = output_results[:, :4]  # [x1, y1, x2, y2]
        id_feature = torch.tensor(output_results[:, 7:])  # [512,] embedding
    
    img_h, img_w = img_info[0], img_info[1]
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    bboxes /= scale  # Convert from model input size to original image size
    
    # ===== STEP 1: Split detections by confidence level =====
    remain_inds = scores > self.args.track_thresh  # High confidence (> 0.6 typical)
    inds_low = scores > 0.1
    inds_high = scores < self.args.track_thresh
    
    inds_second = np.logical_and(inds_low, inds_high)  # 0.1 < score < 0.6
    
    dets_second = bboxes[inds_second]  # Low score detections for second matching
    dets = bboxes[remain_inds]  # High score detections for first matching
    scores_keep = scores[remain_inds]
    scores_second = scores[inds_second]
    id_feature_keep = id_feature[remain_inds]
    id_feature_second = id_feature[inds_second]
    
    # ===== STEP 2: Convert detections to STrack objects =====
    if len(dets) > 0:
        detections = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 60)
            for (tlbr, s, f) in zip(dets, scores_keep, id_feature_keep)
        ]
    else:
        detections = []
    
    # ===== Relevant parts continue... =====
```

---

### 5.2 Organizing the Tracking List

```python
    # ===== STEP 3: Categorize existing tracks =====
    unconfirmed = []
    tracked_stracks = []
    
    for track in self.tracked_stracks:
        if not track.is_activated:
            unconfirmed.append(track)  # New track, not yet confirmed
        else:
            tracked_stracks.append(track)  # Confirmed track
    
    # ===== STEP 4: Create track pool =====
    strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    # strack_pool: lost tracks + active tracks
    # Tracks in this pool are objects we saw in frame N-1
    
    # ===== STEP 5: Kalman Filter Prediction =====
    STrack.multi_predict(strack_pool)
    # Predict each track's position one frame forward (Kalman momentum)
    # Makes it easier to match with current frame detections later
```

**Track States & Pools Animation:**

```
Global Track List (self.tracked_stracks):
├─ Track#1 (is_activated=True)    → goes to tracked_stracks
├─ Track#2 (is_activated=True)    → goes to tracked_stracks
├─ Track#3 (is_activated=False)   → goes to unconfirmed
└─ Track#4 (is_activated=True)    → goes to tracked_stracks

Lost List (self.lost_stracks):
├─ Track#5 (not observed for 3 frames)
├─ Track#6 (not observed for 1 frame)
└─ ...

strack_pool = tracked_stracks + lost_stracks
├─ Track#1, Track#2, Track#4 (from tracked)
├─ Track#5, Track#6 (from lost)
└─ (Track#3 excluded because it went to unconfirmed)

This pool represents candidate tracks to match with detections.
```

---

### 5.3 First Matching Stage (High Score Detections)

```python
    # ===== STEP 6: First matching - high confidence detections =====
    
    if len(strack_pool) > 0 and len(detections) > 0:
        # === Sub-step 6a: Special embedding-based masking ===
        motion = match2(strack_pool, id_feature_keep, pd=self.params[0])
        # match2: Thresholded version of cosine similarity matrix
        # motion: [len(strack_pool), len(detections)] binary matrix
        # motion[i, j] = True → track i and detection j have very different embeddings
        
        s1 = match3(strack_pool, id_feature_keep, pd=self.params[0])
        # s1: The normalized cosine similarity matrix itself
        # s1 ∈ [0, 1], higher = more similar embedding
        
        # === Sub-step 6b: IoU-based cost matrix ===
        dists = matching.iou_distance(strack_pool, detections)
        # dists: [len(strack_pool), len(detections)]
        # dists[i, j] = 1 - IoU(track_i, detection_j)
        # dists ∈ [0, 1], lower = better match (IoU-wise)
        
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
            # Integrate detection confidence into IoU distance
            # Higher confidence detections are preferred
        
        # === Sub-step 6c: Integrate embedding info into dists ===
        dists = 1 - (1 - dists) * np.array(s1)
        # Explanation:
        #   Original dists value: 1 - IoU
        #   s1 value: cosine_similarity ∈ [0, 1]
        #   Formula: dists_new = 1 - (1 - dists) * s1
        #   
        # If s1 = 0 (embeddings very different):
        #   dists_new = 1 - (1 - dists) * 0 = 1 (matching impossible)
        # If s1 = 1 (embeddings identical):
        #   dists_new = 1 - (1 - dists) * 1 = dists (unchanged)
        
        # Also apply motion matrix: very different embeddings → cost = 1
        if dists.shape == (1, 1) and motion.shape == (1, 1):
            if motion[0, 0]:
                dists[0, 0] = 1
        else:
            dists[motion] = 1  # If motion[i,j] = True then dists[i,j] = 1
        
        # === Sub-step 6d: Matching with Hungarian Algorithm ===
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.params[1]  # params[1] = 0.9 typical
        )
        # matches: [(track_idx, det_idx), ...] matched pairs
        # u_track: track indices that couldn't be matched
        # u_detection: detection indices that couldn't be matched
    
    else:
        # Fallback if no tracks or detections
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.params[1])
    
    # === Sub-step 6e: Update matched tracks ===
    for itracked, idet in matches:
        track = strack_pool[itracked]
        det = detections[idet]
        
        if track.state == TrackState.Tracked:  # Active track
            track.update(det, self.frame_id, det.score > 0.7)
            # update: Update Kalman filter, update embedding with EMA
            activated_starcks.append(track)
        
        else:  # Lost track (old)
            track.re_activate(det, self.frame_id, new_id=False)
            # re_activate: Restart Kalman filter, reactivate track
            refind_stracks.append(track)
```

**Matching Algorithm Details:**

```
Input:
- strack_pool: N old objects
- detections: M new detections
- dists: [N, M] matrix

Hungarian Algorithm:
  Find N×M matching that minimizes dists
  
Output:
- matches ⊆ N×M (optimal and below threshold)
- u_track: unmatched strack indices
- u_detection: unmatched detection indices
```

---

### 5.4 Second Matching Stage (Low Score Detections)

```python
    # ===== STEP 7: Second matching - low confidence detections =====
    
    if len(dets_second) > 0:
        detections_second = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 30)
            for (tlbr, s, f) in zip(dets_second, scores_second, id_feature_second)
        ]
    else:
        detections_second = []
    
    # Tracked tracks that couldn't be matched in first matching
    r_tracked_stracks = [
        strack_pool[i] for i in u_track
        if strack_pool[i].state == TrackState.Tracked
    ]
    # u_detection: High-score detections unmatched in 1st matching
    detections = [detections[i] for i in u_detection]
    
    # Match low-score detections with tracked tracks
    if len(r_tracked_stracks) > 0 and len(detections_second) > 0:
        motion = match2(r_tracked_stracks, id_feature_second, pd=self.params[4])
        dists = matching.iou_distance(r_tracked_stracks, detections_second)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections_second)
        dists[motion] = 1
        
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5  # Lower threshold for low-score detections
        )
    else:
        matches, u_track, u_detection_second = matching.linear_assignment(
            dists, thresh=0.5
        )
    
    # Update matched tracks
    for itracked, idet in matches:
        track = r_tracked_stracks[itracked]
        det = detections_second[idet]
        
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id, False)  # Safe update
            activated_starcks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    
    # Move unmatched tracks to lost list
    for it in u_track:
        track = r_tracked_stracks[it]
        if not track.state == TrackState.Lost:
            track.mark_lost()
            lost_stracks.append(track)
```

---

### 5.5 Process Unconfirmed Tracks

```python
    # ===== STEP 8: Process unconfirmed tracks =====
    # Unconfirmed: new tracks not yet matched once
    
    # Unmatched detections remaining from 1st and 2nd matching
    detections = [detections[i] for i in u_detection]
    
    # Match with unconfirmed tracks
    dists = matching.iou_distance(unconfirmed, detections)
    if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
    
    matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    
    for itracked, idet in matches:
        unconfirmed[itracked].update(detections[idet], self.frame_id, detections[idet].score > 0.7)
        activated_starcks.append(unconfirmed[itracked])  # Now becomes active
    
    # Remove unconfirmed tracks that couldn't be matched for two frames
    for it in u_unconfirmed:
        track = unconfirmed[it]
        track.mark_removed()
        removed_stracks.append(track)
```

---

### 5.6 Initialize New Tracks

```python
    # ===== STEP 9: Start new tracks for new objects =====
    
    for inew in u_detection:  # Detections unmatched in all matchings
        track = detections[inew]
        
        if track.score < self.det_thresh:  # det_thresh = 0.6 + 0.1 = 0.7 typical
            continue  # Too low confidence → don't start new track
        
        track.activate(self.kalman_filter, self.frame_id)
        # activate: assign track_id, initialize Kalman state, is_activated=True (no change yet)
        # In 1st frame is_activated = True (MOT challenge definition)
        # In subsequent frames is_activated = False and becomes True after matching
        activated_starcks.append(track)
```

---

### 5.7 State Management

```python
    # ===== STEP 10: Track state management =====
    
    # Remove tracks lost for too long
    for track in self.lost_stracks:
        if self.frame_id - track.end_frame > self.max_time_lost:  # 30 frames
            track.mark_removed()
            removed_stracks.append(track)
    
    # ===== STEP 11: Update global track list =====
    self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
    self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
    
    self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
    
    self.removed_stracks.extend(removed_stracks)
    
    # Remove duplicate tracks (very close tracks)
    self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
        self.tracked_stracks, self.lost_stracks
    )
    
    # ===== STEP 12: Return results =====
    output_stracks = [track for track in self.tracked_stracks if track.is_activated]
    
    return output_stracks  # Only active (confirmed) tracks
```

---

## SECTION 6: Feature Update (Embedding EMA)

### 6.1 STrack.update_features() - EMA Logic

```python
class STrack:
    def __init__(self, tlwh, score, temp_feat, buffer_size=60):
        self.curr_feat = None  # Current embedding
        self.update_features(temp_feat)  # Set initial embedding
        self.alpha = 0.1  # EMA coefficient
    
    def update_features(self, feat):
        """
        Update icon embedding with Exponential Moving Average (EMA)
        
        Args:
            feat: new embedding vector, shape [512]
        """
        if self.curr_feat is None:
            # First time: assign directly
            self.curr_feat = feat
        else:
            # Subsequent updates: EMA combination
            self.curr_feat = self.curr_feat * (1 - self.alpha) + feat * self.alpha
            # curr_feat = 0.9 * old_feat + 0.1 * new_feat
            # Weight: past information 90%, new information 10%
```

**EMA Explanation:**

```
Frame 1: feat_1 = [a1, a2, ..., a512]
  curr_feat = feat_1

Frame 2: feat_2 = [b1, b2, ..., b512]
  curr_feat = 0.9 * [a1, ...] + 0.1 * [b1, ...]
  
Frame 3: feat_3 = [c1, c2, ..., c512]
  curr_feat = 0.9 * (0.9*feat_1 + 0.1*feat_2) + 0.1 * feat_3
           = 0.81*feat_1 + 0.09*feat_2 + 0.1*feat_3

Weights (newer times have higher weight):
- Compared to 3 frames ago: 0.81 * 0.1 ≈ 0.081 (very low)
- Compared to 2 frames ago: 0.09 (higher)
- Current (frame 3): 0.1 (highest)

NOTE: Past information exponentially decays
→ Recent observations are weighted more
→ But protected from sudden camera view changes
```

---

## SECTION 7: Matching Algorithm (Hungarian)

### 7.1 `matching.linear_assignment()` - Hungarian Algorithm

```python
# matching.py
def linear_assignment(cost_matrix, thresh):
    """
    Solve cost matrix with Hungarian algorithm for min-cost matching
    
    Args:
        cost_matrix: [N, M] matrix
        thresh: cost threshold (above this → no assign)
    
    Returns:
        matches: [(i, j), ...] optimal matches
        unmatched_a: unmatched row indices
        unmatched_b: unmatched column indices
    """
    
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))
    
    # Step 1: Solve with Hungarian algorithm (LAP - Linear Assignment Problem)
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # cost: total optimal cost
    # x: which column each row is assigned to (-1 = unassigned)
    # y: which row each column is assigned to (-1 = unassigned)
    
    # Step 2: Extract matched pairs
    matches = []
    for ix, mx in enumerate(x):
        if mx >= 0:  # Assigned
            matches.append([ix, mx])
    
    matches = np.asarray(matches)
    
    # Step 3: Determine unmatched rows/columns
    unmatched_a = np.where(x < 0)[0]  # Rows without assignment
    unmatched_b = np.where(y < 0)[0]  # Columns without assignment
    
    return matches, unmatched_a, unmatched_b
```

**Hungarian Algorithm Visualization:**

```
Example Cost Matrix (3 tracks, 4 detections):
        Det0  Det1  Det2  Det3
Track0  0.2   0.8   0.9   0.7
Track1  0.9   0.1   0.7   0.8
Track2  0.7   0.9   0.2   0.6

Hungarian algorithm goal: minimize total cost
Possible results:
1. Track0→Det0(0.2), Track1→Det1(0.1), Track2→Det2(0.2) = 0.5 ✓ OPTIMAL
2. Track0→Det1(0.8), Track1→Det0(0.9), Track2→Det2(0.2) = 1.9
3. Track0→Det3(0.7), Track1→Det2(0.7), Track2→Det0(0.7) = 2.1

Result:
- matches = [(0,0), (1,1), (2,2)]
- unmatched_a = []  (all tracks assigned)
- unmatched_b = [3]  (Det3 not assigned)
```

---

## SECTION 8: Kalman Filter (Prediction)

### 8.1 STrack.multi_predict() - Kalman Prediction

```python
@staticmethod
def multi_predict(stracks):
    """
    Predict Kalman states of all tracks one frame forward
    
    Args:
        stracks: List[STrack]
    """
    if len(stracks) > 0:
        # Extract Kalman states in batch
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        # [N, 7] - for each track: [cx, cy, w, h, vx, vy, ...]
        
        multi_covariance = np.asarray([st.covariance for st in stracks])
        # [N, 7, 7] - 7x7 covariance matrix for each track
        
        # Kalman prediction: x_{t+1} = F * x_t + noise
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        # multi_mean: [N, 7] - predicted state
        # multi_covariance: [N, 7, 7] - predicted uncertainty
        
        # Assign predicted states back to tracks
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov
```

**Kalman State Representation:**

```
State vector: x = [cx, cy, w, h, vx, vy, ...] (7D or 8D)
- cx, cy: center coordinates (pixels)
- w, h: width, height (pixels)
- vx, vy: velocity (pixels/frame) - how fast it's moving
- ... e.g. aspect ratio change rate

Example from previous frame:
x_t = [100, 50, 200, 150, 5, 2]
     = center (100, 50), size (200, 150), velocity (5, 2) px/frame

Prediction:
x_{t+1} = F @ x_t + noise
        ≈ [105, 52, 200, 150, 5, 2]  (±uncertainty)
        = center (105, 52), size (200, 150), velocity (5, 2)

Covariance: How uncertain each state element is
- New tracks: high covariance
- Long-tracked objects: low covariance
```

---

## SECTION 9: Results and Output Writing

```python
# mot_evaluator_mot17.py
def evaluate_TCB(...):
    # ... previous code ...
    
    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
        # ...
        online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
        
        # online_targets: List[STrack] (activated and tracked objects)
        
        # Write results
        online_tlwhs = []
        online_ids = []
        online_scores = []
        
        for t in online_targets:
            tlwh = t.tlwh  # Top-left-width-height format
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6  # Filter very narrow/tall boxes
            
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        
        results.append((frame_id, online_tlwhs, online_ids, online_scores))
        
        # When frame count reaches video end
        if cur_iter == len(self.dataloader) - 1:
            result_filename = os.path.join(result_folder, '{}.txt'.format(video_names[video_id]))
            write_results(result_filename, results)


def write_results(filename, results):
    save_format = '{frame},{id},{x1},{y1},{w},{h},{s},-1,-1,-1\n'
    # MOT16/17 format:
    # frame_id,track_id,x1,y1,width,height,score,-1,-1,-1
    
    with open(filename, 'w') as f:
        for frame_id, tlwhs, track_ids, scores in results:
            for tlwh, track_id, score in zip(tlwhs, track_ids, scores):
                if track_id < 0:
                    continue
                x1, y1, w, h = tlwh
                line = save_format.format(
                    frame=frame_id,
                    id=track_id,
                    x1=round(x1, 1),
                    y1=round(y1, 1),
                    w=round(w, 1),
                    h=round(h, 1),
                    s=round(score, 2)
                )
                f.write(line)
```

**Output Format Example:**

```
1,1,100.0,50.0,200.0,150.0,0.95,-1,-1,-1
1,2,400.0,300.0,180.0,200.0,0.87,-1,-1,-1
2,1,105.0,52.0,200.0,150.0,0.93,-1,-1,-1
2,3,450.0,310.0,190.0,210.0,0.85,-1,-1,-1
...

frame 1:
  track #1: (100, 50) + (200x150), conf=0.95
  track #2: (400, 300) + (180x200), conf=0.87

frame 2:
  track #1: (105, 52) + (200x150), conf=0.93 [moved with Kalman prediction]
  track #3: (450, 310) + (190x210), conf=0.85 [new track]
```

---

## SUMMARY: Complete Inference Loop

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. INPUT: Video Frame (800x1440)                               │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 2. BACKBONE (YOLOPAFPN): Multi-scale Features                  │
│    [256, 100, 180] + [512, 50, 90] + [1024, 25, 45]            │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 3. HEAD (YOLOXHead2): Parallel Branches                         │
│    ├─ Detector: bbox + confidence                              │
│    └─ Embedding: 512-dim feature maps → [23625, 512]           │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 4. DECODE & CONCATENATE: [23625, 134]                          │
│    [bbox(4) | obj_conf(1) | class_probs(80) | embedding(512)]  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 5. POSTPROCESS: Threshold + NMS                                 │
│    Score filtering (> 0.7) → ~500 detections                   │
│    NMS (IoU > 0.45) → ~100-300 detections                       │
│    Output: [num_dets, 519]                                      │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 6. TRACKER.UPDATE():                                            │
│    ├─ Kalman Predict: Predicted positions of tracks            │
│    ├─ Hungarian Matching: IoU + Embedding Cost Matrix           │
│    ├─ First Match: high-score detections                        │
│    ├─ Second Match: low-score detections                        │
│    ├─ New Tracks: unmatched high-score detections               │
│    └─ EMA Feature Update: curr_feat = 0.9*old + 0.1*new        │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│ 7. OUTPUT: Track Results                                        │
│    [frame_id, track_id, x, y, w, h, score] → .txt file         │
└─────────────────────────────────────────────────────────────────┘
```

## 2. Dual-Branch Inference (JDE Forward Pass)

### `TCB/yolox/models/yolo_head2.py`
- `YOLOXHead2` contains both detection and ReID branches.
- For each FPN level (`strides=[8,16,32]`):
  - `reg_output` -> bbox regression
  - `obj_output.sigmoid()` -> object confidence
  - `cls_output.sigmoid()` -> class probabilities
  - `reid_output` -> dense embedding map, dimension `self.emb_dim = 512`
- In test mode `forward()`:
  - `output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid(), reid_output],1)`
  - `decode_outputs()` converts columns to x,y,w,h format
  - `return torch.cat([...], dim=1)` output shape [batch, all_anchors, 4+1+num_classes+512]
- This matches your 5-step flow's "shared feature map f_t" and "embedding head".

## 3. Sparse Sampling / Point Sampling

- `postprocess()` function converts `YOLOXHead2` output to detection list with boxes and scores.
- `tracker.update()` call uses `outputs[0]` and `id_feature`.
- In `my_byte_tracker_mot17_kal.py`:
  - `STrack` object has `curr_feat` field
  - When creating `detections`, each detection gets `f` (embedding) data
  - So embeddings are used as vectors corresponding to selected detection points after `postprocess`, not for the entire image
- This corresponds to your "system only extracts 1x512 feature vector at that (x, y) point" approach.

## 4. Claim about Cost Matrix without Kalman

### Actual situation
- Kalman filter is still active in `TCB/yolox/tracker/my_byte_tracker_mot17_kal.py`:
  - `STrack.multi_predict(strack_pool)` call exists
  - `self.kalman_filter = KalmanFilter()` object is used
  - Kalman update continues in `track.update(...)` and `re_activate(...)`
- Therefore the code is not a "pure Kalman-free" tracking pipeline.

### However special matching exists
- `match2()` and `match3()` functions:
  - `E = torch.cat([track.curr_feat ...])` gets past template vectors
  - `F = id_feature.permute(1,0)` gets new detection embeddings
  - `M = cosine_similarity(E, F)` is calculated
  - `motion` / `s1` masks are used to correct some matches in `dists` matrix
- So in actual code:
  - Base cost is IoU-based with `matching.iou_distance()`
  - Then corrections are made with `fuse_score()` and embedding-based masks
- This is close to your "Cost = Cosine_Similarity * IoU" formula but uses a different combination instead of exact multiplication.

## 5. Hungarian Algorithm

### Method used
- `linear_assignment(cost_matrix, thresh)` in `TCB/yolox/tracker/matching.py`
  - Hungarian solution calculated using `lap.lapjv()`
  - Returns matches, unmatched tracks and unmatched detections
- In `my_byte_tracker_mot17_kal.py` this function is used in multiple stages:
  - First stage: high-score detections
  - Second stage: low-score detections
  - Third stage: unconfirmed tracks
- So it's not global optimum, but still multi-stage matching based on Hungarian algorithm.

## 6. EMA Memory Update

### `STrack.update_features()`
- In `TCB/yolox/tracker/my_byte_tracker_mot17_kal.py`:
  - If `self.curr_feat` is assigned for the first time, saved directly
  - Afterwards `self.curr_feat = self.curr_feat * (1-self.alpha) + feat * self.alpha`
  - `alpha = 0.1` is set
- This is clearly an EMA / low-pass update.
- Therefore the user's definition "matching target's memory is not blindly changed" is correct.

## 7. `trainer.py` vs `trainer2.py`

- `TCB/yolox/core/trainer.py`:
  - Uses `self.exp.get_model(settings=self.settings)` with `YOLOXHead` based model
  - Not for detection + simple ReID/TCL, follows classic YOLOX training logic
- `TCB/yolox/core/trainer2.py`:
  - Uses `self.exp.get_model2(settings=self.settings)` with `YOLOXHead2` based JDE model
  - This is the real training pipeline for JDE / ReID containing pipeline in this repo
- `track2.py` directly uses `exp.get_model2()` so inference is compatible with `trainer2.py`'s model structure.

## 8. Heatmap / `m_t` bypass

- In `YOLOXHead2` training includes `checknet()` and `get_reid()` functions containing TCL / heatmap logic.
- In test mode (`if not self.training`) these codes are not used.
- So during inference `heatmap` / `m_t` calculations are practically bypassed.

## 9. Conclusion

The inference pipeline for this repo works as follows:

1. `track2.py` calls `exp.get_model2()`.
2. `YOLOX2` + `YOLOXHead2` model produces dense detection + dense embedding maps with `model(imgs)`.
3. `postprocess()` takes only selected boxes and their corresponding embedding vectors.
4. The tracker in `my_byte_tracker_mot17_kal.py` uses these vectors with IoU-based initial cost, then corrects with embedding masks.
5. Matching is done with `linear_assignment()`.
6. EMA-like embedding update is applied in `STrack.update_features()`.

> Note: Kalman filter is still active in the code. If the goal is to verify a completely "Kalman-free" pipeline in inference, either remove Kalman usage in `my_byte_tracker_mot17_kal.py` or use a file that uses full ReID+IoU combination instead.
