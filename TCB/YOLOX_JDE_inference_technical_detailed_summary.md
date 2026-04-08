# TCBTrack / YOLOX-JDE Inference Detaylı Teknik Kılavuzu

Bu doküman, `TCBTrack` içindeki YOLOX-JDE inference hattını başından sonuna kadar detaylı şekilde açıklar. Her fonksiyon, veri yapısı ve kontrol akışı adım-adım incelenecektir.

---

## BÖLÜM 1: İnference Girişi (Entry Point)

### 1.1 `TCB/tools/track2.py` - Ana Giriş Noktası

```bash
python tools/track2.py -f exps/example/mot/mot17.py -c models/mot17.pth.tar -b 1 -d 1 --fuse
```

Bu komutun içinde:

```python
# track2.py -> main()
def main(exp, args, num_gpu):
    # ... kurulum kodu ...
    
    # 1. Model yükleme
    model = exp.get_model2()  # <- YOLOX2 + YOLOXHead2 oluşturulur
    model.cuda(rank)
    model.eval()  # <- Test modu
    
    # 2. Checkpoint yükleme
    ckpt = torch.load(ckpt_file, map_location=loc)
    if "head.reid_classifier.weight" in ckpt["model"]:
        ckpt["model"].pop("head.reid_classifier.weight")  # <- Eğitim sırasında kullanılan ReID classifier çıkarılır
    if "head.reid_classifier.bias" in ckpt["model"]:
        ckpt["model"].pop("head.reid_classifier.bias")
    model.load_state_dict(ckpt["model"], strict=False)
    
    # 3. Takip başlat
    *_, summary = evaluator.evaluate_TCB(
        model, is_distributed, args.fp16, trt_file, decoder, exp.test_size, results_folder
    )
```

**Önemli noktalar:**
- `get_model2()` → `YOLOXHead2` kullanan model (ReID embedding tabanlı)
- `model.eval()` → Batch norm'lar freeze edilir, dropout kapatılır
- Eğitim sırasında kullanılan ReID classifier (`nID` sınıf için) kaldırılır (sadece embedding ihtiyacı var)

---

### 1.2 `TCB/yolox/evaluators/mot_evaluator_mot17.py` - evaluate_TCB()

```python
def evaluate_TCB(self, model, distributed=False, half=False, ...):
    model = model.eval()
    if half:
        model = model.half()
    
    tensor_type = torch.cuda.HalfTensor if half else torch.cuda.FloatTensor
    
    tracker = BYTETracker(self.args)  # Takipçi başlat
    
    # Dataloader üzerinde loop (her frame/batch için)
    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
        with torch.no_grad():  # <- Gradient hesaplaması devre dışı (inference modunda)
            frame_id = info_imgs[2].item()
            video_id = info_imgs[3].item()
            
            # Frame numarası 1 ise yeni tracker oluştur
            if frame_id == 1:
                tracker = BYTETracker(self.args, params)
            
            # Görüntüyü uygun tensor tipine dönüştür
            imgs = imgs.type(tensor_type)
            
            # ===== MODEL FORWARD PASS =====
            outputs = model(imgs)  # <- YOLOX2.forward() → (Backbone + Head)
            
            # ===== POSTPROCESS =====
            outputs = postprocess(outputs, self.num_classes, self.confthre, self.nmsthre)
            
            # ===== TRACKER UPDATE =====
            if outputs[0] is not None:
                online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
                # sonra sonuçları dosyaya yaz...
```

**Kontrol akışı:**
1. Resim yükleniyor: `imgs` → [batch, 3, H, W]
2. Model çalıştırılıyor: `outputs = model(imgs)`
3. Postprocess: threshold + NMS uygulanıyor
4. Tracker güncelleniyor: yeni deteksiyon ve eski track'ler eşleştiriliyor

---

## BÖLÜM 2: Model Forward Pass

### 2.1 `YOLOX2.forward()` - Backbone + Head Kombinasyonu

```python
# yolox2.py
class YOLOX2(nn.Module):
    def forward(self, x, targets=None, last_reid=None):
        if not self.training:  # <- İnference modunda
            # x shape: [batch, 3, H, W] örneğin [1, 3, 800, 1440]
            
            # Step 1: Backbone (Feature Pyramid Network)
            fpn_outs = self.backbone(x)
            # fpn_outs: 3 aşamalı feature map listesi
            # [
            #   [1, 256, 100, 180],  stride=8 ile
            #   [1, 512, 50, 90],    stride=16 ile
            #   [1, 1024, 25, 45]    stride=32 ile
            # ]
            
            # Step 2: Head (Detection + ReID Embedding)
            outputs = self.head(xin=fpn_outs, imgs=x, last_reid=last_reid)
            # outputs: shape [batch, 23625, 134]
            # 23625 = 100*180 + 50*90 + 25*45 (tüm FPN seviyelerindeki anchor sayısı)
            # 134 = 4 (bbox) + 1 (obj_conf) + 80 (class_probs) + 512 (embedding)
            
            return outputs
        else:  # Training mode
            # Training sırasında farklı şekilde işleniyor
            # (inference için önemli değil)
            pass
```

**Backbone nedir?**
- `YOLOPAFPN` → CSPNet + PANet (Feature Pyramid Network)
- Multi-scale features üretir: stride 8, 16, 32
- Her seviyede farklı resolution: 100x180, 50x90, 25x45

---

### 2.2 `YOLOXHead2.forward()` - İki Başlıklı Çıkarım

```python
# yolo_head2.py
class YOLOXHead2(nn.Module):
    def __init__(self, num_classes, ...):
        self.emb_dim = 512  # <- Embedding boyutu
        self.num_classes = 80  # örneğin COCO için
        
        # Her FPN seviyesi için paralel branchlar
        self.cls_convs = nn.ModuleList()   # Sınıf tahmin conv'ları
        self.reg_convs = nn.ModuleList()   # Bbox regresyon conv'ları
        self.reid_convs = nn.ModuleList()  # ReID embedding conv'ları
        
        # ...
    
    def forward(self, xin, labels=None, imgs=None, last_reid=None, feature_id=None):
        # xin: üç FPN seviyesi [[1, 256, 100, 180], [1, 512, 50, 90], [1, 1024, 25, 45]]
        
        outputs = []
        
        # Her FPN seviyesi için
        for k, (cls_conv, reg_conv, stride_this_level, x) in enumerate(
            zip(self.cls_convs, self.reg_convs, self.strides, xin)
        ):
            # x: örneğin [1, 256, 100, 180]
            
            # Stem: reduction channel
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
        
        # outputs artık 3 seviye için: [
        #   [1, 134, 100, 180],
        #   [1, 134, 50, 90],
        #   [1, 134, 25, 45]
        # ]
        
        if not self.training:  # <- İnference modunda
            # decode_outputs() ile tüm başlıkları düzleştir ve grid koordinatlarını ekle
            self.hw = [x.shape[-2:] for x in outputs]
            yolo_outputs = self.decode_outputs(outputs, dtype=xin[0].type())
            
            # final output: tüm seviyeleri birleştir
            return torch.cat([
                x.view(1, -1, 1+4+self.num_classes+self.emb_dim) 
                for x in yolo_outputs
            ], dim=1)
            # return shape: [1, 23625, 134]
```

**Animation (düşünsel):**

```
İnput: [1, 3, 800, 1440]
           ↓
        BACKBONE (YOLOPAFPN)
           ↓
    3 FPN levels:
    ├─ [1, 256, 100, 180]  ← stride 8
    ├─ [1, 512, 50, 90]    ← stride 16
    └─ [1, 1024, 25, 45]   ← stride 32
           ↓
    HEAD (YOLOXHead2) - Her seviye için paralel ↓
    ├─ Detection: [bbox(4) + obj(1) + cls(80)]
    ├─ Embedding: [reID(512)]
    └─ Concat: 134 channels
           ↓
    After concat and decode:
    [1, 100*180 + 50*90 + 25*45, 134]
    [1, 23625, 134]
```

---

### 2.3 `decode_outputs()` - Çıktı Decodlaması

```python
def decode_outputs(self, outputs, dtype):
    # outputs: 3 seviyesi [
    #   [1, 134, 100, 180],
    #   [1, 134, 50, 90],
    #   [1, 134, 25, 45]
    # ]
    
    out = []
    
    for (hsize, wsize), stride, output in zip(self.hw, self.strides, outputs):
        # stride: 8, 16, veya 32
        # hsize, wsize: görüntü seviyesi boyutları
        
        # Step 1: Flatten spatial dimensions
        output = output.flatten(start_dim=2).permute(0,2,1)
        # [1, 134, 100, 180] → [1, 100*180, 134] → [1, 18000, 134]
        
        # Step 2: Create grid
        yv, xv = torch.meshgrid([torch.arange(hsize), torch.arange(wsize)])
        grid = torch.stack((xv, yv), 2).view(1,-1,2).type(dtype)
        # grid shape: [1, 18000, 2] containing (x_grid, y_grid) for each anchor
        # örneğin: grid[0, 0] = [0, 0], grid[0, 1] = [1, 0], vb.
        
        # Step 3: Decode coordinates
        output[...,:2] = ((output[...,:2] + grid) * stride).type(dtype)
        # raw x,y offsets → grid-relative → stride-scaled
        # x_decoded = (x_offset + x_grid) * stride
        # bu şekilde original image space'e akıllıkça coordinate dönüştürülür
        
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

**Koordinat Transformasyonu Açıklaması:**

```
Her FPN seviyesi için:
- Raw model output: [-0.5, 0.3, ...] (center offset relative to grid cell)
- Grid: [[0,0], [1,0], [2,0], ..., [179, 99]] (tüm grid hücreleri)
- Stride: 8 (orijinal 800x1440 → 100x180)

Örneğin grid cell (50, 40) için:
- x_decoded = (-0.5 + 50) * 8 = 396 pixels (orijinal image'da)
- y_decoded = (0.3 + 40) * 8 = 325.4 pixels

Böylece model output doğrudan orijinal image coordinates'ye çevrilir.
```

---

## BÖLÜM 3: Postprocess (NMS ve Threshold)

### 3.1 `postprocess()` Fonksiyonu - Deteksiyonları Filtrele

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
    # Artık prediction[:, :, :4] ∈ [x1, y1, x2, y2] formatında
    
    output = [None for _ in range(len(prediction))]  # Batch başına sonuçlar
    
    # Step 2: Process each image in batch
    for i, image_pred in enumerate(prediction):  # image_pred: [23625, 134]
        
        if not image_pred.size(0):
            continue
        
        # Get class with highest confidence
        class_conf, class_pred = torch.max(
            image_pred[:, 5:5+num_classes],  # tüm class probabilities
            1,                                # max along class dimension
            keepdim=True
        )
        # class_conf: [23625, 1] - en yüksek logit confidence
        # class_pred: [23625, 1] - class index (0-79)
        
        # Step 3: Confidence filtering
        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # obj_confidence * class_confidence >= threshold
        # Yani: obj_score × class_score ≥ 0.7 (default)
        # FALSE: confidence düşük deteksiyon → filtrelenir
        
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
        # Artık sadece confidence > 0.7 olan deteksiyon'lar kaldı
        
        if not detections.size(0):
            continue  # Hiç deteksiyon kalmadıysa, sonraki image'a geç
        
        # Step 6: NMS (Non-Maximum Suppression)
        nms_out_index = torchvision.ops.batched_nms(
            detections[:, :4],              # bboxes
            detections[:, 4] * detections[:, 5],  # confidence scores (obj × class)
            detections[:, 6],               # class indices
            nms_thre                        # IoU threshold (0.45)
        )
        # nms_out_index: kepilecek deteksiyon indeksleri
        # NMS: Aynı sınıfta, high IoU olan düşük confidence box'ları siler
        
        detections = detections[nms_out_index]  # [M, 519] where M <= N
        
        # Step 7: Store output
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))
    
    # output: [tensor1, None, tensor2, ...]
    # her tensor shape: [num_dets_after_nms, 519]
    #   num_dets_after_nms: 50-500 arası tipik olarak
    return output
```

**Postprocess Pipeline:**

```
Input: [1, 23625, 134]
   ↓
Confidence filtering (threshold=0.7):
   obj_score × class_score ≥ 0.7
   ↓ (tipik olarak 500-2000 deteksiyon kalır)
   ↓
NMS (IoU threshold=0.45):
   Aynı sınıftaki overlapping box'ları filtrele
   ↓ (tipik olarak 50-300 deteksiyon kalır)
   ↓
Output: [num_dets, 519]
   [:, :4] = bbox [x1, y1, x2, y2]
   [:, 4] = obj_confidence
   [:, 5] = class_confidence
   [:, 6] = class_id
   [:, 7:519] = embedding (512 dim)
```

---

## BÖLÜM 4: Tracker Başlatma ve İlk Setup

### 4.1 BYTETracker.__init__()

```python
# byte_tracker.py (my_byte_tracker_mot17_kal.py)
from yolox.tracker.my_byte_tracker_mot17_kal import BYTETracker

class BYTETracker(object):
    def __init__(self, args, params=[-100, 0.9, 100, -100, -100]):
        self.tracked_stracks = []   # Şu anda aktif olarak takip edilen objeler
        self.lost_stracks = []      # Kaybolan objeler (birkaç frame'e kadar beklenir)
        self.removed_stracks = []   # Kalıcı olarak kaldırılan objeler
        
        self.frame_id = 0           # Mevcut frame numarası
        self.args = args            # Command-line argümanları
        self.det_thresh = args.track_thresh + 0.1  # Yeni track'ler için threshold
        self.buffer_size = int(frame_rate / 30.0 * args.track_buffer)
        # buffer_size: kaybolan track'i ne kadar süre bekleyeceğimiz
        # Tipik: 30 frame
        
        self.max_time_lost = self.buffer_size
        self.kalman_filter = KalmanFilter()  # Kalman filtresi başlat
        
        self.params = params  # [match_thresh_stage1, iou_thresh_gate, ...]
```

**Data Structures:**

```python
# STrack nesneleri track'leri temsil eder
class STrack:
    def __init__(self, tlwh, score, temp_feat, buffer_size=60):
        self._tlwh = tlwh  # Top-left-width-height
        self.score = score  # Detection confidence
        
        self.curr_feat = None  # Current embedding feature [512,]
        self.update_features(temp_feat)  # İlk feature'ı ayarla
        self.alpha = 0.1  # EMA update coefficient
        
        self.kalman_filter = None  # Will be set when activated
        self.mean, self.covariance = None, None  # Kalman state
        self.is_activated = False  # Did we confirm this track yet?
        self.track_id = -1  # Track ID (negative = not assigned yet)
        self.frame_id = None  # Frame where this track was first seen
        self.state = TrackState.NEW  # Can be: NEW, TRACKED, LOST, REMOVED
```

---

## BÖLÜM 5: Tracker.update() - Takip Ana Döngüsü

### 5.1 tracker.update() Yapısı

```python
def update(self, output_results, img_info, img_size, id_feature=None):
    """
    Args:
        output_results: [num_dets, 519]
           [:, :4] = bbox, [:, 4] = obj_conf, [:, 5] = cls_conf,
           [:, 6] = cls_id, [:, 7:] = embedding
        
        img_info: [H, W, 1, 1, path_string]
        img_size: (input_H, input_W) örneğin (800, 1440)
    
    Returns:
        output_stracks: List of STrack objects (only activated ones)
    """
    
    self.frame_id += 1
    
    # ===== STEP 0: Deteksiyonları parse et =====
    if output_results.shape[1] == 5:  # Eski format (sadece bbox + conf)
        scores = output_results[:, 4]
        bboxes = output_results[:, :4]
    else:  # Yeni format (bbox + conf + class + embedding)
        output_results = output_results.cpu().numpy()
        scores = output_results[:, 4] * output_results[:, 5]  # obj_conf × class_conf
        bboxes = output_results[:, :4]  # [x1, y1, x2, y2]
        id_feature = torch.tensor(output_results[:, 7:])  # [512,] embedding
    
    img_h, img_w = img_info[0], img_info[1]
    scale = min(img_size[0] / float(img_h), img_size[1] / float(img_w))
    bboxes /= scale  # Model input size'dan orijinal image size'a dönüştür
    
    # ===== STEP 1: Deteksiyonları confidence seviyesine göre böl =====
    remain_inds = scores > self.args.track_thresh  # High confidence (> 0.6 tipik)
    inds_low = scores > 0.1
    inds_high = scores < self.args.track_thresh
    
    inds_second = np.logical_and(inds_low, inds_high)  # 0.1 < score < 0.6
    
    dets_second = bboxes[inds_second]  # İkinci eşleme için düşük skor deteksiyon'lar
    dets = bboxes[remain_inds]  # Birinci eşleme için yüksek skor deteksiyon'lar
    scores_keep = scores[remain_inds]
    scores_second = scores[inds_second]
    id_feature_keep = id_feature[remain_inds]
    id_feature_second = id_feature[inds_second]
    
    # ===== STEP 2: Deteksiyonları STrack objelerine dönüştür =====
    if len(dets) > 0:
        detections = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 60)
            for (tlbr, s, f) in zip(dets, scores_keep, id_feature_keep)
        ]
    else:
        detections = []
    
    # ===== İlgili kısımlar devam ediyor... =====
```

---

### 5.2 Takip Listesinin Organize Edilmesi

```python
    # ===== STEP 3: Mevcut track'leri kategorize et =====
    unconfirmed = []
    tracked_stracks = []
    
    for track in self.tracked_stracks:
        if not track.is_activated:
            unconfirmed.append(track)  # Yeni track, henüz teyit edilmedi
        else:
            tracked_stracks.append(track)  # Kanıtlanmış track
    
    # ===== STEP 4: Track pool oluştur =====
    strack_pool = joint_stracks(tracked_stracks, self.lost_stracks)
    # strack_pool: kaybolan track'ler + aktif track'ler
    # Bu pool'un içindeki track'ler frame N-1'de gördüğümüz objeler
    
    # ===== STEP 5: Kalman Filter Prediction =====
    STrack.multi_predict(strack_pool)
    # Her track'in konumunu bir frame ileri tahmin et (Kalman momentum)
    # Daha sonra current frame'deki deteksiyon'larla eşleştirmeyi kolaylaştırır
```

**Track States & Pools Animation:**

```
Küresel Track Liste (self.tracked_stracks):
├─ Track#1 (is_activated=True)    → tracked_stracks'a gider
├─ Track#2 (is_activated=True)    → tracked_stracks'a gider
├─ Track#3 (is_activated=False)   → unconfirmed'a gider
└─ Track#4 (is_activated=True)    → tracked_stracks'a gider

Kayıp Liste (self.lost_stracks):
├─ Track#5 (gözlemlenmeyen 3 frame)
├─ Track#6 (gözlemlenmeyen 1 frame)
└─ ...

strack_pool = tracked_stracks + lost_stracks
├─ Track#1, Track#2, Track#4 (from tracked)
├─ Track#5, Track#6 (from lost)
└─ (Track#3 hariç çünkü unconfirmed'a ayrıldı)

Bu pool, deteksiyon'larla eşleştirilecek aday track'leri temsil eder.
```

---

### 5.3 Birinci Eşleme Aşaması (High Score Detections)

```python
    # ===== STEP 6: İlk matching - yüksek confidence deteksiyon'lar =====
    
    if len(strack_pool) > 0 and len(detections) > 0:
        # === Sub-step 6a: Özel embedding-tabanlı maskeleme ===
        motion = match2(strack_pool, id_feature_keep, pd=self.params[0])
        # match2: Cosine similarity matrisinin thresholded hali
        # motion: [len(strack_pool), len(detections)] binary matrix
        # motion[i, j] = True → track i ve detection j'nin embedding'leri çok farklı
        
        s1 = match3(strack_pool, id_feature_keep, pd=self.params[0])
        # s1: Normalized cosine similarity matrisinin kendisi
        # s1 ∈ [0, 1], higher = daha benzer embedding
        
        # === Sub-step 6b: IoU-tabanlı cost matrix ===
        dists = matching.iou_distance(strack_pool, detections)
        # dists: [len(strack_pool), len(detections)]
        # dists[i, j] = 1 - IoU(track_i, detection_j)
        # dists ∈ [0, 1], lower = better match (IoU açısından)
        
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
            # Detection confidence'ı IoU distance'ına entegre et
            # daha yüksek confidence deteksiyon'lar tercih edilir
        
        # === Sub-step 6c: Embedding bilgisini dists'a entegre et ===
        dists = 1 - (1 - dists) * np.array(s1)
        # Açıklama:
        #   dists'ın orijinal değeri: 1 - IoU
        #   s1'ın değeri: cosine_similarity ∈ [0, 1]
        #   Formül: dists_new = 1 - (1 - dists) * s1
        #   
        # Eğer s1 = 0 (embedding çok farklı):
        #   dists_new = 1 - (1 - dists) * 0 = 1 (matching imkansız)
        # Eğer s1 = 1 (embedding aynı):
        #   dists_new = 1 - (1 - dists) * 1 = dists (değişmez)
        
        # motion matrisini de uygulan: çok farklı embedding'ler → cost = 1
        if dists.shape == (1, 1) and motion.shape == (1, 1):
            if motion[0, 0]:
                dists[0, 0] = 1
        else:
            dists[motion] = 1  # motion[i,j] = True ise dists[i,j] = 1 yap
        
        # === Sub-step 6d: Hungarian Algorithm ile eşleme ===
        matches, u_track, u_detection = matching.linear_assignment(
            dists, thresh=self.params[1]  # params[1] = 0.9 tipik
        )
        # matches: [(track_idx, det_idx), ...]  matched pairs
        # u_track: track indeksleri eşleştirilemeyen
        # u_detection: detection indeksleri eşleştirilemeyen
    
    else:
        # Fallback if no tracks or detections
        dists = matching.iou_distance(strack_pool, detections)
        if not self.args.mot20:
            dists = matching.fuse_score(dists, detections)
        matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.params[1])
    
    # === Sub-step 6e: Eşleşmiş track'leri güncelle ===
    for itracked, idet in matches:
        track = strack_pool[itracked]
        det = detections[idet]
        
        if track.state == TrackState.Tracked:  # Aktif track
            track.update(det, self.frame_id, det.score > 0.7)
            # update: Kalman filtresi güncelle, embedding'i EMA'yla güncelle
            activated_starcks.append(track)
        
        else:  # Lost track (eski)
            track.re_activate(det, self.frame_id, new_id=False)
            # re_activate: Kalman filtresi geri başlat, tekrar track'ı etkinleştir
            refind_stracks.append(track)
```

**Matching Algoritması Detaylı:**

```
Girdi:
- strack_pool: N eski objeler
- detections: M yeni tespit
- dists: [N, M] matris

Hungarian Algorithm:
  dists'ı minimuma getirecek şekilde N×M eşleşmeyi bul
  
Output:
- matches ⊆ N×M (optimal ve threshold'ın altında)
- u_track: eşleştirilemeyen strack indeksleri
- u_detection: eşleştirilemeyen detection indeksleri
```

---

### 5.4 İkinci Eşleme Aşaması (Low Score Detections)

```python
    # ===== STEP 7: İkinci matching - düşük confidence deteksiyon'lar =====
    
    if len(dets_second) > 0:
        detections_second = [
            STrack(STrack.tlbr_to_tlwh(tlbr), s, f, 30)
            for (tlbr, s, f) in zip(dets_second, scores_second, id_feature_second)
        ]
    else:
        detections_second = []
    
    # İlk matching'ten eşleştirilemeyen tracked track'ler
    r_tracked_stracks = [
        strack_pool[i] for i in u_track
        if strack_pool[i].state == TrackState.Tracked
    ]
    # u_detection: 1. matching'ten eşleştirilemeyen yüksek skor deteksiyon'lar
    detections = [detections[i] for i in u_detection]
    
    # Düşük skor deteksiyon'larla tracked track'leri eşleştir
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
    
    # Eşleşmiş track'leri güncelle
    for itracked, idet in matches:
        track = r_tracked_stracks[itracked]
        det = detections_second[idet]
        
        if track.state == TrackState.Tracked:
            track.update(det, self.frame_id, False)  # Güvenli güncelleme
            activated_starcks.append(track)
        else:
            track.re_activate(det, self.frame_id, new_id=False)
            refind_stracks.append(track)
    
    # Eşleştirilemeyen track'leri kaybolan listesine taşı
    for it in u_track:
        track = r_tracked_stracks[it]
        if not track.state == TrackState.Lost:
            track.mark_lost()
            lost_stracks.append(track)
```

---

### 5.5 Unconfirmed Track'leri İşle

```python
    # ===== STEP 8: Unconfirmed track'leri işle =====
    # Unconfirmed: henüz bir kez eşleştirilmemiş yeni track'ler
    
    # 1. ve 2. matching'ten kalan eşleştirilemeyen deteksiyon'lar
    detections = [detections[i] for i in u_detection]
    
    # Unconfirmed track'lerle eşleştir
    dists = matching.iou_distance(unconfirmed, detections)
    if not self.args.mot20:
        dists = matching.fuse_score(dists, detections)
    
    matches, u_unconfirmed, u_detection = matching.linear_assignment(dists, thresh=0.7)
    
    for itracked, idet in matches:
        unconfirmed[itracked].update(detections[idet], self.frame_id, detections[idet].score > 0.7)
        activated_starcks.append(unconfirmed[itracked])  # Şimdi aktif hale gelir
    
    # Iki frame boyunca eşleştirilemeyen unconfirmed track'leri sil
    for it in u_unconfirmed:
        track = unconfirmed[it]
        track.mark_removed()
        removed_stracks.append(track)
```

---

### 5.6 Yeni Track'ler Başlat

```python
    # ===== STEP 9: Yeni objeler için yeni track'ler başlat =====
    
    for inew in u_detection:  # Tüm matching'lerden eşleştirilemeyen deteksiyon'lar
        track = detections[inew]
        
        if track.score < self.det_thresh:  # det_thresh = 0.6 + 0.1 = 0.7 tipik
            continue  # Çok düşük confidence → yeni track başlatma
        
        track.activate(self.kalman_filter, self.frame_id)
        # activate: track_id ata, Kalman state başlat, is_activated=True (henüz değişiklik olmadı)
        # 1. frame'de is_activated = True (özellik: MOT challenge tanımı)
        # sonraki frame'lerde is_activated = False olur ve eşleştirildikten sonra True olur
        activated_starcks.append(track)
```

---

### 5.7 State Management

```python
    # ===== STEP 10: Track state yönetimi =====
    
    # Çok uzun süre kayıp kalan track'leri tamamen kaldır
    for track in self.lost_stracks:
        if self.frame_id - track.end_frame > self.max_time_lost:  # 30 frame
            track.mark_removed()
            removed_stracks.append(track)
    
    # ===== STEP 11: Global track listesini güncelle =====
    self.tracked_stracks = [t for t in self.tracked_stracks if t.state == TrackState.Tracked]
    self.tracked_stracks = joint_stracks(self.tracked_stracks, activated_starcks)
    self.tracked_stracks = joint_stracks(self.tracked_stracks, refind_stracks)
    
    self.lost_stracks = sub_stracks(self.lost_stracks, self.tracked_stracks)
    self.lost_stracks.extend(lost_stracks)
    self.lost_stracks = sub_stracks(self.lost_stracks, self.removed_stracks)
    
    self.removed_stracks.extend(removed_stracks)
    
    # Duplicate track'leri kaldır (çok yakın olan track'ler)
    self.tracked_stracks, self.lost_stracks = remove_duplicate_stracks(
        self.tracked_stracks, self.lost_stracks
    )
    
    # ===== STEP 12: Sonuç dönür =====
    output_stracks = [track for track in self.tracked_stracks if track.is_activated]
    
    return output_stracks  # Yalnızca aktif (teyit edilmiş) track'ler
```

---

## BÖLÜM 6: Feature Update (Embedding EMA)

### 6.1 STrack.update_features() - EMA Logic

```python
class STrack:
    def __init__(self, tlwh, score, temp_feat, buffer_size=60):
        self.curr_feat = None  # Current embedding
        self.update_features(temp_feat)  # İlk embedding'i ata
        self.alpha = 0.1  # EMA coefficient
    
    def update_features(self, feat):
        """
        İkon embedding'i Exponential Moving Average (EMA) ile güncelle
        
        Args:
            feat: yeni embedding vektörü, shape [512]
        """
        if self.curr_feat is None:
            # İlk kez: doğrudan ata
            self.curr_feat = feat
        else:
            # Sonraki güncellemeler: EMA kombinasyonu
            self.curr_feat = self.curr_feat * (1 - self.alpha) + feat * self.alpha
            # curr_feat = 0.9 * old_feat + 0.1 * new_feat
            # Ağırlık: geçmiş bilgi %90, yeni bilgi %10
```

**EMA Açıklaması:**

```
Frame 1: feat_1 = [a1, a2, ..., a512]
  curr_feat = feat_1

Frame 2: feat_2 = [b1, b2, ..., b512]
  curr_feat = 0.9 * [a1, ...] + 0.1 * [b1, ...]
  
Frame 3: feat_3 = [c1, c2, ..., c512]
  curr_feat = 0.9 * (0.9*feat_1 + 0.1*feat_2) + 0.1 * feat_3
           = 0.81*feat_1 + 0.09*feat_2 + 0.1*feat_3

Ağırlıklar (yeni zamanlar yerel):
- 3 frame öncesine kıyasen: 0.81 * 0.1 ≈ 0.081 (çok düşük)
- 2 frame öncesine kıyasen: 0.09 (daha yüksek)
- Şu an (frame 3): 0.1 (en yüksek)

NOT: Geçmiş bilgi exponentially decay ediyor
→ Son gözlem'e ağırlık verilir
→ Ancak ani kamera görüntü değişimlerinden korunur
```

---

## BÖLÜM 7: Matching Algoritması (Hungarian)

### 7.1 `matching.linear_assignment()` - Hungarian Algorithm

```python
# matching.py
def linear_assignment(cost_matrix, thresh):
    """
    Cost matrix'i Hungarian algorithm ile min-cost matching'e çöz
    
    Args:
        cost_matrix: [N, M] matrix
        thresh: cost threshold (above this → không assign)
    
    Returns:
        matches: [(i, j), ...] optimal matches
        unmatched_a: unmatched row indices
        unmatched_b: unmatched column indices
    """
    
    if cost_matrix.size == 0:
        return (np.empty((0, 2), dtype=int),
                tuple(range(cost_matrix.shape[0])),
                tuple(range(cost_matrix.shape[1])))
    
    # Step 1: Hungarian algorithm çöz (LAP - Linear Assignment Problem)
    cost, x, y = lap.lapjv(cost_matrix, extend_cost=True, cost_limit=thresh)
    # cost: total optimal cost
    # x: her row'un hangi column'a assign olduğu (-1 = unassigned)
    # y: her column'un hangi row'a assign olduğu (-1 = unassigned)
    
    # Step 2: Matched pairs'ı çıkar
    matches = []
    for ix, mx in enumerate(x):
        if mx >= 0:  # Assigned
            matches.append([ix, mx])
    
    matches = np.asarray(matches)
    
    # Step 3: Unmatched rows/columns'ı belirle
    unmatched_a = np.where(x < 0)[0]  # Rows without assignment
    unmatched_b = np.where(y < 0)[0]  # Columns without assignment
    
    return matches, unmatched_a, unmatched_b
```

**Hungarian Algorithm Visualization:**

```
Cost Matrix örneği (3 track, 4 detection):
        Det0  Det1  Det2  Det3
Track0  0.2   0.8   0.9   0.7
Track1  0.9   0.1   0.7   0.8
Track2  0.7   0.9   0.2   0.6

Hungarian algorithm hedefi: toplam cost'u minimize etmek
Olası sonuçlar:
1. Track0→Det0(0.2), Track1→Det1(0.1), Track2→Det2(0.2) = 0.5 ✓ OPTIMAL
2. Track0→Det1(0.8), Track1→Det0(0.9), Track2→Det2(0.2) = 1.9
3. Track0→Det3(0.7), Track1→Det2(0.7), Track2→Det0(0.7) = 2.1

Sonuç:
- matches = [(0,0), (1,1), (2,2)]
- unmatched_a = []  (tüm track'ler assign edildi)
- unmatched_b = [3]  (Det3 assign edilmedi)
```

---

## BÖLÜM 8: Kalman Filter (Prediction)

### 8.1 STrack.multi_predict() - Kalman Prediction

```python
@staticmethod
def multi_predict(stracks):
    """
    Tüm track'lerin Kalman state'lerini bir frame ileri tahmin et
    
    Args:
        stracks: List[STrack]
    """
    if len(stracks) > 0:
        # Yığın halinde Kalman state'lerini çıkar
        multi_mean = np.asarray([st.mean.copy() for st in stracks])
        # [N, 7] - her track için: [cx, cy, w, h, vx, vy, ...]
        
        multi_covariance = np.asarray([st.covariance for st in stracks])
        # [N, 7, 7] - her track için 7x7 covariance matrix
        
        # Kalman prediction: x_{t+1} = F * x_t + noise
        multi_mean, multi_covariance = STrack.shared_kalman.multi_predict(
            multi_mean, multi_covariance
        )
        # multi_mean: [N, 7] - tahmin edilen state
        # multi_covariance: [N, 7, 7] - tahmin edilen belirsizlik
        
        # Tahmin edilen state'leri track'lere geri ata
        for i, (mean, cov) in enumerate(zip(multi_mean, multi_covariance)):
            stracks[i].mean = mean
            stracks[i].covariance = cov
```

**Kalman State Representation:**

```
State vector: x = [cx, cy, w, h, vx, vy, ...] (7D veya 8D)
- cx, cy: center koordinatları (pixel)
- w, h: width, height (pixel)
- vx, vy: velocity (pixel/frame) - ne kadar hızlı hareket ediyor
- ... örneğin aspect ratio değişim hızı

Örnek önceki frame'de:
x_t = [100, 50, 200, 150, 5, 2]
     = center (100, 50), size (200, 150), velocity (5, 2) px/frame

Tahmin:
x_{t+1} = F @ x_t + noise
        ≈ [105, 52, 200, 150, 5, 2]  (±uncertainty)
        = center (105, 52), size (200, 150), velocity (5, 2)

Covariance: Her state element'inin ne kadar belirsiz olduğu
- Yeni track'ler: yüksek covariance
- Uzun takip edilen track'ler: düşük covariance
```

---

## BÖLÜM 9: Sonuç ve Output Yazma

```python
# mot_evaluator_mot17.py'den
def evaluate_TCB(...):
    # ... bir önceki kodlar ...
    
    for cur_iter, (imgs, _, info_imgs, ids) in enumerate(progress_bar(self.dataloader)):
        # ...
        online_targets = tracker.update(outputs[0], info_imgs, self.img_size)
        
        # online_targets: List[STrack] (activated ve tracked olan objeler)
        
        # Results yazma
        online_tlwhs = []
        online_ids = []
        online_scores = []
        
        for t in online_targets:
            tlwh = t.tlwh  # Top-left-width-height format
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > 1.6  # Çok dar/uzun box'ları filtrele
            
            if tlwh[2] * tlwh[3] > self.args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
        
        results.append((frame_id, online_tlwhs, online_ids, online_scores))
        
        # Frame sayısı video sonuna erişilirse
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

**Output Format Örneği:**

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
  track #1: (105, 52) + (200x150), conf=0.93 [Kalman prediction ile hareket]
  track #3: (450, 310) + (190x210), conf=0.85 [Yeni track]
```

---

## ÖZETLEME: Tam İnference Döngüsü

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
│    ├─ Kalman Predict: Track'lerin tahmini konumu               │
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

## 2. İki Başlıklı Çıkarım (JDE Forward Pass)

### `TCB/yolox/models/yolo_head2.py`
- `YOLOXHead2` hem deteksyon hem de ReID dalları içerir.
- Her bir FPN seviyesinde (`strides=[8,16,32]`):
  - `reg_output` -> bbox regresyonu
  - `obj_output.sigmoid()` -> obje güveni
  - `cls_output.sigmoid()` -> sınıf olasılıkları
  - `reid_output` -> yoğun embedding map, boyut `self.emb_dim = 512`
- Test modunda `forward()`:
  - `output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid(), reid_output],1)`
  - `decode_outputs()` ile sütunlar x,y,w,h formatına dönüştürülüyor
  - `return torch.cat([...], dim=1)` çıktısı [batch, tüm_ankorlar, 4+1+num_classes+512] şeklinde
- Bu, senin 5 adımlı akışındaki "ortak özellik haritası f_t" ve "embedding head" ile eşleşiyor.

## 3. Sparse Sampling / Noktasal Çekim

- `postprocess()` fonksiyonu, `YOLOXHead2` çıktısını kutulara ve skorları içeren deteksiyon listesine çevirir.
- `tracker.update()` çağrısı `outputs[0]` ve `id_feature` kullanarak yapılır.
- `my_byte_tracker_mot17_kal.py` içinde:
  - `STrack` nesnesi `curr_feat` alanına sahip
  - `detections` oluşturulurken her deteksiyona `f` (embedding) verisi ekleniyor
  - Yani embeddingler tüm görüntü için değil, `postprocess` sonrası seçilen tespit noktalarına karşılık gelen vektörler olarak kullanılıyor
- Bu, senin "sistemde sadece o (x, y) noktasında 1x512 özellik vektörü çekme" yaklaşımına denk geliyor.

## 4. Kalman'sız Maliyet Matrisi iddiası

### Gerçek durum
- `TCB/yolox/tracker/my_byte_tracker_mot17_kal.py` içinde Kalman filtresi halen aktif:
  - `STrack.multi_predict(strack_pool)` çağrısı var
  - `self.kalman_filter = KalmanFilter()` nesnesi kullanılıyor
  - `track.update(...)` ve `re_activate(...)` içinde Kalman güncellemesi devam ediyor
- Bu nedenle kodda "saf Kalman'sız" bir iz takip hattı değil.

### Ancak özel eşleme var
- `match2()` ve `match3()` fonksiyonları:
  - `E = torch.cat([track.curr_feat ...])` ile geçmiş şablon vektörleri alınır
  - `F = id_feature.permute(1,0)` ile yeni deteksiyon embeddingleri alınır
  - `M = cosine_similarity(E, F)` hesaplanır
  - `motion` / `s1` maskeleri kullanılarak bazı eşleşmeler `dists` matrisinde düzeltiliyor
- Yani gerçek kodda:
  - temel maliyet `matching.iou_distance()` ile IoU tabanlı
  - sonra `fuse_score()` ve embedding kaynaklı maskelerle düzeltme yapılıyor
- Bu, senin "Cost = Cosine_Similarity * IoU" formülüne yakın ama tam çarpım yerine farklı bir kombinasyon kullanıyor.

## 5. Macar Algoritması (Hungarian)

### Kullanılan yöntem
- `TCB/yolox/tracker/matching.py` içinde `linear_assignment(cost_matrix, thresh)`
  - `lap.lapjv()` kullanılarak Hungarian çözümü hesaplanıyor
  - Eşleşme, unmatched track ve unmatched detection sonuçları dönüyor
- `my_byte_tracker_mot17_kal.py` içinde bu fonksiyon birden fazla aşamada kullanılıyor:
  - İlk aşama: yüksek skor tespitleri
  - İkinci aşama: düşük skor tespitleri
  - Üçüncü aşama: unconfirmed trackler
- Yani global optimum değil, yine de Macar algoritması temelinde çok aşamalı bir eşleme var.

## 6. EMA ile Hafıza Güncellemesi

### `STrack.update_features()`
- `TCB/yolox/tracker/my_byte_tracker_mot17_kal.py` içinde:
  - `self.curr_feat` ilk kez atanırsa direkt kaydedilir
  - sonrasında `self.curr_feat = self.curr_feat * (1-self.alpha) + feat * self.alpha`
  - `alpha = 0.1` olarak belirlenmiş
- Bu net olarak bir EMA / low-pass güncellemesi.
- Bu yüzden kullanıcı tanımıyla "eşleşen hedefin hafızası körü körüne değiştirilmez" doğru.

## 7. `trainer.py` vs `trainer2.py`

- `TCB/yolox/core/trainer.py`:
  - `self.exp.get_model(settings=self.settings)` ile `YOLOXHead` tabanlı model kullanır
  - detection + basit ReID/TCL amaçlı değil, klasik YOLOX eğitimi mantığında
- `TCB/yolox/core/trainer2.py`:
  - `self.exp.get_model2(settings=self.settings)` ile `YOLOXHead2` tabanlı JDE modeli kullanır
  - bu repo için JDE / ReID içeren hattın gerçek eğitim hattı budur
- `track2.py` ise doğrudan `exp.get_model2()` kullandığı için inference `trainer2.py`'ın model yapısı ile uyumlu.

## 8. Heatmap / `m_t` bypass

- `YOLOXHead2` içinde eğitimde `checknet()` ve `get_reid()` fonksiyonları TCL / heatmap mantığını içeriyor.
- Test modunda (`if not self.training`) bu kodlar kullanılmıyor.
- Yani inference sırasında `heatmap` / `m_t` hesaplamaları pratikte bypass ediliyor.

## 9. Sonuç

Bu repo için inference hattı şöyle çalışıyor:

1. `track2.py` ile `exp.get_model2()` çağrılır.
2. `YOLOX2` + `YOLOXHead2` modeli, `model(imgs)` ile dense deteksiyon + dense embedding haritaları üretir.
3. `postprocess()` ile sadece seçilmiş kutular ve bunlara ait embedding vektörleri alınır.
4. `my_byte_tracker_mot17_kal.py` içindeki takipçi bu vektörleri IoU temelli ilk maliyetle, sonra embedding maskeleriyle düzeltir.
5. `linear_assignment()` ile eşleme yapılır.
6. `STrack.update_features()` içinde EMA benzeri embedding güncellemesi uygulanır.

> Not: Kodun içinde Kalman filtresi halen aktif. Eğer amacın inference'ta tamamen "Kalman'sız" bir hattı doğrulamak ise, `my_byte_tracker_mot17_kal.py` Kalman kullanımını kaldırmak veya bu dosyayı yerine tam ReID+IoU kombinasyonunu kullanmak gerekir.
