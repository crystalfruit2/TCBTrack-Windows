# TCBTrack / YOLOX-JDE Inference Teknik Özeti

Bu doküman, `TCBTrack` içindeki YOLOX-JDE inference hattını ve kodun izlediği adımları özetler.

## 1. Kullanılan inference hattı

### `TCB/yolox/tools/track2.py`
- `main()` fonksiyonunda `exp.get_model2()` çağrısı yapılıyor.
- Yani çalıştırılan model `YOLOX2` + `YOLOXHead2` tabanlı.
- `MOTEvaluator.evaluate_TCB()` çağrısı ile takip işlemi başlatılıyor.

### `TCB/yolox/evaluators/mot_evaluator_mot17.py`
- `evaluate_TCB()` içinde:
  - model `model.eval()` moduna alınıyor
  - `outputs = model(imgs)` ile doğrudan modelden çıkış alınıyor
  - `outputs = postprocess(outputs, ...)` ile NMS, threshold ve sınıf filtrelemesi uygulanıyor
  - `tracker.update(outputs[0], info_imgs, self.img_size)` ile takipçi güncelleniyor
- Bu demektir ki inference'ta heatmap veya `m_t` hesaplamaları burada kullanılmıyor.

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
