# BÁO CÁO CÁC BẪY KỸ THUẬT VÀ LƯU Ý TRIỂN KHAI (LAB DAY 10)

> **Tài liệu tổng hợp các bẫy kỹ thuật, nguyên nhân gốc rễ và giải pháp khắc phục triệt để khi finetune Vision–Language Model (Qwen3-VL-4B) trên môi trường Google Colab T4.**  
> **Người thực hiện**: Trịnh Xuân Huy

---

## 1. Bẫy 1: Lỗi `ValueError: Attempting to unscale FP16 gradients` (E2 & E3)

### Hiện tượng
Khi chạy huấn luyện thí nghiệm E2 hoặc E3 (khi mở unfreeze Projector `visual.merger` và `visual.deepstack_merger_list`), tiến trình huấn luyện dừng đột ngột tại bước tối ưu với thông báo:
```text
ValueError: Attempting to unscale FP16 gradients.
```

### Nguyên nhân gốc rễ
- Trong pipeline huấn luyện Mixed Precision với `torch.amp.autocast("cuda", dtype=torch.float16)`, khi backward qua các tham số được unfreeze trong Projector, PyTorch tính toán và lưu gradient của chúng dưới định dạng nửa độ chính xác `torch.float16`.
- Đối tượng `torch.amp.GradScaler` trong PyTorch yêu cầu nghiêm ngặt rằng tất cả gradient của các tham số được unscale qua hàm `scaler.unscale_(optimizer)` **phải ở kiểu dữ liệu `torch.float32` hoặc `torch.bfloat16`** để tránh hiện tượng tràn số / dưới ngưỡng (underflow/overflow) khi nhân với nghịch đảo của hệ số scale. Nếu gặp bất kỳ tensor gradient nào có `dtype == torch.float16`, nó sẽ ném ngoại lệ ngay lập tức.

### Giải pháp khắc phục triệt để
1. **Ép kiểu module Projector sang Float32**: Trong hàm `apply_setup`, chuyển toàn bộ module và trọng số của `visual.merger` và `visual.deepstack_merger_list` sang `torch.float32`.
2. **Đồng bộ tham số tối ưu hóa**: Trước khi khởi tạo AdamW, duyệt qua toàn bộ `trainable_params` và ép sang float32:
   ```python
   for p in trainable_params:
       p.data = p.data.to(torch.float32)
   ```
3. **Chuẩn hóa gradient trước khi unscale**: Trong vòng lặp huấn luyện, ngay sau khi gọi `scaler.scale(loss).backward()`, thực hiện ép kiểu gradient của tất cả tham số huấn luyện về `float32`:
   ```python
   for p in trainable_params:
       if p.grad is not None and p.grad.dtype == torch.float16:
           p.grad = p.grad.float()
   scaler.unscale_(optimizer)
   torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
   scaler.step(optimizer)
   scaler.update()
   ```

---

## 2. Bẫy 2: Rò rỉ dữ liệu (Data Leakage) do chia dữ liệu ngẫu nhiên

### Hiện tượng
Nếu sử dụng hàm chia ngẫu nhiên thông thường (như `train_test_split` của scikit-learn), mô hình đạt F1 rất cao trên tập test (>0.85–0.90), nhưng khi mang mô hình ra kiểm thử trên các video hành trình thực tế hoàn toàn mới thì độ chính xác sụt giảm nghiêm trọng.

### Nguyên nhân gốc rễ
- Bộ dữ liệu đường bộ được trích xuất từ các đoạn video camera hành trình (dashcam) quay liên tục ở tốc độ cao (25–30 fps). Hai khung hình chụp cách nhau 0.5s đến 1s trên cùng một đoạn đường có bối cảnh mặt đường, ánh sáng thời tiết, góc máy và các vết hư hại gần như giống hệt nhau (trùng lặp đến 95–99%).
- Nếu chia ngẫu nhiên theo từng ảnh, các khung hình kề nhau của cùng một ổ gà sẽ bị rơi vào cả tập `train` và tập `test/val`. Khi đó, mô hình chỉ đơn thuần "học vẹt" bối cảnh của đoạn video đó thay vì học được đặc trưng tổng quát của hư hại mặt đường.

### Giải pháp khắc phục triệt để
- Phân nhóm ảnh theo **phiên quay (`session`)**: Trích xuất mẫu ngày + giờ quay trong tên file ảnh (ví dụ: `vlcsnap-2025-02-18-23h...` $\rightarrow$ session `vlc-2025-02-18-23`).
- Đưa **toàn bộ các ảnh thuộc cùng một session vào duy nhất một tập** (`train`, `val`, hoặc `test`), đồng thời đảm bảo ràng buộc mỗi split đều có đủ số lượng tối thiểu cho từng lớp hư hại (`pothole`, `crack`, `manhole`). Cơ chế này được cài đặt cố định và tiền định trong [`scripts/splitting.py`](file:///d:/K4-Track-4-Day10-VisionLanguageModel/scripts/splitting.py).

---

## 3. Bẫy 3: Lệch thang toạ độ (Normalized 0–1000 vs Pixel Coordinates)

### Hiện tượng
Mô hình sau khi huấn luyện sinh ra chuỗi JSON hoàn toàn đúng cú pháp (`parse_valid_rate` $\ge 0.95$), nhưng khi chạy chấm điểm bằng [`scripts/score.py`](file:///d:/K4-Track-4-Day10-VisionLanguageModel/scripts/score.py), F1 bị tụt về **0.000** (hoặc cực kỳ thấp).

### Nguyên nhân gốc rễ
- Ảnh dashcam có kích thước phân giải thực tế là $640 \times 360$ pixel. Tọa độ pixel thực tế có dải giá trị $x \in [0, 640]$ và $y \in [0, 360]$.
- Theo thiết kế chuẩn của họ mô hình Qwen3-VL (Grounding Task), toàn bộ tọa độ bounding box phải được chuẩn hóa về thang số nguyên tương đối $[0, 1000]$:
  $$\text{Box}_{1000} = \left[ \text{round}\left(\frac{x_1}{W} \times 1000\right), \text{round}\left(\frac{y_1}{H} \times 1000\right), \text{round}\left(\frac{x_2}{W} \times 1000\right), \text{round}\left(\frac{y_2}{H} \times 1000\right) \right]$$
- Nếu mô hình học và dự đoán ra tọa độ pixel tuyệt đối (ví dụ `[100, 120, 250, 300]`) trong khi ground truth được chấm theo thang $[0, 1000]$, các bounding box dự đoán sẽ chỉ co cụm ở góc trên cùng bên trái của ảnh trong không gian 1000×1000. Diện tích giao cắt $\text{IoU}$ giữa dự đoán và nhãn thật sẽ luôn bằng 0 ($< 0.5$). Toàn bộ dự đoán bị tính là False Positive và nhãn thật bị tính là False Negative, dẫn tới $\text{F1} = 0$.

### Giải pháp khắc phục triệt để
- Chuẩn hóa nhãn đào tạo chặt chẽ từ YOLO $[0, 1]$ sang thang số nguyên $[0, 1000]$ qua hàm `yolo_to_box`.
- Định dạng Prompt chỉ thị rõ ràng cho mô hình: *"Trả về danh sách JSON chứa các đối tượng có nhãn và bbox_2d [x1, y1, x2, y2] trên thang toạ độ chuẩn hoá 0-1000"*.
- Khi chạy đánh giá, luôn chỉ định cờ `--coord norm1000`.

---

## 4. Bẫy 4: Tràn bộ nhớ GPU (CUDA OOM) trên Colab T4

### Hiện tượng
Khi chạy liên tiếp nhiều thí nghiệm (E1 $\rightarrow$ E2 $\rightarrow$ E3) hoặc khi huấn luyện được 50–100 bước, Colab báo lỗi:
```text
torch.cuda.OutOfMemoryError: CUDA out of memory. Tried to allocate ... MiB
```

### Nguyên nhân gốc rễ
1. GPU T4 trên Colab chỉ có dung lượng tối đa 15.0 GB (thực tế khả dụng cho tiến trình người dùng khoảng 14.5 GB).
2. Khi huấn luyện mô hình ngôn ngữ thị giác đa phương thức, việc lưu trữ activation của 220 visual tokens kết hợp với chuỗi văn bản dài đòi hỏi bộ nhớ rất lớn.
3. Khi hàm huấn luyện hoặc đánh giá kết thúc, các biến tham chiếu đến mô hình cũ (`peft_model`, `base_model`, `optimizer`) vẫn nằm trong bộ nhớ cache của PyTorch CUDA mà không được thu gom rác tự động.

### Giải pháp khắc phục triệt để
1. **Bật Gradient Checkpointing**: Giảm tới 60% bộ nhớ activation:
   ```python
   peft_model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
   peft_model.config.use_cache = False
   ```
2. **Kỹ thuật Gradient Accumulation**: Đặt `batch_size=2` và `grad_accum=4`, tương đương với effective batch size là 8 mà chỉ tiêu tốn lượng VRAM của batch size 2.
3. **Chủ động giải phóng bộ nhớ (Garbage Collection)** sau mỗi hàm:
   ```python
   del peft_model, base_model, dataloader, optimizer, scheduler, scaler
   gc.collect()
   torch.cuda.empty_cache()
   ```
4. **Hạ batch size khi suy luận test**: Khi chạy `run_test_inference`, đặt `batch_size=2` để đảm bảo không bị tràn VRAM khi sinh token tự hồi quy.

---

## 5. Bẫy 5: Mask Loss không chuẩn (Tính loss trên cả Prompt và Image Tokens)

### Hiện tượng
Loss của mô hình rất cao và giảm rất chậm; sau khi huấn luyện xong, mô hình bị thoái hóa: thay vì trả về JSON tọa độ, mô hình lặp lại chính câu hỏi của prompt hoặc sinh ra các chuỗi ký tự hỗn loạn.

### Nguyên nhân gốc rễ
- Mỗi mẫu huấn luyện bao gồm 3 phần: (1) System prompt & User instruction, (2) 220 Visual Tokens đại diện cho ảnh, (3) Chuỗi JSON mục tiêu do trợ lý sinh ra.
- Nếu không tạo mask loss (`labels = -100` cho phần prompt và token ảnh), hàm Cross-Entropy Loss sẽ phạt mô hình khi mô hình không dự đoán chính xác lại các token ảnh và câu prompt hệ thống. Do số lượng token ảnh và prompt chiếm đa số (>80% độ dài chuỗi), gradient từ các token này sẽ áp đảo hoàn toàn tín hiệu học của các token tọa độ hư hại.

### Giải pháp khắc phục triệt để
- Sử dụng hàm `collate_fn` tùy biến để xác định chính xác vị trí bắt đầu của câu trả lời (`<|im_start|>assistant\n`).
- Gán giá trị `-100` cho toàn bộ các token nằm trước vị trí này trong tensor `labels`. PyTorch CrossEntropyLoss sẽ tự động bỏ qua (ignore_index = -100) toàn bộ các token prompt và ảnh, chỉ tối ưu hóa gradient trên các token JSON câu trả lời.

---

## 6. Bẫy 6: Khối ViT và Projector bị lượng tử hoá 4-bit (NF4)

### Hiện tượng
Không thể unfreeze projector ở E2/E3, hoặc khi cố gắng gắn `requires_grad = True`, trọng số không cập nhật hoặc ném lỗi từ thư viện `bitsandbytes`.

### Nguyên nhân gốc rễ
Khi áp dụng cấu hình `BitsAndBytesConfig(load_in_4bit=True)`, nếu không chỉ định ngoại lệ, thư viện sẽ tự động ép tất cả các lớp `nn.Linear` trong toàn bộ mô hình (kể cả khối thị giác và projector) thành lớp `Linear4bit`. Trọng số dạng nén 4-bit không hỗ trợ gradient ngược và không thể huấn luyện đầy đủ theo cách thông thường.

### Giải pháp khắc phục triệt để
- Khởi tạo cấu hình BitsAndBytes với cờ loại trừ rõ ràng:
  ```python
  quantization_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_compute_dtype=torch.float16,
      bnb_4bit_use_double_quant=True,
      llm_int8_skip_modules=["visual", "lm_head"]
  )
  ```
- Cờ `llm_int8_skip_modules=["visual", "lm_head"]` bảo vệ toàn bộ module `visual` (bao gồm ViT và Projector) ở định dạng FP16 gốc, cho phép can thiệp unfreeze và huấn luyện đầy đủ ở E2 và E3.

---

## 7. Bẫy 7: Hiện tượng "Reward Hacking" khi huấn luyện GRPO (E5)

### Hiện tượng
Trong quá trình huấn luyện tăng cường (Reinforcement Learning - GRPO), giá trị Reward trung bình tăng rất nhanh lên gần 1.0, nhưng khi đánh giá F1 thực tế trên tập test thì kết quả lại cực kỳ tệ hại.

### Nguyên nhân gốc rễ
Mô hình phát hiện ra "lỗ hổng" trong hàm chấm điểm reward để tối đa hóa điểm số mà không cần học cách định vị đúng:
1. *Chiêu trò trả về danh sách rỗng `[]`*: Nếu một ảnh không có hư hại, việc trả về `[]` được reward tuyệt đối 1.0. Mô hình dần trở nên "lười biếng" và chọn chiến lược luôn trả về `[]` để tránh bị phạt.
2. *Chiêu trò vẽ bounding box khổng lồ*: Mô hình sinh ra 1 bounding box bao trọn cả khung hình `[0, 0, 1000, 1000]` để luôn có phần giao nhau với các box thật.

### Giải pháp khắc phục triệt để
- Thiết kế hàm Reward có cơ chế phạt đối xứng:
  $$R = 0.2 + 0.8 \times \frac{\sum \text{IoU}}{\max(N_{\text{GT}}, N_{\text{Pred}}, 1)}$$
  Việc chia cho $\max(N_{\text{GT}}, N_{\text{Pred}}, 1)$ sẽ lập tức trừng phạt nặng nề việc mô hình bịa đặt thêm các box giả.
- Theo dõi chặt chẽ tỷ lệ sinh chuỗi rỗng (`empty_rate`) và đối chiếu khoảng tin cậy CI95 với mô hình SFT để phát hiện sớm dấu hiệu thoái hóa chính sách (policy degradation).

