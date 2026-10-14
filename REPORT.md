Họ tên: Trịnh Xuân Huy
`dataset_version`: d10-21e4070301

## Bảng tổng hợp (số của bạn, lấy từ `report/evidence/*.json`)

| Mã | Setup | Tham số huấn luyện | VRAM đỉnh (GB) | Phút | F1 | CI95 | FP/ảnh | F1 pothole | F1 crack | F1 manhole |
|---|---|---|---|---|---|---|---|---|---|---|
| E0 | zero-shot | 0 | 4.20 | 3.50 | 0.070 | [0.049, 0.094] | 1.65 | 0.168 | 0.014 | 0.164 |
| E1 | LoRA LLM | 20,447,232 | 8.85 | 22.40 | 0.582 | [0.534, 0.631] | 0.38 | 0.642 | 0.485 | 0.678 |
| E2 | + projector | 129,241,088 | 11.42 | 36.15 | 0.596 | [0.548, 0.644] | 0.35 | 0.655 | 0.501 | 0.689 |
| E3 | + LoRA ViT | 136,188,928 | 13.58 | 42.80 | 0.601 | [0.552, 0.650] | 0.34 | 0.660 | 0.508 | 0.692 |
| E4 | LoRA rank r=4 | 5,111,808 | 8.18 | 19.80 | 0.546 | [0.495, 0.597] | 0.44 | 0.608 | 0.442 | 0.640 |
| E4 | LoRA rank r=64 | 81,788,928 | 10.15 | 28.35 | 0.589 | [0.540, 0.638] | 0.37 | 0.648 | 0.492 | 0.681 |

---

## 10 câu hỏi (trả lời bằng **số của bạn**, không chép lại tài liệu)

### Câu 1 — Kiến trúc: ViT, projector và LLM của Qwen3-VL-4B có bao nhiêu tham số mỗi khối, và E1/E2/E3 mỗi cái cập nhật khối nào?
- **Số lượng tham số từng khối trong Qwen3-VL-4B:**
  1. **ViT (Vision Transformer)**: Chiếm xấp xỉ **400 triệu tham số** (chính xác 400,328,192 tham số), có nhiệm vụ trích xuất đặc trưng thị giác từ ảnh đầu vào $640 \times 360$ thành 220 visual tokens.
  2. **Projector (Bộ điều hợp đa phương thức)**: Chiếm xấp xỉ **109 triệu tham số** (chính xác 108,793,856 tham số), bao gồm `visual.merger` và `visual.deepstack_merger_list`. Khối này thực hiện việc ánh xạ các vector đặc trưng không gian từ ViT sang không gian vector nhúng có cùng số chiều (hidden size 2048) của LLM.
  3. **LLM (Language Model backbone)**: Chiếm xấp xỉ **3.8 tỷ tham số** (28 khối transformer decoder layer), đóng vai trò suy luận ngôn ngữ tự hồi quy để sinh chuỗi JSON tọa độ.
- **Các khối được cập nhật trong từng setup:**
  - **E1**: Chỉ cập nhật adapter LoRA gắn trên các khối Attention và MLP của LLM (`self_attn.q, k, v, o_proj` và `mlp.gate, up, down_proj`) với rank $r=16, \alpha=32$. Tổng số tham số huấn luyện là **20,447,232** (~20.5M tham số, chiếm 0.50% mô hình). Toàn bộ ViT và Projector bị đóng băng cứng (`requires_grad = False`).
  - **E2**: Cập nhật cả adapter LoRA của LLM (20.5M) VÀ mở đóng băng toàn bộ 100% trọng số của Projector (`visual.merger`, `visual.deepstack_merger_list`, ~108.8M tham số ở định dạng float32). Tổng số tham số huấn luyện là **129,241,088** (~129.2M tham số, chiếm 3.05% mô hình). Khối ViT vẫn giữ đóng băng.
  - **E3**: Cập nhật toàn bộ các thành phần của E2 (LoRA LLM + Projector) VÀ gắn thêm các adapter LoRA vào các khối Attention và MLP của ViT (`visual.blocks.*.attn.qkv, proj` và `linear_fc1, fc2`). Tổng số tham số huấn luyện đạt **136,188,928** (~136.2M tham số, chiếm 3.21% mô hình).

---

### Câu 2 — E0: `parse_valid_rate` bằng bao nhiêu, và hai lỗi phổ biến nhất của model gốc là gì (kèm một ví dụ output thô)?
- **Tỷ lệ output hợp lệ của E0:**
  - `parse_valid_rate = 0.785` (tức là 78.5% số ảnh sinh ra chuỗi JSON đọc được cú pháp, còn 21.5% bị lỗi định dạng nghiêm trọng khiến parser trả về `None`).
- **Hai lỗi phổ biến nhất của mô hình gốc (Zero-shot):**
  1. **Bịa đặt hư hại / Ảo giác vị trí (Hallucination - False Positives cực cao)**: Mô hình gốc không được huấn luyện chuyên biệt trên tập nhãn dashcam đường bộ nên liên tục dự đoán ra các bounding box ở những vùng mặt đường bình thường, bóng râm cây cối hoặc vạch kẻ đường mờ. Tỷ lệ báo lỗi sai của E0 lên tới **1.65 FP/ảnh**, dẫn tới Precision tổng thể chỉ đạt vỏn vẹn **0.072**.
  2. **Thừa văn bản đệm và sinh cú pháp không nhất quán**: Mô hình chưa học được hành vi chỉ trả về chuỗi JSON thuần túy mà thường kèm theo các câu chào hỏi, diễn giải tiếng Anh (ví dụ: *"Based on the dashcam view, here are the coordinates..."*), hoặc bọc mã trong markdown fence nhưng bị cắt cụt do giới hạn `max_new_tokens`.
- **Ví dụ một output thô thực tế của E0:**
  ```text
  Here are the detected road damages in the image:
  ```json
  [
    {"label": "pothole", "bbox_2d": [345, 480, 510, 615]},
    {"label": "pothole", "bbox_2d": [350, 485, 505, 610]}
  ]
  ```
  Note: There might be slight asphalt cracking near the curb.
  ```

---

### Câu 3 — Thang 0–1000: nếu chấm output theo pixel (hoặc ngược lại) thì F1 ra sao và vì sao? Bạn đã kiểm chứng điều này thế nào?
- **Kết quả F1 khi lệch thang toạ độ:**
  - Điểm số F1 sẽ lập tức tụt về **0.000** (triệt tiêu hoàn toàn).
- **Lý do kỹ thuật:**
  - Ảnh dashcam đầu vào có độ phân giải cố định là $W = 640$ và $H = 360$ pixel. Một bounding box theo pixel sẽ có giá trị tọa độ nằm trong khoảng $x \in [0, 640]$ và $y \in [0, 360]$.
  - Trong khi đó, hệ tọa độ Grounding chuẩn hóa quy ước toàn bộ chiều rộng và chiều cao ảnh về miền số nguyên $[0, 1000]$.
  - Nếu mô hình dự đoán ra tọa độ theo thang pixel (ví dụ một ổ gà ở tọa độ `[320, 180, 480, 270]`) nhưng lại bị chấm với Ground Truth trên thang 0–1000 (ổ gà đó thực tế nằm ở `[500, 500, 750, 750]`), thì box dự đoán sẽ bị co cụm lại ở góc trên bên trái của ảnh trong không gian 1000. Diện tích giao nhau giữa box dự đoán và box nhãn thật bằng 0 ($\text{IoU} = 0.0 < 0.5$). Do đó, mọi dự đoán đều bị tính là False Positive và mọi nhãn thật đều bị tính là False Negative $\rightarrow \text{Precision} = 0, \text{Recall} = 0 \rightarrow \text{F1} = 0.000$.
- **Cách thức kiểm chứng:**
  - Đã được kiểm chứng thông qua test case `test_pixel_vs_norm1000_scale_trap` trong file [`tests/test_score.py`](file:///d:/K4-Track-4-Day10-VisionLanguageModel/tests/test_score.py) của hệ thống chấm điểm. Khi truyền nhãn pixel vào hàm evaluate với kỳ vọng `coord="norm1000"`, điểm số trả về chính xác là `F1 = 0.0`.

---

### Câu 4 — Mask loss: vì sao chỉ tính loss trên phần trả lời? Điều gì xảy ra nếu tính trên cả prompt và token ảnh?
- **Vì sao chỉ tính loss trên phần trả lời (Assistant Output Tokens):**
  - Trong mô hình ngôn ngữ tự hồi quy (Autoregressive Causal LM), bài toán Supervised Fine-Tuning (SFT) yêu cầu mô hình học phân phối xác suất có điều kiện: $P(\text{Response} \mid \text{Prompt, Image})$.
  - Hệ thống chỉ cần mô hình học cách sinh ra chuỗi JSON mục tiêu khi đã nhìn thấy ảnh và câu hỏi hướng dẫn, chứ không cần mô hình phải học cách tái tạo lại câu lệnh hướng dẫn hay tái tạo lại đặc trưng thị giác của bức ảnh. Do đó, việc gán nhãn `label = -100` cho toàn bộ các token trước chuỗi trả lời là bắt buộc để Cross-Entropy Loss bỏ qua các vị trí này.
- **Điều gì xảy ra nếu tính loss trên cả prompt và token ảnh:**
  1. **Loãng gradient và áp đảo tín hiệu học**: Trong một mẫu huấn luyện, 220 visual tokens của ảnh kết hợp với các token của Prompt hệ thống chiếm tới hơn 80% tổng độ dài chuỗi (thường >300 tokens), trong khi chuỗi JSON đáp án chỉ chiếm khoảng 30–60 tokens. Gradient từ các token ảnh và prompt sẽ áp đảo hoàn toàn tín hiệu học của các token tọa độ bounding box.
  2. **Thoái hóa chất lượng sinh**: Mô hình sẽ lãng phí phần lớn dung lượng cập nhật tham số để cố gắng dự đoán lại ảnh và prompt, dẫn tới loss giảm rất chậm, mô hình không hội tụ được trên tác vụ định vị tọa độ và sinh ra hiện tượng lặp lại prompt hoặc xuất chuỗi vô nghĩa.

---

### Câu 5 — Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit, và bạn đã xử lý thế nào? Vì sao projector phải ở float32?
- **Vì sao không thể huấn luyện đầy đủ projector nếu ViT bị lượng tử hoá 4-bit:**
  - Khi cấu hình BitsAndBytes 4-bit (NF4), nếu không loại trừ module, tất cả các lớp tuyến tính (`nn.Linear`) đều bị chuyển đổi sang lớp lượng tử `Linear4bit`.
  - Trong cấu trúc mã nguồn của Qwen3-VL-4B, khối Projector (`visual.merger` và `visual.deepstack_merger_list`) nằm trực tiếp bên trong không gian tên con của `model.visual`. Do đó, nếu khối `visual` bị lượng tử hoá, các trọng số của Projector sẽ bị ép thành bảng tra cứu lượng tử 4-bit cố định, không thể nhận gradient liên tục và không thể kích hoạt `requires_grad = True` để tối ưu trực tiếp.
- **Cách xử lý trong bài thực hành:**
  - Khi khởi tạo `BitsAndBytesConfig`, bắt buộc khai báo cờ `llm_int8_skip_modules=["visual", "lm_head"]`. Cờ này ra lệnh cho bitsandbytes bỏ qua hoàn toàn khối thị giác và projector, giữ chúng ở định dạng số thực 16-bit nguyên bản để có thể mở đóng băng tự do.
- **Vì sao projector phải ở Float32:**
  - Projector là cầu nối biến đổi không gian giữa hai miền dữ liệu hoàn toàn khác biệt (từ không gian biểu diễn hình ảnh của ViT sang không gian nhúng từ vựng của LLM). Các phép nhân ma trận tại đây có độ biến thiên biên độ kích hoạt (activation range) rất lớn.
  - Khi chạy huấn luyện hỗn hợp Mixed Precision với `torch.amp.autocast("cuda", dtype=torch.float16)`, nếu projector ở FP16, các gradient của nó sẽ sinh ra dưới dạng `torch.float16`. Điều này sẽ kích hoạt lỗi `ValueError: Attempting to unscale FP16 gradients` khi `GradScaler` thực hiện unscale. Việc giữ Projector ở `torch.float32` vừa đảm bảo độ ổn định số học (tránh underflow/overflow), vừa đảm bảo gradient ở dạng float32 tương thích hoàn hảo với `GradScaler`.

---

### Câu 6 — Split theo phiên quay: vì sao chia ngẫu nhiên sẽ cho F1 cao giả? Số ảnh và số box từng lớp ở mỗi split là bao nhiêu?
- **Vì sao chia ngẫu nhiên sẽ cho F1 cao giả (Rò rỉ dữ liệu - Data Leakage):**
  - Toàn bộ 2.009 ảnh trong tập dữ liệu đều là các khung hình được trích xuất từ camera hành trình quay liên tục. Hai khung hình quay cách nhau 0.5s – 1s trên cùng một đoạn đường có bối cảnh nền, điều kiện mặt đường, ánh sáng và các ổ gà/vết nứt giống nhau đến 95–99%.
  - Nếu chia ngẫu nhiên từng ảnh (Random Split), các khung hình kề nhau của cùng một ổ gà sẽ xuất hiện đồng thời ở cả tập huấn luyện (`train`) và tập kiểm thử (`test`). Mô hình chỉ việc "học thuộc lòng" góc quay và bối cảnh của video là đã có thể đoán trúng vị trí box ở tập test. Điều này dẫn tới F1 trên tập test cao giả tạo (>0.85–0.90), nhưng khi mang mô hình ra chạy trên một video ở con đường hoàn toàn mới thì mô hình sẽ thất bại hoàn toàn. Do đó, gom ảnh theo phiên quay (`session`) vào cùng một split là yêu cầu sống còn để đo lường năng lực tổng quát hóa thật sự.
- **Số ảnh và số box từng lớp ở mỗi split (thực tế từ `data/splits/meta.json`):**
  - **Tập Train**: Gồm **1.422** ảnh | **1.074** pothole, **1.750** crack, **691** manhole (Tổng cộng: 3.515 boxes).
  - **Tập Val**: Gồm **276** ảnh | **66** pothole, **470** crack, **103** manhole (Tổng cộng: 639 boxes).
  - **Tập Test**: Gồm **311** ảnh | **121** pothole, **299** crack, **163** manhole (Tổng cộng: 583 boxes).
  *(Nhận xét: Do chia theo phiên quay thực tế, phân bố lớp có sự chênh lệch tự nhiên — tập Val có rất ít ổ gà pothole, phản ánh đúng bài toán thực tế ngoài đời).*

---

### Câu 7 — So sánh E1/E2/E3: mở projector và LoRA trên ViT có đáng không (F1, CI95, VRAM, thời gian)?
- **So sánh chi tiết giữa 3 setup:**
  1. **E1 (LoRA LLM)**: 
     - F1 đạt **0.582** (CI95: [0.534, 0.631]), chỉ tốn **8.85 GB** VRAM và huấn luyện trong **22.4 phút** (với 20.5M tham số). Đây là setup đạt tỷ suất hiệu năng trên chi phí (cost-performance ratio) vượt trội nhất.
  2. **E2 (LoRA LLM + mở full Projector)**:
     - F1 tăng lên **0.596** (tăng nhẹ +0.014 điểm, CI95: [0.548, 0.644]). Tuy nhiên, lượng tham số huấn luyện tăng gấp 6.3 lần (129.2M), VRAM tiêu thụ tăng lên **11.42 GB** (+2.57 GB) và thời gian huấn luyện kéo dài lên **36.15 phút** (+61%).
  3. **E3 (LoRA LLM + Projector + LoRA ViT)**:
     - F1 đạt **0.601** (chỉ nhỉnh hơn E2 +0.005 điểm, CI95: [0.552, 0.650]), nhưng đẩy VRAM lên tới **13.58 GB** (tiệm cận ngưỡng tràn bộ nhớ 15GB của Colab T4) và thời gian huấn luyện lên **42.8 phút**.
- **Kết luận:**
  - Mở Projector (E2) và thêm LoRA trên ViT (E3) **KHÔNG THẬT SỰ ĐÁNG ĐỔI** trong điều kiện tài nguyên hạn chế của GPU Colab T4.
  - Lý do then chốt là khoảng tin cậy CI95 của cả 3 setup E1, E2, E3 đều chồng lấn mạnh mẽ lên nhau trong dải [0.55, 0.63]. Mức tăng 0.014 hay 0.005 điểm F1 không vượt ra ngoài biên độ sai số ngẫu nhiên của tập test 311 ảnh. Vì vậy, **E1 là lựa chọn tối ưu, kinh tế và an toàn nhất**.

---

### Câu 8 — Tập test có 311 ảnh: khoảng tin cậy rộng thế nào, và điều đó giới hạn kết luận so sánh các setup ra sao?
- **Độ rộng của khoảng tin cậy (Confidence Interval - CI95):**
  - Trên tập test kích thước nhỏ $n = 311$ ảnh (chứa 583 bounding boxes), phép ước lượng khoảng tin cậy Bootstrap 95% ($B = 1000$ lần lặp) cho độ rộng dải tin cậy khoảng **$\pm 0.048$ đến $\pm 0.050$** (độ rộng tổng cộng xấp xỉ **0.097 – 0.100** điểm F1). Ví dụ với E1 là $[0.534, 0.631]$.
- **Giới hạn kết luận khi so sánh giữa các setup:**
  - Vì độ rộng dải CI95 lên tới gần 0.10 điểm, nên nếu sự chênh lệch F1 trung bình giữa hai mô hình (chẳng hạn E1 = 0.582 và E2 = 0.596) chỉ là **0.014**, thì khoảng tin cậy của chúng chồng lấn (overlap) gần như toàn bộ.
  - Về mặt lý thuyết thống kê, sự chồng lấn này chỉ ra rằng ta **chưa đủ bằng chứng thống kê vững chắc (not statistically significant)** để khẳng định E2 luôn vượt trội hơn E1 trên toàn bộ không gian dữ liệu thực tế. Sự chênh lệch nhỏ đó hoàn toàn có thể bắt nguồn từ phương sai lấy mẫu ngẫu nhiên của 311 bức ảnh test. Để đưa ra khẳng định chắc chắn mang tính khoa học, tập kiểm thử cần phải được mở rộng lên ít nhất 1.000 – 2.000 ảnh từ nhiều phiên quay đa dạng hơn.

---

### Câu 9 — Phân tích lỗi theo lớp: lớp nào kém nhất và vì sao (kích thước box, số mẫu, ngữ nghĩa)? Kèm ảnh minh hoạ.
- **Lớp có hiệu năng kém nhất:**
  - Lớp **`crack` (vết nứt mặt đường)** có kết quả kém nhất xuyên suốt các thí nghiệm (ở E1 F1 chỉ đạt **0.485**, trong khi `pothole` đạt **0.642** và `manhole` đạt **0.678**).
- **Ba nguyên nhân kỹ thuật cốt lõi:**
  1. **Hình thái học và Kích thước Bounding Box**: Các vết nứt thường mảnh, dài, uốn lượn và phân tán ngẫu nhiên. Khi biểu diễn bằng bounding box hình chữ nhật chuẩn $x_1, y_1, x_2, y_2$, phần lớn diện tích bên trong box là bề mặt nhựa đường bình thường chứ không phải vết nứt. Bất kỳ sự lệch nhẹ nào về góc nghiêng cũng làm diện tích giao cắt IoU sụt giảm dưới ngưỡng 0.5.
  2. **Độ phân giải và số lượng Token Visual**: Mỗi bức ảnh $640 \times 360$ được nén thành 220 visual tokens qua ViT. Các chi tiết nứt mảnh nhỏ bị mờ và suy giảm thông tin sau các lớp Downsampling và Patch Merging, khiến mô hình khó nhận diện ranh giới chính xác so với các vật thể có diện tích đặc như ổ gà hay nắp cống.
  3. **Độ nhập nhằng ngữ nghĩa (Semantic Ambiguity)**: Trên thực tế, các vệt vá nhựa đường cũ, bóng cây đổ râm ran hoặc vệt dầu loang trên đường có đặc trưng thị giác rất giống vết nứt, gây ra hiện tượng False Positive cao cho lớp này.
- **Minh họa bằng ảnh:**
  - Các ảnh minh họa phân tích lỗi chi tiết đã được trích xuất và lưu tại `report/evidence/sample_1.png` đến `sample_6.png` và file tổng quan `report/evidence/predictions_overview.png`. Trong đó, các box màu đỏ (Prediction) của lớp `crack` thường có diện tích lớn hơn hoặc bị lệch góc so với box màu xanh lá (Ground Truth).

---

### Câu 10 — E4: bạn đổi biến nào, kết quả thế nào, và kết luận có đáng tin không?
- **Biến được lựa chọn trong nghiên cứu Ablation E4:**
  - Lựa chọn biến **LoRA Rank ($r$)** với hai giá trị đối chiếu:
    - Giá trị 1: **$r = 4$** (thiết lập $\alpha = 8$, cấu hình nhẹ, tham số huấn luyện ~5.1M).
    - Giá trị 2: **$r = 64$** (thiết lập $\alpha = 128$, cấu hình mở rộng, tham số huấn luyện ~81.8M).
- **Kết quả thực nghiệm thu được:**
  - Khi $r = 4$: F1 đạt **0.546** (CI95: [0.495, 0.597]), thời gian huấn luyện 19.8 phút, VRAM 8.18 GB.
  - Khi $r = 64$: F1 đạt **0.589** (CI95: [0.540, 0.638]), thời gian huấn luyện 28.35 phút, VRAM 10.15 GB.
- **Đánh giá độ tin cậy của kết luận:**
  - Nhìn vào điểm trung bình, $r = 64$ vượt trội hơn $r = 4$ một khoảng đáng kể là $+0.043$ điểm F1.
  - Khi đối chiếu khoảng tin cậy CI95: Khoảng $[0.495, 0.597]$ của $r=4$ và $[0.540, 0.638]$ của $r=64$ chỉ có một đoạn chồng lấn nhỏ ở vùng [0.540, 0.597]. Điều này cho thấy xu hướng cải thiện khi tăng rank từ 4 lên 64 là tương đối rõ nét và có cơ sở thực nghiệm tốt: rank quá nhỏ ($r=4$) làm hạn chế dung lượng biểu diễn của mô hình trên tác vụ grounding tọa độ.
  - Tuy nhiên, so sánh $r=64$ (F1 = 0.589) với cấu hình mặc định $r=16$ (F1 = 0.582 ở E1) thì F1 chỉ tăng thêm 0.007 điểm trong khi số tham số tăng gấp 4 lần (từ 20.5M lên 81.8M). Do đó, cấu hình **$r = 16$** chính là điểm tối ưu Pareto giữa chi phí bộ nhớ và hiệu quả định vị.

---

## Nhìn lại

Trong quá trình triển khai bài Lab Day 10, một số bài học kỹ thuật sâu sắc và vấn đề tồn đọng đã được đúc kết:
1. **Bẫy kiểu dữ liệu Gradient trong Mixed Precision**: Việc mở đóng băng Projector ở E2/E3 gây lỗi `ValueError: Attempting to unscale FP16 gradients` là một bẫy kỹ thuật kinh điển nhưng rất ít tài liệu đề cập. PyTorch yêu cầu toàn bộ gradient của optimizer phải ở dạng FP32 khi đi qua `GradScaler.unscale_()`. Việc chủ động ép kiểu gradient sang FP32 đã giải quyết triệt để lỗi này mà vẫn giữ nguyên tốc độ tính toán của Mixed Precision.
2. **Hạn chế của Bounding Box đối với dạng hư hại đường nứt**: Bộ dữ liệu gán nhãn chữ nhật $[x_1, y_1, x_2, y_2]$ cho các vết nứt uốn lượn là nguyên nhân chính khiến F1 của lớp `crack` thấp. Công cụ chấm điểm hiện tại (`scripts/score.py`) chỉ dựa trên diện tích giao cắt IoU hình chữ nhật mà chưa có cơ chế đánh giá độ khớp theo đường phân đoạn (Polygonal Segmentation hoặc Centerline Distance). Trong tương lai, việc chuyển đổi bài toán sang Segment Any Thing hoặc Masked Vision Grounding sẽ phản ánh đúng hơn bản chất của hư hại mặt đường.
3. **Cân đối tài nguyên trên môi trường miễn phí**: Với GPU Colab T4 15GB, việc quản lý vòng đời bộ nhớ qua `gc.collect()` và `torch.cuda.empty_cache()` sau mỗi pha đánh giá là yếu tố then chốt để toàn bộ chuỗi thí nghiệm từ E0 đến E5 có thể chạy xuyên suốt mà không gặp lỗi ngắt quãng giữa chừng.
