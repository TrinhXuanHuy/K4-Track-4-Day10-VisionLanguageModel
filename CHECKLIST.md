# CHECKLIST TIẾN ĐỘ VÀ ĐÓNG GÓI BÀI NỘP LAB DAY 10

> **Dự án**: Finetune Qwen3-VL-4B (LoRA) để định vị hư hại mặt đường  
> **Học viên**: Trịnh Xuân Huy  
> **Repository**: [K4-Track-4-Day10-VisionLanguageModel](https://github.com/TrinhXuanHuy/K4-Track-4-Day10-VisionLanguageModel)

---

## 1. Cấu trúc thư mục hồ sơ bài nộp chuẩn

Hồ sơ nộp trên GitHub bắt buộc phải có đầy đủ cấu trúc sau:

```text
K4-Track-4-Day10-VisionLanguageModel/
├── report/
│   └── evidence/
│       ├── e0.json                  # Kết quả E0 Zero-shot
│       ├── e1.json                  # Kết quả E1 LoRA LLM
│       ├── e2.json                  # Kết quả E2 LoRA LLM + Projector
│       ├── e3.json                  # Kết quả E3 LoRA LLM + Projector + LoRA ViT
│       ├── e4_rank_4.json           # Kết quả E4 (rank r=4)
│       ├── e4_rank_64.json          # Kết quả E4 (rank r=64)
│       ├── e5.json                  # (Tuỳ chọn) Kết quả E5 GRPO Bonus
│       ├── sample_1.png ... 6.png   # Ít nhất 6 ảnh trực quan hoá kết quả
│       └── predictions_overview.png # Ảnh tổng quan so sánh GT vs PRED
├── REPORT.md                        # Đã điền đủ bảng và 10 câu hỏi bằng số liệu thật
├── lab_day10.ipynb                  # Mã nguồn notebook hoàn chỉnh
├── CHECKLIST.md                     # Danh mục kiểm tra tiến độ
├── BAO_CAO_LOI_VA_LUU_Y.md         # Báo cáo các bẫy kỹ thuật & debug
├── configs/                         # GIỮ NGUYÊN BẢN GỐC (e0.yaml ... e5.yaml)
├── scripts/                         # GIỮ NGUYÊN BẢN GỐC (Không sửa)
└── tests/                           # GIỮ NGUYÊN BẢN GỐC (Không sửa)
```

---

## 2. Danh mục rà soát trước khi đẩy code lên GitHub (5/5 mục)

- [x] **Mục 1**: File `REPORT.md` đã điền đầy đủ họ tên, `dataset_version: d10-21e4070301`, bảng tổng hợp số liệu và trả lời đầy đủ 10 câu hỏi bằng số liệu thực nghiệm.
- [x] **Mục 2**: Tuyệt đối **không còn bất kỳ chuỗi `<điền>` nào** trong file `REPORT.md` (đảm bảo script kiểm tra tự động `check_submission.py` thông qua).
- [x] **Mục 3**: Đã đính kèm ít nhất **6 ảnh minh họa** trực quan hoá bounding box dự đoán (màu đỏ) và bounding box đúng (màu xanh lá) tại `report/evidence/`.
- [x] **Mục 4**: Lệnh `git status` xác nhận không có file cấm nào bị track:
  - [x] Không có dữ liệu `data/`
  - [x] Không có thư mục checkpoint `work/`
  - [x] Không có file cấu hình ẩn `.env` hoặc file khóa `kaggle.json`
  - [x] Không có file trọng số lớn `*.safetensors`, `*.pt`, `*.bin`, `*.ckpt`
- [x] **Mục 5**: Lệnh `git rev-list --count HEAD` trả về kết quả **≥ 5 commits**, thể hiện rõ lịch sử làm bài theo từng giai đoạn.

---

## 3. Các bước đẩy mã nguồn lên GitHub và nộp bài trên LMS

### Bước 1: Kiểm tra bài nộp bằng công cụ của đề
Trước khi đẩy code, chạy lệnh kiểm tra tự động:
```bash
python scripts/check_submission.py
```
*Yêu cầu*: Phải hiển thị `0 FAIL` (tất cả các mục đều `[PASS]`).

### Bước 2: Commit và Push mã nguồn
```bash
git add -A
git commit -m "Hoàn thành Lab Day 10 — Trịnh Xuân Huy"
git push origin main
```

### Bước 3: Đảm bảo Repository ở trạng thái Public
1. Truy cập vào trang GitHub cá nhân: `https://github.com/TrinhXuanHuy/K4-Track-4-Day10-VisionLanguageModel/settings`
2. Kéo xuống mục **Danger Zone** -> **Change repository visibility** -> Chuyển thành **Public**.
*(Lưu ý: Repository để Private sẽ khiến hệ thống chấm tự động chấm 0 điểm).*

### Bước 4: Nộp bài trên hệ thống VinUni LMS
1. Sao chép liên kết public GitHub:
   ```text
   https://github.com/TrinhXuanHuy/K4-Track-4-Day10-VisionLanguageModel
   ```
2. Dán link vào ô nộp bài Day 10 trên hệ thống VinUni LMS.
3. Giữ repository ở trạng thái Public cho đến khi có điểm số chính thức.

---

## 4. Ôn tập nội dung vấn đáp 5 phút

Để sẵn sàng cho phần phỏng vấn vấn đáp 5 phút với giảng viên, hãy nắm vững các câu hỏi trọng tâm sau:
1. **Luồng dữ liệu trong mô hình**:
   - `Ảnh input (640×360)` → `ViT` (trích xuất 220 visual tokens) → `Projector` (merger + deepstack merger chuyển sang 2048 chiều embedding) → Ghép vào `LLM` (autoregressive generation) → Output chuỗi JSON tọa độ.
2. **Cơ chế Mask Loss**:
   - Chỉ tính loss trên các token của câu trả lời (`assistant`), gán loss = `-100` cho toàn bộ prompt và visual tokens để không làm loãng gradient.
3. **Lý do mở Projector ở Float32**:
   - Tránh lỗi tràn số/underflow dưới FP16 mixed precision và ngăn chặn lỗi `Attempting to unscale FP16 gradients` khi backward với `GradScaler`.
4. **Chia dữ liệu theo Session**:
   - Gom ảnh theo phiên quay video (ngày + giờ) để tránh rò rỉ dữ liệu (data leakage) giữa các frame kề nhau, ngăn chặn hiện tượng F1 cao giả tạo.
5. **Đọc khoảng tin cậy (Confidence Interval - CI95)**:
   - Tập test 311 ảnh cho khoảng tin cậy rộng (~$\pm 0.05$). Nếu CI95 của hai model chồng lấn thì chưa thể kết luận chắc chắn mô hình nào vượt trội về mặt thống kê.

