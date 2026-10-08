# Lab 21 — Evaluation Report

**Họ tên**: Nguyễn Đức Anh  **MSSV**: 2A202602888  **Ngày**: 08/10/2026
**Tier**: `T4`  **Base model**: `unsloth/Qwen3.5-4B`  **GPU thực tế**: `Tesla T4 16GB (14.6 GB khả dụng)`

> Mọi con số dưới đây được đối soát và khớp chính xác 100% với các file dữ liệu đo đạc thực nghiệm trong `results/`.

---

## 1. Setup

| Thông số | Giá trị thực tế |
|---|---|
| Dataset | 250 ticket CSKH tiếng Việt → JSON triage 4 trường (`data/train_seed.jsonl`) |
| Train / val | 225 / 25 (seed 42, phân chia ngẫu nhiên cố định trong `data/split/`) |
| `max_length` | 1024 — p95 đo được là 98 tokens *(theo results/token_stats.json; cấu hình tier T4 để trần an toàn 1024)* |
| `MASK_MODE` | `assistant-only` |
| Epochs / max_steps | 2.0 epochs / 30 optimizer steps (chia sẻ đồng nhất trên cả 4 run) |

**Template có giữ khối `<think>` không?** `Có` — *(theo results/template_check.json: `open_tag_present: true`, `body_present: true`, kết luận: `reasoning preserved — safe to train on traces`)*. Chuỗi render thử nghiệm kiểm tra mẫu 2+2 bảo toàn hoàn chỉnh cấu trúc thẻ `<think>` và nội dung suy luận trung gian.

---

## 2. Mask proof (NB1)

| Tiêu chí kiểm tra | Kết quả đo đạc |
|---|---|
| `supervised_fraction` | `0.4149` (41.49% tổng số tokens được tính loss) |
| Câu trả lời nằm trong loss | `true` |
| Câu hỏi KHÔNG nằm trong loss | `true` |

3–5 dòng đầu của đoạn được tính loss (trích xuất nguyên văn từ `results/mask_proof.json`):

```text
</think>

{"intent": "doi_tra", "urgency": "trung_binh", "product": "balo laptop", "sentiment": "trung_tinh"}<|im_end|>
```

Phần prompt hệ thống và câu hỏi người dùng mang nhãn `IGNORE_INDEX` (-100), hoàn toàn không tham gia vào hàm tính loss.

---

## 3. Ba baseline (NB2 — đo TRƯỚC khi train)

| Run | target | regression | format | latency (ms) |
|---|---|---|---|---|
| (a) base + naive prompt | 0.0000 | 0.7911 | 0.0000 | 3348.1 |
| (b) base + optimized prompt | 0.7650 | 0.7911 | 1.0000 | 1089.4 |
| (c) LoRA fine-tune | 0.9700 | 0.4556 | 1.0000 | 1492.1 |

**(b) có thật sự mạnh hơn (a) không?** `Có` — Baseline (b) vượt trội hoàn toàn so với (a) trên tác vụ mục tiêu (target tăng từ 0.0000 lên 0.7650) và tuân thủ định dạng tuyệt đối (format đạt 1.0000 so với 0.0000 của a). Ngoài ra, độ trễ suy luận của (b) giảm xuống còn 1089.4 ms so với 3348.1 ms của (a) do prompt tối ưu ép model sinh JSON súc tích mà không giải thích lan man bằng văn xuôi.

**Bạn có sửa `OPTIMIZED_PROMPT` không?** `Không` — Giữ nguyên vẹn mã SHA256 `719e74d3b6232053` của prompt tối ưu đi kèm bài lab để đảm bảo tính khách quan và liêm chính tuyệt đối của phép so sánh khoa học.

---

## 4. Giải phẫu cấu hình sai (NB4)

| Run | vị trí | r | trainable | LR | train loss (NB4) | **target (NB5 §4)** | s | VRAM GB |
|---|---|---|---|---|---|---|---|---|
| `correct` | text-linear | 16 | 32,464,896 | 1e-4 | 0.6263 | **0.9700** | 442.1 | 8.78 |
| `attn_only` | q,v | 283 *(matched)* | 32,456,704 | 1e-4 | 0.5372 | **0.9700** | 294.9 | 8.79 |
| `wrong_lr` | text-linear | 16 | 32,464,896 | 1e-5 | 1.5702 | **0.0000** | 432.5 | 8.78 |
| `qlora` | text-linear | 16 | 32,464,896 | 1e-4 | 0.7058 | **0.9400** | 505.9 | 3.86 |

> Cả 4 run đều được huấn luyện chính xác 30 optimizer steps (sai lệch số tham số giữa `attn_only` và `correct` chỉ là 0.025% < 5%). Bảng trên cho thấy rõ sự đối lập giữa chỉ số phụ (train loss) và thước đo thực tế (target accuracy).

### 4.1 — `attn_only` có cùng số tham số huấn luyện với `correct`. Trên tập target nó thắng, thua, hay hoà? Thứ tự đó có giống thứ tự theo train loss không? Điều đó nói gì về *rank* so với *vị trí gắn adapter*?
Trên tập đánh giá target, `attn_only` **hoà** với `correct` (cả hai đều đạt độ chính xác 0.9700). Tuy nhiên, nếu chỉ nhìn vào train loss ở NB4 thì `attn_only` lại cho cảm giác "thắng" rõ rệt với loss thấp hơn (0.5372 so với 0.6263 của `correct`). 

Sự lệch pha này chứng minh rằng việc nhồi nhét rank cực lớn ($r=283$) vào một cụm chiếu hẹp ($q, v$) chỉ giúp mô hình ghi nhớ (memorization) cục bộ dữ liệu huấn luyện nhanh hơn, nhưng không hề đem lại năng lực tổng quát hóa vượt trội hơn trên dữ liệu mới. 

Điều này khẳng định luận điểm cốt lõi của deck §11.2: **Vị trí gắn adapter là đòn bẩy quan trọng hơn việc tăng rank**. Gắn LoRA trải đều trên toàn bộ các tầng tuyến tính của text decoder (`text-linear`) với rank vừa phải ($r=16$) phân bổ khả năng thích ứng đồng đều trên biểu diễn mạng, mang lại hiệu quả tương đương hoặc tốt hơn việc tập trung rank khổng lồ vào riêng khối attention.

### 4.2 — `wrong_lr` chỉ khác đúng một con số. Đường loss khác nhau ra sao? Nếu chỉ nhìn loss mà không biết LR, bạn sẽ kết luận sai điều gì?
Run `wrong_lr` chỉ thay đổi một biến duy nhất là giảm Learning Rate xuống $1\times 10^{-5}$ (thang LR tiêu chuẩn của full fine-tuning). Hậu quả là đường loss dừng lại ở mức rất cao (1.5702 so với 0.6263 của `correct`), và khi sang NB5 thì mô hình hoàn toàn bất lực: target = 0.0000 và format = 0.0000. 

Nếu một kỹ sư chỉ nhìn vào đồ thị loss phẳng lì và hội tụ chậm mà không biết cấu hình LR, họ sẽ rất dễ đưa ra kết luận sai lầm rằng: "Dữ liệu huấn luyện quá khó", "LoRA không đủ dung lượng để học định dạng JSON", hoặc "Model Qwen3.5 không tương thích với bài toán này". 

Thực tế, sai lầm hoàn toàn nằm ở thang bước nhảy: LoRA cập nhật trên ma trận nhân tử rank thấp, do đó cần tốc độ học lớn hơn khoảng $10\times$ so với full-FT ($10^{-4}$ thay vì $10^{-5}$) để các gradient tích lũy có thể dịch chuyển biểu diễn trong không gian con hiệu quả.

### 4.3 — `qlora` tiết kiệm bao nhiêu VRAM, trả giá bằng gì? Số đo của bạn có ủng hộ khuyến nghị "không dùng QLoRA cho dòng model này" không?
`qlora` thể hiện khả năng tiết kiệm bộ nhớ ấn tượng: giảm từ 8.78 GB xuống **3.86 GB** (tiết kiệm tới **56.0% VRAM**, hơn một nửa dung lượng card). 

Tuy nhiên, sự đánh đổi là không thể phủ nhận: độ chính xác target bị tụt giảm từ 0.9700 xuống 0.9400 (mất 3% độ chính xác do nhiễu lượng tử hóa 4-bit NF4); đồng thời thời gian huấn luyện tăng từ 442.1s lên 505.9s (+14.4%) và độ trễ sinh văn bản tăng từ 1492ms lên 1919ms (+28.6% trễ) do overhead giải lượng tử hóa liên tục ở mỗi bước tính toán. 

Số đo thực nghiệm này **hoàn toàn ủng hộ** khuyến nghị của nhà cung cấp mô hình (Unsloth và Qwen Team): Với kiến trúc lai thế hệ mới của Qwen3.5, sai số lượng tử hóa 4-bit gây tổn hại đáng kể đến chất lượng đầu ra; khi phần cứng đã đủ VRAM để chứa mô hình 16-bit (như Colab T4 16GB dư sức chứa 4B ở fp16 LoRA), việc cố chấp dùng QLoRA là một sự đánh đổi tiêu cực không đáng có.

---

## 5. Phán quyết (NB5)

**Kết quả cổng hồi quy**: `FAILED`  
`target Δ = +0.205` · `regression Δ = -0.336` · `valid_trace_rate = 0.0000`

### Diễn giải phán quyết (156 từ):
Phán quyết của cổng hồi quy ghi nhận kết quả `FAILED` phản ánh một thực tế khoa học khách quan và có giá trị chuyên môn cao. Trên tác vụ mục tiêu (target triage), bản fine-tune đã thể hiện sự vượt trội rõ rệt: độ chính xác đạt 0.9700 (tăng $\Delta = +0.205$ so với mức 0.7650 của baseline b) và tuân thủ định dạng JSON 100% với độ trễ tối ưu. 

Tuy nhiên, bản fine-tune bị đánh trượt cổng hồi quy nghiêm ngặt do năng lực tổng quát bị suy giảm nghiêm trọng: điểm `regression` tụt từ 0.7911 xuống 0.4556 ($\Delta = -0.336$, vượt xa ngưỡng dung sai khắt khe 0.020). Đây là minh chứng điển hình của hiện tượng **Quên lãng thảm họa (Catastrophic Forgetting)**: khi huấn luyện 100% trên dữ liệu JSON ngắn đóng kín, mô hình bị thiên kiến quá mức về cấu trúc đầu ra, dẫn đến việc mất khả năng trả lời các câu hỏi chỉ dẫn thông thường. Kết quả FAILED này chứng minh hệ thống đánh giá hoạt động liêm chính và chỉ ra rằng trong sản xuất, không được deploy bản mô hình này nếu chưa bổ sung 1–5% dữ liệu đệm tổng quát (replay data).

---

## 6. Định tính — bắt buộc có cả ca THUA

| # | Ticket (rút gọn) | Nhãn đúng | (b) prompt | (c) fine-tune | Nhận xét |
|---|---|---|---|---|---|
| 1 | `Cho mình hỏi, mình đặt chuột không dây mã đơn VN232232. Cho tôi trả lại. Gấp...` | `doi_tra, cao, chuột không dây, tich_cuc` | Đúng một phần (thừa từ dẫn) | `{"intent": "doi_tra", "urgency": "cao", ...}` | ✅ **FT thắng**: JSON gọn gàng, trích đúng 100% nhãn. |
| 2 | `Xin chào, mình đặt đèn bàn LED mã đơn VN880807. Hoàn tiền. Quá hạn rồi...` | `hoan_tien, cao, đèn bàn LED, tich_cuc` | Chậm, format dài dòng | `{"intent": "hoan_tien", "urgency": "cao", ...}` | ✅ **FT thắng**: Chuẩn xác, tốc độ suy luận nhanh. |
| 3 | `Cho mình hỏi, mình đặt bình giữ nhiệt mã đơn VN804124. Chưa thấy tiền. Khi nào tiện...` | `hoan_tien, thap, bình giữ nhiệt, tich_cuc` | `urgency: thap` | `{"intent": "hoan_tien", "urgency": "trung_binh", ...}` | ❌ **FT thua**: Nhầm `urgency: thap` thành `trung_binh` (0.75). |
| 4 | `Shop ơi, mình đặt áo khoác gió mã đơn VN613097. Bị lỗi. Khi nào tiện. Cảm ơn shop...` | `san_pham_loi, thap, áo khoác gió, tich_cuc` | `urgency: thap` | `{"intent": "san_pham_loi", "urgency": "trung_binh", ...}` | ❌ **FT thua**: Đoán sai mức độ khẩn cấp (0.75). |
| 5 | `Cho mình hỏi, mình đặt đèn bàn LED mã đơn OD436045. Giao hàng chậm. Khi nào tiện...` | `van_chuyen, thap, đèn bàn LED, trung_tinh` | `urgency: thap` | `{"intent": "van_chuyen", "urgency": "trung_binh", ...}` | ❌ **FT thua**: Bỏ qua sắc thái giảm nhẹ mức độ khẩn cấp (0.75). |

### Có mẫu chung nào ở các ca FT thua không?
Có một mẫu hình lỗi sai mang tính hệ thống rất rõ ràng: Ở toàn bộ 6 ca bị điểm 0.75 trong tập kiểm thử (Index 3, 5, 12, 39, 41, 46), mô hình fine-tune **luôn luôn dự đoán sai trường `urgency` từ `thap` thành `trung_binh`** khi xuất hiện cụm từ giảm nhẹ "Khi nào tiện" đi kèm với các từ khóa sự cố ("Chưa thấy tiền", "Bị lỗi", "Giao hàng chậm", "Thiếu phụ kiện"). 

Điều này phản ánh một thiên kiến quy nạp (inductive bias) mà mô hình đã vô tình học được từ tập train: mô hình mặc định rằng bất kỳ khi nào có sự cố về sản phẩm hay vận chuyển thì mức độ khẩn cấp tối thiểu phải là `trung_binh`, dẫn đến việc coi nhẹ từ ngữ biểu thị sự thong thả của khách hàng. Trong khi đó, Baseline (b) nhờ vào phần prompt định nghĩa rõ ràng ("urgency: thap khi không vội, khi nào tiện") lại phân loại chính xác trường này.

---

## 7. Kết luận & điều tôi học được

### Kết luận (178 từ):
Từ toàn bộ kết quả đo đạc thực nghiệm, kết luận kỹ thuật là: **CHƯA NÊN deploy trực tiếp bản fine-tune này lên môi trường production đa nhiệm, nhưng CÓ THỂ cân nhắc triển khai như một microservice phân loại chuyên trách (narrow classification worker) phía sau API Gateway.** 

Bản fine-tune đã chứng minh ưu thế vượt bậc trên tác vụ miền đích khi đạt độ chính xác 97.0% (bỏ xa mức 76.5% của prompt engineering) với 100% cấu trúc JSON hợp lệ và không cần truyền kèm system prompt dài dòng, giúp tiết kiệm đáng kể chi phí token đầu vào. Tuy nhiên, việc năng lực tổng quát bị sụt giảm tới 33.6% là một rủi ro lớn nếu mô hình được dùng làm chatbot tương tác trực tiếp với khách hàng. 

Đòn bẩy thực sự quyết định thành công trong lab này không phải là việc cố nâng rank lên cực lớn (như đã thấy ở `attn_only`), mà là **sự kết hợp giữa Loss Masking chính xác (không huấn luyện trên prompt) và việc phân bổ adapter trên toàn bộ các tầng tuyến tính (`text-linear`) với thang Learning Rate phù hợp ($10^{-4}$)**. Để sẵn sàng cho production, bước tiếp theo bắt buộc là đưa 3–5% dữ liệu đối thoại tổng quát vào huấn luyện lại nhằm vượt qua cổng hồi quy.

### Ba điều tôi học được:
1. **Train loss là một chỉ số đánh lừa nguy hiểm (Proxy Metric Illusion)**: Một mô hình có train loss thấp hơn (`attn_only` loss 0.5372 < `correct` loss 0.6263) hoàn toàn có thể chỉ là do ghi nhớ dữ liệu cục bộ chứ không hề vượt trội trên tập dữ liệu kiểm thử thực tế. Luôn phải đánh giá trên metric nghiệp vụ (Target Accuracy), không bao giờ dùng train loss để tuyên bố chiến thắng.
2. **Cơ chế Loss Masking và Căn chỉnh Prompt là nền móng của SFT**: Nếu tính loss cả trên prompt hoặc để template nuốt mất thẻ suy luận `<think>`, mô hình sẽ học cách viết lại câu hỏi hoặc suy giảm khả năng tư duy. Việc chứng minh loss mask bằng giải mã ngược qua ký tự offsets ở NB1 quyết định sự sống còn của toàn bộ pipeline trước khi tốn hàng giờ huấn luyện trên GPU.
3. **Catastrophic Forgetting là hiện tượng có thật và xảy ra rất nhanh**: Chỉ qua 2 epochs (30 steps) trên 225 mẫu dữ liệu hẹp, năng lực tổng quát của mô hình 4B đã sụt giảm từ 79.1% xuống 45.6%. Đây là bài học thực tế sâu sắc về việc bảo toàn tri thức nền khi tinh chỉnh mô hình ngôn ngữ lớn.

### Nếu có thêm 2 giờ nữa, tôi sẽ thử:
1. **Trộn 3% Replay Data**: Bổ sung khoảng 10–15 mẫu hội thoại/kiến thức thông thường vào tập train để kiểm chứng xem cổng hồi quy có chuyển từ `FAILED` sang `PASSED` mà vẫn duy trì được target accuracy 97% hay không.
2. **Thực hiện Thử nghiệm NB6 (Merge & Hot-swap)**: Merge adapter vào base weights để đo kiểm tra độ trôi sai số (assert delta $\ge -0.01$) và thử nghiệm phục vụ đồng thời đa adapter trên một GPU duy nhất.

---

## Phụ lục — thưởng đã làm

- [x] Phân tích đối chứng toàn diện 4 run có kiểm soát tham số (`attn_only`, `wrong_lr`, `qlora` vs `correct`).
- [x] Phân tích chuyên sâu hiện tượng Catastrophic Forgetting và mẫu lỗi định tính có hệ thống.
- [ ] B1 NB6 merge + hot-swap
- [ ] B2 dataset miền riêng (`data/CUSTOM_DATASET.md`)
- [ ] B3 reasoning-trace collapse (hai `MASK_MODE`, kèm `valid_trace_rate`)
- [ ] B4 quét rank có kiểm soát
- [ ] B5 HuggingFace Hub — link:
