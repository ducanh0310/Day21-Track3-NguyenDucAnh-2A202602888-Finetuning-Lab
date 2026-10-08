# Reflection — Lab 21

*Ngắn gọn, thành thật. Phần này chấm theo độ cụ thể, không theo độ dài.*

**1. Điều gì làm bạn ngạc nhiên nhất?**
Điều làm tôi ngạc nhiên nhất là hiện tượng sụt giảm năng lực tổng quát (Catastrophic Forgetting) diễn ra nhanh và mạnh đến vậy: chỉ sau đúng 30 optimizer steps (2 epochs trên 225 mẫu), điểm general capability đã tụt từ 79.1% xuống 45.6%, dù mô hình học tác vụ target đạt tới 97%. Ngoài ra, việc run `attn_only` có train loss thấp hơn `correct` nhưng điểm target thực tế lại chỉ bằng nhau là một bất ngờ lớn về sự sai lệch giữa proxy metric và task metric.

**2. Bạn mất nhiều thời gian nhất ở đâu? Nó có phải chỗ bạn dự đoán không?**
Tôi mất nhiều thời gian nhất ở công đoạn chạy 4 run huấn luyện và sinh text trên GPU Colab T4 (~50 phút cho NB3-NB5). Ban đầu tôi dự đoán phần cài đặt môi trường và sửa lỗi thư viện sẽ lâu nhất, nhưng thực tế nhờ pipeline được đóng gói sẵn với `colab_run.py`, việc thiết lập lại rất nhanh; thời gian chủ yếu nằm ở chi phí tính toán forward/backward và autoregressive generation không thể đốt cháy giai đoạn.

**3. Trước lab này bạn tin điều gì về fine-tuning mà giờ bạn không còn tin?**
Trước lab này, tôi từng tin rằng: "Cứ fine-tune với loss giảm càng sâu thì mô hình càng thông minh hơn base model", và "Muốn LoRA mạnh hơn thì chỉ cần tăng rank $r$ lên thật cao". Giờ đây tôi hiểu rằng train loss thấp có thể chỉ là ghi nhớ dữ liệu cục bộ, và việc tăng rank ở vị trí hẹp ($q, v$) hoàn toàn thua kém việc phân bổ adapter trên toàn bộ các tầng (`text-linear`) với rank vừa phải.

**4. Bạn dùng AI assistant vào việc gì trong lab? Chỗ nào nó sai?**
Tôi sử dụng AI assistant để rà soát kiến trúc mã nguồn trong thư viện `labkit`, giải thích cơ chế mapping token-character offset của chat template, và phân tích các số đo thực nghiệm. Chỗ nó từng nhầm là khi phân tích git status ở local Windows: nó tưởng file bị sửa đổi nhưng thực chất chỉ do Git tự động chuyển đổi ký tự xuống dòng CRLF sang LF.

**5. Nếu ngày mai phải fine-tune cho một khách hàng thật, bước đầu tiên bạn làm là gì?**
Bước đầu tiên tôi làm không phải là mở notebook để train ngay, mà là: **Xây dựng baseline prompt tối ưu (Baseline b) và đóng băng một tập kiểm thử khách quan (Golden Test Set)**. Tôi phải chứng minh được rằng prompt engineering đã chạm trần năng lực và bài toán thực sự cần fine-tune; đồng thời chuẩn bị sẵn 3–5% dữ liệu đệm tổng quát (replay data) để ngăn chặn hiện tượng quên kiến thức nền.
