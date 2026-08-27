# Báo Cáo Phân Tích & Tối Ưu Hóa Chi Phí GPU
---

## 1. Baseline vs. Optimized

Qua việc áp dụng toàn diện các nguyên lý và kỹ thuật FinOps cho hệ thống AI của NimbusAI, hiệu quả kinh tế và chi phí đơn vị đã được tối ưu hóa vượt bậc:

* **Tổng chi phí GPU hàng tháng:**
  * **Baseline Spend (Chưa tối ưu):** **$27,133 / tháng**
  * **Optimized Spend (Đã tối ưu):** **$14,626 / tháng**
  * **Tổng tiền tiết kiệm:** **$12,507 / tháng** (tương đương giảm **46.1%** ngân sách GPU).
* **Đơn giá suy luận (Unit Economics $/1M-token):**
  * **Baseline:** **$6.488 / 1M-token** (do định tuyến toàn bộ request lên mô hình Large mà không có cache hay batching).
  * **Optimized:** **$1.126 / 1M-token** (nhờ áp dụng Model Cascade, Prompt Caching và Batch API).
  * **Tỷ lệ giảm chi phí inference:** Giảm **82.6%** trên mỗi triệu token phục vụ.

---

## 2. Phân Tích Đóng Góp Của Từng Đòn Bẩy (Cost Levers)

| Đòn bẩy tối ưu | Tiết kiệm ($/tháng) | Tỷ trọng tiết kiệm | Nguyên nhân cốt lõi |
| :--- | :---: | :---: | :--- |
| **Purchasing Strategy (Spot + Reserved)** | **$10,040** | **80.3%** | Chuyển đổi các workload huấn luyện có khả năng ngắt quãng (`interruptible`) sang **Spot Tier** (giảm ~40%) và các workload 24/7 sang **Reserved 3-Year** (giảm ~44%). |
| **Inference Levers (Cascade + Cache + Batch)** | **$1,212** | **9.7%** | Kết hợp nhân tử chiết khấu (*multiplicative discount stack*): định tuyến prompt đơn giản sang mô hình Small, chiết khấu 90% cho cached prefix và giảm 50% cho batching không cần real-time. |
| **Right-size GPU-Util Lies** | **$655** | **5.2%** | Hạ cấp các instance GPU bị dư thừa năng lực tính toán (ví dụ H100 chạy workload memory-bound hoặc dưới ngưỡng MFU) xuống các GPU phù hợp hơn như A100/A10G/L4. |
| **Eliminate Idle Waste** | **$600** | **4.8%** | Tự động phát hiện và tắt các instance GPU không có tải (util < 10%) trong ca đêm hoặc sau khi job training kết thúc (cụ thể là `gpu-h100-5` bị lãng phí 8h/ngày). |
| **TỔNG CỘNG** | **$12,507** | **100.0%** | **Tiết kiệm tổng thể 46.1% toàn bộ hạ tầng** |

> **Insight Đòn bẩy lớn nhất:** Chiết khấu mua sắm (**Purchasing Strategy**) đóng góp lớn nhất ($10,040/tháng - 80.3%) bởi vì chi phí thuê GPU phần cứng (GPU-hours) cho các mô hình nền tảng chiếm phần lớn cấu trúc chi phí cố định; việc chuyển dịch thông minh sang Spot với cơ chế checkpointing tự động đem lại ROI tức thì.

---

## 3. GPU-Util Lie

* **Các GPU bị "Lie" phát hiện trong audit:**
  * **`gpu-h100-4`:** `GPU-Util` hiển thị **98.2%**, nhưng `MFU` thực tế chỉ đạt **19.4%** (`MBU = 20.7%`).
  * **`gpu-a10g-1`:** `GPU-Util` hiển thị **96.9%**, nhưng `MFU` chỉ đạt **26.8%** (`MBU = 30.2%`).
* **Bản chất kỹ thuật:**
  * `GPU-Util` của `nvidia-smi` chỉ là thước đo **thời gian hoạt động (time-active metric)**: nó trả về 100% nếu có ít nhất 1 thread/kernel đang thực thi trên GPU trong chu kỳ đo, bất kể kernel đó có khai thác Tensor Cores hay không.
  * Trong thực tế, GPU bị **Memory Stall** (nghẽn băng thông HBM, chờ nạp dữ liệu), **CPU Dataloader Bottleneck** hoặc giao tiếp phân tán (AllReduce synchronization). Tensor Cores bị bỏ đói chu kỳ tính toán (compute cycles).
* **Tác động tài chính:**
  * Doanh nghiệp trả $2.50/h cho H100 với kỳ vọng ~990 TFLOPS FP16, nhưng chỉ nhận được ~192 TFLOPS công có ích.
  * Tương đương với việc **lãng phí hơn 50% tiền thuê GPU** trên từng instance bị ảnh hưởng (~$900–$1,200/tháng cho mỗi GPU H100).

---

## 4. Báo Cáo Thực Hiện Toàn Bộ 5 Phần Mở Rộng (Extensions)

### Extension 1 — Cải thiện `recommend_tier()` và `recommend_tier_advanced()`
* **Nội dung thực hiện:** Đã nâng cấp hàm gợi ý tier trong `finops/pricing.py` để cân nhắc tỷ lệ ngắt quãng thực tế của từng loại GPU (`GPU_INTERRUPT_RATES`, ví dụ H100 spot có preemption rate thấp ~3% so với L4/A10G ~8-10%) và thời gian thực tế của job (`job_days`) để so sánh giữa Reserved 1 năm vs 3 năm.
* **Kết quả đo lường:** Các job training 14 ngày trên H100 được định tuyến chính xác sang Spot tiết kiệm **38.6%**, job eval ngắn ngày sang Spot tiết kiệm **50%**, và các job 24/7 liên tục 3 năm sang Reserved 3yr tiết kiệm **44.0%**.

### Extension 2 — Right-sizing theo MBU (Model Bandwidth Utilization)
* **Nội dung thực hiện:** Xây dựng hàm `suggest_mbu_rightsizing()` và các chỉ số kinh tế đơn vị `vram_unit_cost ($/GB-hr)` và `bandwidth_unit_cost ($/(TB/s)-hr)`.
* **Kết quả đo lường:**
  * H100 có chi phí băng thông $0.746/(TB/s)-hr, A100 là $0.895/(TB/s)-hr, A10G là $1.667/(TB/s)-hr.
  * Với các GPU có MBU thấp (như `gpu-h100-4` MBU 0.207), hệ thống gợi ý hạ cấp sang A100 giúp tiết kiệm **$0.71/giờ ($511.20/tháng, giảm 28.4%)** trong khi vẫn đáp ứng 100% băng thông yêu cầu.

### Extension 3 — Phân tích điểm hòa vốn Prompt Caching (`cache_is_worth_it()`)
* **Nội dung thực hiện:** Xây dựng mô hình toán học đánh giá điểm hòa vốn:
  $$N_{\text{break-even}} = \frac{\text{Write Cost}}{\text{Read Cost} \times (1 - \text{Read Discount})}$$
  Với chiết khấu đọc 90% (`read_discount=0.10`), điểm hòa vốn là **$N_{\text{break-even}} \approx 1.11$ lần đọc**.
* **Kết quả đo lường:** Prompt caching chỉ mang lại lợi nhuận tài chính thực sự khi một cached prefix được tái sử dụng $\ge 2$ lần (như Doc-QA, RAG, Agent loops). Đối với các truy vấn one-shot (chỉ đọc 1 lần), caching gây lỗ $0.10 / 1M-token.

### Extension 4 — Ngân sách Token & Năng lượng cho Reasoning Queries
* **Nội dung thực hiện:** Tách biệt và phân tích các truy vấn suy luận (`is_reasoning=1`) so với truy vấn thông thường (`is_reasoning=0`) với hệ số tiêu thụ năng lượng $80\times$.
* **Kết quả đo lường:**
  * Truy vấn reasoning chỉ chiếm **3.7% số lượng request** và **6.8% chi phí $**, nhưng lại tiêu thụ tới **75.4% tổng năng lượng Wh** của toàn hệ thống serving!
  * **Insight cốt lõi:** Cần thiết lập dynamic confidence threshold routing để chỉ kích hoạt chế độ reasoning khi điểm tự tin của mô hình nhỏ thấp hơn ngưỡng an toàn.

### Extension 5 — Carbon-aware Scheduling cho Workload Ngắt Quãng
* **Nội dung thực hiện:** Mô phỏng điều phối các job training/batch `interruptible=1` từ vùng `us-east-1` (380 gCO2e/kWh) sang vùng xanh `europe-north1` (30 gCO2e/kWh - thủy điện Nauy/Bắc Âu).
* **Kết quả đo lường:**
  * Giảm phát thải carbon từ **1,114.7 kg CO2e xuống 88.0 kg CO2e**, tiết kiệm **1,026.7 kg CO2e (giảm 92.1%)**.
  * Đồng thời tiết kiệm thêm **$81.05** tiền điện do chênh lệch giá điện sạch ($0.09 vs $0.12/kWh).

---

## 5. Ba Khuyến Nghị Hành Động Hàng Đầu Cho FinOps Lead

1. **Thiết lập Chính sách Mua sắm Tự động (Automated Purchasing Policy & Spot Checkpointing):**
   * Bắt buộc tích hợp thư viện tự động lưu checkpoint định kỳ (mỗi 30 phút) cho toàn bộ pipeline training và fine-tuning để chuyển dịch 100% job interruptible sang Spot instances (tiết kiệm ngay ~$10,000/tháng).
   * Cam kết Reserved Instance 3 năm đối với các cụm inference phục vụ API 24/7 ổn định.
2. **Triển khai Guardrail Tagging & Chargeback Gates:**
   * Áp dụng chính sách tag bắt buộc (`team`, `project`, `environment`) tại CI/CD và Kubernetes level.
   * Kích hoạt cơ chế Chargeback / Showback định kỳ theo chuẩn FOCUS (`outputs/focus_export.csv`) để gắn trách nhiệm tài chính về từng nhóm kỹ sư.
3. **Chuẩn hóa Bộ chỉ số MFU/MBU và Dynamic Model Routing:**
   * Ngừng sử dụng `GPU-Util` làm KPI hiệu năng. Đưa chỉ số **MFU** ($\ge 40\%$) và **MBU** vào dashboard Datadog/Prometheus; tự động cảnh báo các GPU bị "Util Lie".
   * Kích hoạt cascade routing kết hợp Prompt Caching cho 100% hệ thống RAG và Chatbot để duy trì đơn giá suy luận $\le \$1.20 / \text{1M-token}$.
