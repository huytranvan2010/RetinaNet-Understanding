# RetinaNet-Understanding

Trong bài này chúng ta sẽ tìm hiểu về one-stage detector có tên là RetinaNet. Đầu tiên cùng tìm hiểu về **Focal Loss**, chính loss function này làm nên sự khác biệt của RetinaNet. Các bài toán object detection trước đó luôn gặp phải một vấn đề về class imbalance (ám chỉ sự chênh lệch giữa foreground và background là quá lớn).

Nhớ lại cross entropy (CE) cho bài toán phân loại đối với nhãn thực tế là $\mathbf{y}$ và nhãn dự đoán là $\mathbf{\hat{y}}$:

$$CE(\mathbf{\hat{y}}, \mathbf{y}) = - \sum_{i}  y_i log(\hat{y_i}) $$

Đối với bài toán phân loại nhị phân ta có:

$$CE(p, y) = - ~ y log(p) - (1-y) log(1-p) $$

trong đó $p$ (hay $\hat{y}$) là xác suất dự đoán đầu ra cho foreground với label $y=1$. $1-p$ là xác suất dự đoán cho class còn lại background với label $y=0$. Mình kí hiệu như này giống bài báo để tiện theo dõi.

Để thuận tiện đặt:

$$p_t = \left\{\begin{matrix}
p ~~~~~~~~~~~~~~~\text{if} ~ y=1\\
1-p ~~~~~~~~ \text{if} ~ y=0
\end{matrix}\right.$$

Khi đó $CE(p,y) = CE(p_t) = -log(p_t)$ bởi vì:
- $y=1$ thì $CE(p, y=1) = -log(p) = log(p_t)$
- $y=0$ thì $CE(p, y=0) = -log(1-p) = log(p_t)$

<img src="https://miro.medium.com/max/810/1*Rm-vU6yZjB9lnCIhWhza4Q.png">

Cùng xem hình trên với $\gamma = 0$ chính là trường hợp cross entropy bình thường. Với well-classified class loss nhận được rất nhỏ. Tuy nhiên nên nhớ lại đây là loss cho một examples. Trong bài toán object detection số lượng background của chúng ta tạo ra lớn hơn rất nhiều số lượng foreground (chênh lệch nhau đến hàng nghìn lần), vì vậy loss tổng hợp cho tất cả background examples sẽ lớn hơn loss cho tất cả foreground examples. 

Xem thêm ví dụ dưới đây

<img src="https://miro.medium.com/max/614/1*b8Z0SprNLpNRLv8-8lzpdQ.png">

Chúng ta có 100000 easy examples (0.1 loss cho mỗi example) và 100 hard examples (2.3 loss cho mỗi example). Khi đó tập hợp lại ta được
- Loss cho tất cả easy examples = 100000×0.1 = 10000
- Loss cho tất cả hard examples = 100×2.3 = 230
- 10000 / 230 = 43. Loss từ easy examples lớn hơn rất nhiều loss từ hard examples. 

Cross entropy loss không phải là sự lựa chọn tốt cho trường hợp rất mất cân bằng dữ liệu. Nếu dùng cross entropy thông thường này thì  
- Ngay cả mô hình khi dự đoán sai foreground (predicted probability of foreground thấp, gần gốc tọa độ) thì loss do việc dự đoán foreground sai này vẫn nhỏ, mô hình thậm chí không cần cải thiện thêm vẫn được. Điều này là không chấp nhận được. **Mô hình ít quan tâm đến dự đoán đúng foreground vì loss do dự đoán sai foreground không ảnh hưởng nhiều đến loss chung**.
- Hay khi mô hình dự đoán sai các easy examples (predicted probbability of foreground cao - gần điểm 1 trên trục hoành, tương đương với probability of background nhỏ) thì tổng loss cho việc dự đoán sai background vẫn lớn. Do đó mô hình lại càng cố gắng dự đoán đúng nhất backgound để giảm loss xuống. **Mô hình đang tập trung dự đoán đúng background để giảm loss**.

Chính vì những điều trên chúng ta cần một loss function hiệu quả hơn giúp **điều chỉnh loss lớn hơn khi dự đoán sai foreground (object)**. Điều này giúp chúng ta hạn chế dự đoán sai đối với foreground vì khi dự đoán sai loss sẽ tăng lên đáng kể.



## RetinaNet


## Tài liệu tham khảo
1. https://arxiv.org/abs/1708.02002
2. https://towardsdatascience.com/review-retinanet-focal-loss-object-detection-38fba6afabe4
3. https://towardsdatascience.com/retinanet-the-beauty-of-focal-loss-e9ab132f2981
4. https://www.phamduytung.com/blog/2018-12-06-what-do-we-learn-from-single-shot-object-detection/
5. https://phamdinhkhanh.github.io/2020/08/23/FocalLoss.html 

