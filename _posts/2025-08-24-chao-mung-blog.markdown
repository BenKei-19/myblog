---
layout: my-post
title: "Chào mừng blog mới"
date: 2025-08-24
---

Một cách đơn giản nhất, chúng ta có thể thấy rằng: i) diện tích nhà càng lớn thì giá nhà càng cao; ii) số lượng phòng ngủ càng lớn thì giá nhà càng cao; iii) càng xa trung tâm thì giá nhà càng giảm. Một hàm số đơn giản nhất có thể mô tả mối quan hệ giữa giá nhà và 3 đại lượng đầu vào là:

Mối quan hệ giữa biến đầu ra và đầu vào có thể viết gọn: $y \approx f(\mathbf{x}) = \hat{y}$.

$$ f(\mathbf{x}) = w_1 x_1 + w_2 x_2 + w_3 x_3 + w_0 \qquad (1) $$

Trong đó, $w_1, w_2, w_3, w_0$ là các hằng số; $w_0$ còn được gọi là bias. Mối quan hệ $y \approx f(\mathbf{x})$ bên trên là tuyến tính (linear). Bài toán chúng ta đang làm là regression; đi tìm các hệ số tối ưu $\{w_1, w_2, w_3, w_0\}$ chính là bài toán Linear Regression.

Chú ý 1: $y$ là giá trị thực của outcome (dựa trên số liệu thống kê chúng ta có trong tập training data), trong khi $\hat{y}$ là giá trị mà mô hình Linear Regression dự đoán được. Nhìn chung, $y$ và $\hat{y}$ là hai giá trị khác nhau do có sai số mô hình; tuy nhiên, chúng ta mong muốn sự khác biệt này rất nhỏ.