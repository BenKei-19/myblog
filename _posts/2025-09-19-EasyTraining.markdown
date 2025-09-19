---
layout: my-post
title: "Một số phương pháo để huấn luyện cũng như là sử dụng nếu như phần cứng không đủ đáp ứng trong AI"
date: 2025-09-19

summary: "Phương pháp để mình có thể sử dụng cũng như Fine-tunning AI-Model khi điều kiện phần cứng chưa đủ đáp ứng."
---

<div style="text-align: center; font-weight: bold; font-size: 20px"> Contents </div>
<h2>Happy Birth Day SUCCESS</h2>
<hr./>
**I. Giới thiệu bối cảnh**  
**II. Cách sử dụng mô hình khi phần cứng không đáp ứng đủ (Knowledge Distillation)**  
**III. Một số cách huấn luyện mô hình khi gặp khó khăn về không đủ phân cứng**  

<div style="text-align: center; font-weight: bold; font-size: 20px"> I. Giới thiệu bối cảnh </div>
Như các bạn đã biết thì gần đây với sự nổi lên của ChatGPT, hay còn được gọi là các mô hình ngôn ngữ lớn (Large Language Models - LLMs), thì các mô hình(có cả trả phí) hay miễn phí được public lên với hàng tỷ tham số, gần đây nhất Open AI có phát hành hai mô hình gpt-oss-120b và gpt-oss-20b với kích cỡ size là 120 tỷ tham số và 20 tỷ tham số, đây là con số không tưởng vì kích thước mô hình càng ngày càng được tăng lên, nhưng đối với một số lượng tham số khủng như thế thì các bạn học sinh, hay sinh viên muốn làm dự án hay project về AI, các doanh nghiệp cũng phải bỏ ra số tiền quá lớn để có thể huấn luyện và sử dụng. Vì vậy các nhà khoa học đã nghĩa ra một số cách để tối ưu, có thể thực hiện làm được ngay trên máy local(có GPU) hay trên các nền tảng online như Google Colab hay Kaggle có thể huấn luyện được mình sẽ cố gắng tổng hợp trong bài viết này.
<div style="text-align: center; font-weight: bold; font-size: 20px"> II. Cách sử dụng mô hình khi phần cứng không đáp ứng đủ (Knowledge Distillation) </div>

<div>
<ol>
<li>
<p style="font-family: bold; color: red;"> Khái niệm về Knowledge Distillation </p>
Khái niệm về Knowledge Distillation được giới thiệu vào năm 2015, là một trong những kĩ thuật thuộc họ transfer learning. Có một điều cơ bản mình có thể nhận thấy ở quá trình học tập của người rằng mình cần đến những người thầy cô, những người hướng dẫn mình để mình có thể dễ dàng hơn trong quá trình học, nghĩa là quá trình mà kiến thức được truyền đạt từ người có kiến thức sâu rộng hơn đối với người có hiểu biết kém hơn. Mapping ý tưởng đó sang AI, thì kĩ thuật này là ta sẽ huấn luyện hai model, một model có kích cỡ lớn đóng vai trò là teacher nhằm chuyển giao kiến thức sang model nhỏ hơn hay đồng nghĩa là student. Điều này đồng nghĩa giúp việc ta có thể triển khai các mô hình nhỏ hơn trên các thiết bị phần cứng yếu hơn mà không làm ảnh hưởng quá nhiều đến hiệu năng.
</li>

<li>
<p style="font-family: bold; color: red;"> Quá trình Distillation </p>
Quá trình Distillation được chia thành các bước:
<ul>
<li>
<b> Huấn luyện teacher </b>: Đói với việc huấn luyện teacher như này, ta tiến hành huấn luyện với dữ liệu được gán nhãn như bình thường. Sau khi đạt được accuracy tốt rồi thì ta sẽ tiến hành thực hiện mô hình teacher huấn luyện cùng với mô hình student có tham số nhỏ hơn.
</li>
<li>
<b> Huấn luyện student</b>: Ta vẫn sẽ huấn luyện student giống như có label nhưng ta sẽ có thêm "sự chỉ dẫn" của teacher nữa. Thông thường khi ta huấn luyện student, ta sẽ áp dụng hàm Loss Function dạng cross-entropy:
$$
CE(\mathbf{q}, \mathbf{p}) \triangleq \mathbf{H}(\mathbf{q}, \mathbf{p}) 
= - \sum_{i=1}^{C} q_i \log(p_i)
$$
&#9679; <b> Với C là số class cần phân loại</b> <br>
Ý nghĩa của hàm Cross-Entropy ở đây là đo khoảng cách giữa phân phối thật \( q \) so với phân phối dự đoán \( p \). Nếu \( p \) mà gần giống \( q \) thì CE của nó sẽ nhỏ, và ngược lại.<br>
<b>Ví dụ</b>: so với bài toán MNIST phân biệt chữ viết tay, sau khi ta trích xuất được các đặc trưng của hình ảnh, ta đưa qua hàm Softmax có 10 phần từ tương ứng với xác suất nó sẽ rơi vào số từ 0 đến 9, nếu xác suất nào cao hơn thì ta sẽ lấy đó là kết quả. Và trong quá trình huấn luyện này, ta dùng hàm loss là cross-entropy để kéo xác suất mà mình vừa qua hàm Softmax so với lại one-hot vector có chứa label (là vector toàn số 0 và chỉ có một số 1 ở label) mục tiêu là kéo gần vector dự đoán gần hơn so với one-hot vector mà mình vừa nói ở trên.<br>
Và để cho mô hình có thể dự đoán tốt hơn, thì kết quả của teacher cũng sẽ đóng góp vào quá trình học của Student model, giống như là "đường dẫn" để chỉ học sinh có thể học được tốt hơn. Nghĩa là ta sẽ giảm impact của one-hot label vector trong quá trình học của student và thay vào đó là dùng gợi ý từ teacher để học. <br>
Như vậy \( distillation \) loss tại dữ liệu thứ \( x_i \) sẽ là:
$$
\mathcal{L}_{\text{dl}}(\mathbf{x}_i; \mathbf{W}) = \mathbf{H}(\mathbf{q}_{it}, \mathbf{q}_{is})
$$
Trong đó: <br>
&#9679; \( q_{it}\) là vector xác suất tại data thứ i và của mô hình Teacher <br>
&#9679; \( q_{is}\) là vector xác suất tại data thứ i và của mô hình Student <br>
<p style="color: red;">Vậy tại sao cách học này lại tốt hơn so với cách học so với toàn label thông thường?</p>
Điểm tốt hơn khi ta dùng vector của teacher chứ không phải one-hot vector là teacher không chỉ đưa được thông tin đúng về label, mà teacher còn có thể cung cấp được là à đây được phân loại là con mèo này nhưng nó cũng hao hao khá giống với con chó đó thông qua vector label của teacher. <br>
Còn một điều nữa là từ quan sát, thì tác giả thấy rằng có nhiều trường hợp mà vector của teacher khá là thô cứng, nghĩa là nó gần như là một one-hot vector của label luôn rồi, điều này làm mất đi khả năng học các kiến thức "dark knowledge" từ teacher, lúc này tác giả đã sử dụng một kĩ thuật để tạo ra một vector mà "mềm" hơn, nghĩa là nó phân bổ đều hơn khác với one-hot vector. Lúc này tác giả sử dụng phương pháp \( temperature scale \).
</li>
</ul>
</li>
<h5>2.1 <span style="color:red; font-size:16px"> Temperature Scale </span> </h5>
Đây là một phương pháp calibration score làm cho softmax score trở nên được mượt mà hơn. Ví dụ ta có một phân phối dựa trên hàm softmax của vector input \(
\mathbf{x} = (x_{1}, x_{2}, \ldots, x_{C})
\) như sau:
\[
e_i = \frac{\exp(x_i)}{\sum_{k=1}^{C} \exp(x_k)}
\]
Trong đó \( C \in \mathbb{N} \) là số class ta cần phân loại. Khi áp dụng calibration score theo hệ số temperature scale T thì ta sẽ thu được xác suất mới như sau:
\[
e_i' = \frac{\exp(x_i / T)}{\sum_{k=1}^{C} \exp(x_k / T)}
\]
\(
\forall i \in \{1,2,\ldots,C\}
\)
Xác xuất mới \( e_i' \) còn được gọi là softmax temperature, được đề xuất bởi tác giả Geoffrey Hinton vào 2015. Tác giả chọn hệ số \(T \) từ giá trị 1 đến 5, với \( T = 1 \) thì \( e = e'\) và đây là giá trị softmax như thông thường. Khi \( T \) càng tăng thì giá trị phân phối của xác suất softmax temperature \( e' \) sẽ càng smoothing hơn so với e. Sau đây là đoạn code để chúng ta có thể thấy được điều đó.
{% highlight python %}
import numpy as np
import matplotlib.pyplot as plt
x = np.random.rand(9)

def softmax(x):
    return np.exp(x) / np.sum(np.exp(x))

def temperature_scale(x, T):
    temp = [X / T for X in x ]
    return softmax(temp)

temper_scale = temperature_scale(x, 2)
non_scale = softmax(x)
range = np.arange(len(temper_scale))
plt.plot()
plt.plot(range, non_scale, label = "Non_scale")
plt.plot(range, temper_scale, label = "Scale with T = 2")
plt.legend()
plt.show()
{% endhighlight %}
<figure style="text-align: center;">
    <img src="{{ site.baseurl }}/assets/images/Temperature_Score.png" alt="Ảnh ví dụ về độ smooth Khi T=1 với T=2" style="max-width: 100%; height: auto;">
</figure>
Từ hình ảnh ta thấy rõ được sự khác biệt khi ta dùng temperature scale(đường màu vàng). Ta nhận thấy rằng thứ tự lớn nhất vẫn không thay đổi nhưng ta thấy nó được smooth hơn nhiều.<br>
Vậy tại sao khi áp dụng calibration lại giúp tạo ra một chuỗi smoothing hơn? Nguyên nhân đó chính là hệ số temperature scale đã làm giảm phương sai của phân phối xác suất. Thật vậy ta sẽ chứng minh rằng phương sai
của phân phối xác suất \( e \) sẽ lớn hơn \( e' \).
Trước tiên ta sẽ có biểu thức như sau
\[
\mathrm{Var}(e) = \mathbb{E}(e^2) - [\mathbb{E}(e)]^2
\]

\[
= \mathbb{E}(e^2) - \frac{1}{C^2}
\]

Dòng thứ 2 là vì 
\[
\mathbb{E}(e) = \frac{1}{C} \quad \text{với $C$ là số lượng classes.}
\]

Như vậy để chứng minh phương sai của $e$ lớn hơn $e'$ ta qui về chứng minh:

\[
\mathbb{E}(e^2) \geq \mathbb{E}({e'}^2)
\]

Đặt:

\[
f(x,T) \triangleq C \, \mathbb{E}(e^2)
\]

\[
= \sum_{i=1}^C \left[ \frac{\exp(x_i/T)}{\sum_{i=1}^C \exp(x_i/T)} \right]^2
\]

\[
= \sum_{i=1}^C \bigl[ \sigma(x_i/T) \bigr]^2
\]
Trong đó $\mathbf{x}$ là một vectơ phân phối xác suất của $C$ classes và $T$ là hệ số temperature, $T \geq 1$.

Để đơn giản hoá, ta sử dụng $\sigma(x_i)$ là ký hiệu của hàm sigmoid thay cho công thức 
\[
\sigma(x_i) = \frac{\exp(x_i)}{\sum_{i=1}^C \exp(x_i)}.
\]

Một tính chất khá quan trọng của đạo hàm sigmoid:
\[
\frac{\delta \, \sigma(x)}{\delta x} = \sigma(x)(1 - \sigma(x)).
\]

Như vậy:
\[
\frac{\delta f(\mathbf{x}, T)}{\delta T} 
= \sum_{i=1}^C \frac{\delta \, \sigma(x_i/T)^2}{\delta T}
\]

\[
= \sum_{i=1}^C \frac{\delta \, \sigma(x_i/T)^2}{\delta \, \sigma(x_i/T)} 
   \cdot \frac{\delta \, \sigma(x_i/T)}{\delta \, (x_i/T)} 
   \cdot \frac{\delta \, (x_i/T)}{\delta T}
\]

\[
= \sum_{i=1}^C 2 \, \sigma(x_i/T) \, \sigma(x_i/T)(1 - \sigma(x_i/T)) \cdot \frac{-x_i}{T^2}
\]

\[
= \sum_{i=1}^C 2 \, \sigma(x_i/T)^2 \, (1 - \sigma(x_i/T)) \cdot \frac{-x_i}{T^2}.
\]

Do $\sigma(x_i/T) \in (0,1)$ và $x_i \geq 0$ nên
\[
2 \, \sigma(x_i/T)^2 \, (1 - \sigma(x_i/T)) \cdot \frac{-x_i}{T^2} \leq 0, 
\quad \forall x_i, \, i = 1,\dots, C.
\]

Tức là:
\[
\frac{\delta f(\mathbf{x},T)}{\delta T} \leq 0.
\]

Suy ra $f(\mathbf{x}, T)$ là một hàm nghịch biến.  
Do đó $f(\mathbf{x}, T) \leq f(\mathbf{x}, 1)$, suy ra 
\[
\mathbb{E}(e^2) \geq \mathbb{E}(e'^2) \quad \Rightarrow \quad \mathrm{Var}(e) \geq \mathrm{Var}(e').
\]
<h5>2.2 <span style="color:red; font-size:16px"> Distillation loss </span> </h5>
Sau khi áp dụng phương pháp temperature scale thì phân phối xác suất của teacher và student sẽ mềm mại hơn. Để hiểu chúng ta đang dùng xác suất sau khi đã áp dụng temperature scale, mình sẽ kí hiệu lần lượt là \(q'_t\) và \(q'_s\) lần lượt là phân phối xác suất của teacher và student sau khi đã scale. Lúc lày Distillation loss sẽ trở thanh:
\[
\mathcal{L}_{\mathrm{dl}}(x_i; \mathbf{W}) 
= \mathbf{H}(q_{it}, q'_{is}) 
= \mathbf{H}\bigl(\sigma(z_{it}; T = \tau), \, \sigma(z_{is}; T = \tau)\bigr)
\]
với \( \sigma(z_i; T = \tau) \) là kí hiệu của hàm softmax sau khi làm mềm logit \( z_i \) với temperature scale \( \tau \), index \( it \) tương ứng với data set thứ i dưới góc nhìn của teacher, tương tự với \( is \) tương ứng với dataset thứ i dưới góc nhìn của student. <br>
Lúc này tác giả cũng thấy thông qua việc thực nghiệm rằng việc học sẽ hiệu quả hơn đôi chút nếu có cả sự kết hợp của thầy giáo cũng như ground-truth, lúc này, hàm Loss khi huấn luyện cho student sẽ được biểu diễn như sau
\begin{align}
\mathcal{L}_{\mathrm{final}}(x_i; \mathbf{W}) 
&= \alpha \, \mathcal{L}_{\mathrm{student}}(x_i; \mathbf{W})
 &+ \beta \, \mathcal{L}_{\mathrm{dl}}(x_i; \mathbf{W}) \\[6pt]
&= \alpha \, \mathbf{H}(y_i, q_{is})
 &+ \beta \, \mathbf{H}(q_{it}, q'_{is}) \\[6pt]
&= \alpha \, \mathbf{H}\!\bigl(y_i, \sigma(z_{is}; T = 1)\bigr)
 &+ \beta \, \mathbf{H}\!\bigl(\sigma(z_{it}; T = \tau), \, \sigma(z_{is}; T = \tau)\bigr)
\end{align}
Lúc này \(y_i\) chính là ground truth đối với data set thứ i, trường hợp \( T= 1\) chính là lấy hàm softmax mà không có temperature scale.<br>
Thông thường \(\beta = 1 - \alpha\), \( \alpha \) sẽ được chọn như là một hệ số rất nhỏ => \( 1 - \beta\) sẽ lớn hơn, điều này cho thấy chủ yếu student sẽ được học từ teacher nhiều hơn do tác động của teacher vào hàm loss của student sẽ lớn hơn.<br>
Để hiểu rõ hơn, dưới đây là cách mà distillation hoạt động
<figure style="text-align: center;">
    <img src="{{ site.baseurl }}/assets/images/Distillation.png" alt="Distillation" style="max-width: 100%; height: auto;">
    <figcaption><i>Nguồn: https://phamdinhkhanh.github.io/2021/03/13/KnownledgeDistillation.html</i></figcaption>
</figure>
Mình sẽ giải thích mô hình trên như sau:<br>
Ban đầu ta sẽ có input đầu vào là \(x\) cả hai đều đi qua hết cả các teacher layer và các student layer, lúc này ở layer cuối cùng sẽ trả cho ta về các vector giá trị(Gọi là logit), xong sau đó ta áp dụng temperature scale cho hai logit vừa nói ở trên, lúc này ta dùng cross-entropy để khéo hai vector xác suất này vào với nhau, để cho việc học ổn định hơn thì ta sẽ thêm hàm loss của ground truth nữa, lúc này logit của student(lúc chưa làm smooth) sẽ được đưa vào hàm cross entropy cùng với ground-truth để kéo vector của student cũng gần hơn với ground-truth.
</ol>
<h5> <span style="color:red; font-size:22px">3. Tổng kết </span> </h5>
<p>
Knowledge Distillation (tạm dịch: <i>chưng cất tri thức</i>) là một phương pháp trong học máy, 
lấy cảm hứng từ cách con người học.
</p>

<p>Bạn có thể hình dung như sau:</p>

<ul>
  <li><b>Học sinh (student model)</b>: mô hình nhỏ, đơn giản, cần được học hỏi.</li>
  <li><b>Giáo viên (teacher model)</b>: mô hình lớn, phức tạp, đã học nhiều kiến thức.</li>
  <li>Student học lại từ teacher, giống như học sinh được thầy cô hướng dẫn.</li>
</ul>
<h5> <span style="color:red; font-size:22px">4. Bàn luận </span> </h5>
<h4>1. Điểm mạnh của Knowledge Distillation</h4>
<ul>
  <li><b>Giảm kích thước mô hình</b>: Student model nhỏ gọn nhưng vẫn giữ được phần lớn độ chính xác từ teacher.</li>
  <li><b>Tốc độ suy luận nhanh hơn</b>: Thích hợp triển khai trên thiết bị di động hoặc môi trường hạn chế tài nguyên.</li>
  <li><b>Tận dụng được mô hình phức tạp</b>: Teacher có thể rất lớn và mạnh (ensemble, deep networks), student chỉ cần học lại phiên bản rút gọn.</li>
  <li><b>Ổn định hơn so với training trực tiếp</b>: Student không chỉ học từ nhãn cứng (<i>hard labels</i>) mà còn học từ phân phối mềm (<i>soft labels</i>), giúp giảm overfitting.</li>
  <li><b>Linh hoạt</b>: Có thể kết hợp với nhiều kiến trúc khác nhau (CNN, RNN, Transformer, attention, multi-task...).</li>
</ul>


<h4>2. Điểm yếu của Knowledge Distillation</h4>
<ul>
  <li><b>Phụ thuộc vào teacher model</b>: Nếu teacher kém chất lượng, student cũng không học được nhiều.</li>
  <li><b>Tăng chi phí huấn luyện</b>: Cần train một teacher lớn trước rồi mới train student → tốn tài nguyên và thời gian.</li>
  <li><b>Khó khăn trong thiết kế</b>: Việc chọn temperature, loss function, hay cách truyền tri thức (logits, feature maps, attention…) ảnh hưởng nhiều đến hiệu quả.</li>
  <li><b>Giới hạn bởi năng lực student</b>: Nếu student quá nhỏ so với teacher, nó không thể “chứa” hết kiến thức → hiệu năng bị giảm.</li>
  <li><b>Không phải bài toán nào cũng phù hợp</b>: KD thường hiệu quả trong classification, nhưng với các bài toán phức tạp như text generation cần điều chỉnh kỹ hơn.</li>
</ul>
<h4>3. Một số cách trong huấn luyện khác</h4>
Ở trên mình đã nói đến thường sẽ có cách huấn luyện teacher trước, sau khi teacher đã đạt một accuracy cao rồi thì ta có thể mới tiếp tục đến student, nhưng thực tế, dưới góc nhìn của mình thì việc cả student và cả teacher đều học thì mình sẽ nghĩ là sẽ có khả năng mô hình sẽ tốt hơn đúng không. Lúc này ta sẽ có thể train của student và teacher cùng một lúc với hàm loss của teacher cũng giống như hàm loss của student. Nhưng điều này xảy ra một hiện tượng là, ban đầu, cả hai người đều không biết mà lại còn hướng dẫn nhau?. Có một cách đơn giản là ban đầu ta sẽ lựa chọn hệ số \( \alpha \) cao hơn, xong sau một số epoch rồi ta có thể giảm hệ số \(\alpha\) thì lúc này ta sẽ tránh được trường hợp hai người không biết hướng dẫn nhau (cách này ta gọi là scheduling). Ngoài cách mình vừa nói trên ra thì mình sẽ có một số cách nữa như sau:
<ol>
<li>
Self-distillation
<ul>
  <li><b>Ý tưởng</b>: Teacher và Student có cùng kiến trúc.</li>
  <li>Teacher là phiên bản của model ở giai đoạn huấn luyện trước (checkpoint tốt hơn).</li>
  <li>Student học lại từ phiên bản cũ của chính mình.</li>
</ul>
<p>Cách này thường giúp cải thiện hiệu năng mà <b>không cần teacher quá lớn</b>.</p>
</li>

<li>
Assistant-based Distillation (Teacher Assistant KD)
<ul>
  <li><b>Ý tưởng</b>: Thay vì truyền kiến thức trực tiếp từ Teacher <b>rất lớn</b> → Student <b>rất nhỏ</b>, ta thêm một hoặc nhiều mô hình trung gian (Assistant).</li>
  <li>Kiến thức được truyền <b>từng bước</b> từ Teacher → Assistant → Student.</li>
</ul>
<p> Giống như học sinh tiểu học không thể học thẳng từ giáo sư đại học, mà cần qua thầy cô trung gian.</p>

</li>

<li>Layer-wise Distillation
<ul>
  <li><b>Ý tưởng</b>: Student không chỉ học từ <b>output logits</b> mà còn học <b>representation từ từng layer của Teacher</b>.</li>
  <li>Nhờ đó Student nắm được nhiều mức độ đặc trưng hơn (từ <b>low-level → high-level</b>).</li>
</ul>
</li>
</ol>
Như vậy mình đã giới thiệu cho các bạn cách để có thể sử dụng một mô hình nhỏ hơn mà không làm ảnh hưởng đến quá nhiều hiệu năng của mô hình, vì paper này đã được ra đời vào năm 2015 nên đã có rất nhiều cải tiến cho cách này, nếu hứng thú các bạn hay tìm đọc nhé. Bài sau mình sẽ nói đến một số phương pháp Parameter-Efficient Fine-Tuning (PEFT) để có thể dễ dàng hơn trong việc huấn luyện khi thiết bị GPU chưa có quá nhiều vram nha.
<h5> <span style="color:red; font-size:22px">5. Reference </span> </h5>
<a href="https://phamdinhkhanh.github.io/2021/03/13/KnownledgeDistillation.html" target="_blank" rel="noopener">
  Anh Khánh Đình Phạm
</a><br>
<a href="https://www.youtube.com/watch?v=ueUAtFLtukM" target="_blank" rel="noopener">
  Knowledge Distillation with TAs
</a><br>
<a href="https://docs.aws.amazon.com/prescriptive-guidance/latest/ml-quantifying-uncertainty/temp-scaling.html" target="_blank" rel="noopener">
  Temperature scaling
</a><br>
<a href="https://viblo.asia/p/knowledge-distillation-chat-loc-tri-thuc-tu-nhung-mo-hinh-thanh-cong-naQZRBYjZvx#_uu-diem-cua-knowledge-distillation-4" target="_blank" rel="noopener">
  Knowledge Distillation – Chắt lọc tri thức từ những mô hình thành công
</a><br>
<a href="https://medium.com/analytics-vidhya/knowledge-distillation-in-a-deep-neural-network-c9dd59aff89b" target="_blank" rel="noopener">
  Knowledge Distillation in a Deep Neural Network
</a><br>

</div>



