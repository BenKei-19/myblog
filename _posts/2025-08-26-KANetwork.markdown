---
layout: my-post
title: "Kolmogorov–Arnold Networks(KAN) sự thay thế cho MultiLayer Perceptron?"
date: 2025-08-26
summary: "Giới thiệu về KAN và sự phát triển của KAN so với MLP"
---
<div style="text-align: center; font-weight: bold; font-size: 20px"> Contents </div>

**I. Giới thiệu đôi nét về KAN**  
**II. Review: Multi-Layer Perceptron(MLP)**  
**III. Kolmogorov–Arnold Networks(KAN)**  

<div style="text-align: center; font-weight: bold; font-size: 20px"> I. Giới thiệu đôi nét về KAN </div>

<div>
    <ol>
        <li>
            Được lấy cảm hứng từ Kolmogorov–Arnold representation theorem, các tác giả trong paper về KAN đã phát triển từ biểu diễn của Kolmogorov–Arnold bằng neuron networks, để vinh danh hai nhà toán học vĩ đại này, tác giả của bài báo về KAN(mình sẽ để ở phần tài liệu tham khảo) đã gọi chúng là Kolmogorov–Arnold Network
        </li>
        <li>
            Về định lý này nói rằng cho hàm liên tục và nhiều biến \( f(x_1, x_2, \ldots, x_n) \) đều có thể biểu diễn thành một tổng hữu hạn của các hàm liên tục một biến với phép cộng
            $$
            f(x_1, x_2, \ldots, x_n) = \sum_{q=1}^{2n+1} \phi_q \left( \sum_{p=1}^n \psi_{pq}(x_p) \right)
            $$
        Trong đó:
        <ul>
            <li>
                \( f(x_1, x_2, \ldots, x_n) \) là hàm nhiều biến(n biến) mà mình muốn biểu diễn
            </li>
            <li>
                Chỉ số \(q\) chạy từ 1 đến 2n+1 có nghĩa là ta chỉ cần nhiều nhất đến 2n+1 hàm
            </li>
            <li>
                \(\psi_{pq}(x_p)\): các hàm một biến phụ thuộc vào từng biến đầu vào \(x_p\).
            </li>
            <li>
                \(\phi_q(\cdot)\): các hàm một biến khác, tác động lên tổng bên trong.
            </li>
            <li>
                Và sau đó ta tổng tất cả lại, à còn lí do vì sao lại là 2n+1 thì các bạn đào sâu vào bài của hai nhà toán học mình sẽ để ở phần tài liệu tham khảo nha.
            </li>
        </ul>
        </li>
        <li>
            KAN và MLP có một chút điểm giống và khác nhau nhưng KAN được cố gắng biểu diễn dưới dạng của một MLP
            <figure style="text-align: center;">
                <img src="{{ site.baseurl }}/assets/images/KANandMLP.png" alt="Ảnh Phân Biệt KAN và MLP" style="max-width: 100%; height: auto;">
                <figcaption><i>Nguồn: https://www.dailydoseofds.com/a-beginner-friendly-introduction-to-kolmogorov-arnold-networks-kan/</i></figcaption>
            </figure>
            <ul>
                <li>
                    Dựa vào hình trên các bạn thấy rằng, trong MLP, ta sẽ học các trọng số trên các cạnh (màu xanh và màu đỏ), và ta fix cứng hàm activation trong MLP (có thể là sigmoid, relu, v.v)
                </li>
                <li>
                    Còn đối với KAN, ta sẽ học cả các cạnh và học cả các activation để có thể biểu diễn tốt hơn, như các bạn thấy trong hình chữ nhật màu xanh ở KAN, các hình vẽ tượng chưng cho các function mà mô hình học được.
                </li>
            </ul>
        </li>
    </ol>
</div>

<div style="text-align: center; font-weight: bold; font-size: 20px"> II. Review: Multi-Layer Perceptron(MLP) </div>

<div>
    <figure style="text-align: center;">
        <img src="{{ site.baseurl }}/assets/images/MLPExample.jpg" alt="MLP Example" style="max-width: 100%; height: auto;">
        <figcaption><i>Nguồn: https://www.sciencedirect.com/topics/computer-science/multilayer-perceptron</i></figcaption>
    </figure>
    <ul>
        <li>
            INPUT LAYER
            <ul>
                <li>
                    Là những Layer ở ngoài cùng bên trái, mình có thể hiểu mỗi một ô khoanh tròn là một số ví dụ như với bài toán dự đoán giá nhà dựa trên diện tích, số phòng ngủ,... thì diện tích là x1 ở phần Input Layer, số phòng ngủ là x2 của input layer,.... Và có duy nhất một Input Layer
                </li>
            </ul>
        </li>
        <li>
            HIDDEN LAYER
            <ul>
                <li>
                    Là những Layer ở giữa, chúng có thể có nhiều HIDDEN LAYER với nhau, và số NODE ở mỗi hidden layer đều cho mình có thể quy ước, về mặt trực quan, số node ở hidden layer càng nhiều và càng nhiều hidden layer thì mô hình có thể sẽ học được sự phức tạp của dữ liệu, và ở hình trên bạn thấy mỗi node ở Layer trước sẽ có edge nối toàn bộ với các node ở layer sau
                </li>
            </ul>
        </li>
        <li>
            OUTPUT LAYER 
             <ul>
                <li>
                    Tùy theo đề bài mà ta sẽ quy định số node của Output layer, nếu bài toán phân loại về Chó, Mèo thì output layer sẽ là 2, nếu bài toán về dataset Mnist thì output layer sẽ là 10.
                </li>
            </ul>
        </li>
    </ul>
     <figure style="text-align: center;">
        <img src="{{ site.baseurl }}/assets/images/MLPMath.png" alt="MLP Example" style="max-width: 100%; height: auto;">
        <figcaption><i>Nguồn: https://machinelearningcoban.com/2017/02/24/mlp/</i></figcaption>
    </figure>
    Và biểu thức toán học của MLP được viết dưới dạng như sau:
    \[
\mathbf{W}^{(l)} \in \mathbb{R}^{d^{(l-1)} \times d^{(l)}}
\]

\[
\mathbf{b}^{(l)} \in \mathbb{R}^{d^{(l)} \times 1}
\]

\[
z^{(l)}_j = \mathbf{w}^{(l)T}_j \, \mathbf{a}^{(l-1)} + b^{(l)}_j
\]

\[
\mathbf{z}^{(l)} = \mathbf{W}^{(l)T} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}
\]

\[
\mathbf{a}^{(l)} = f\big(\mathbf{z}^{(l)}\big)
\]
Trong đó:
\[
\mathbf{W}^{(l)} \in \mathbb{R}^{d^{(l-1)} \times d^{(l)}} 
\quad \text{: ma trận trọng số giữa layer } (l-1) \text{ và layer } l
\]

\[
\mathbf{b}^{(l)} \in \mathbb{R}^{d^{(l)} \times 1} 
\quad \text{: vector bias của layer } l
\]

\[
z^{(l)}_j = \mathbf{w}^{(l)T}_j \, \mathbf{a}^{(l-1)} + b^{(l)}_j 
\quad \text{: tổng trọng số (weighted sum) cho neuron } j \text{ ở layer } l
\]

\[
\mathbf{z}^{(l)} = \mathbf{W}^{(l)T} \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)} 
\quad \text{: vector tổng trọng số toàn layer } l
\]

\[
\mathbf{a}^{(l)} = f\big(\mathbf{z}^{(l)}\big) 
\quad \text{: output sau khi qua activation function } f
\]
Với lưu ý ở trên, ta chú ý vào phần \(z^{(l)}_j\) và \(a^{(l)}\). Ở phần \(z^{(l)}_j\) ta có thể hiểu giống như một biểu thức \(y = ax + b\), sau đó  \(a^{(l)}\) như là kết quả của một hàm f trong đó input đầu vào của f là vector \(z^{(l)}\) vừa tính được ở trước ở đây làm f ta được gọi là hàm activation, mục đích của hàm này là phá vỡ sự tuyến tính của \(z^{(l)}\) để mô hình có thể học được đối với những dữ liệu phức tạp hơn. Với MLP, mô hình có thể học trên các giá trị \(w\) mà mình vừa nêu ở phía bên trên thông qua quá trình lan truyền ngược (Back Propagation).<br>
<p> Chú ý rằng các chữ in hoa mà in đậm như \( \mathbf{W}^{(l)} \) là một ma trận, những kí tự viết in đậm chữ thường như \( \mathbf{z}^{(l)}_j \) là một vector và \( z^{(l)}_j \) là một biến như bình thường </p>

</div>

<div style="text-align: center; font-weight: bold; font-size: 20px"> III. Kolmogorov–Arnold Networks(KAN) </div>

<div>
Như đã giới thiệu qua ở phía trên, KAN được lấy cảm hứng từ Kolmogorov–Arnold với biểu thức toán học:
$$
f(\mathbf{x})=f(x_1, x_2, \ldots, x_n) = \sum_{q=1}^{2n+1} \phi_q \left( \sum_{p=1}^n \psi_{pq}(x_p) \right)
$$
Ví dụ trong trường hợp hàm có hai biến gọi là \( x_1, x_2 \) ta có function tương ứng như sau:
$$
f(x_1, x_2) = \sum_{q=1}^{5} \phi_q \left( \psi_{1q}(x_1) + \psi_{2q}(x_2) \right)
$$
Từ function ví dụ trên ta có thể thấy khi \( x_1 \) đi vào function \( \psi_{1q} \) tổng với \( x_2 \) đi vào function \( \psi_{2q} \) gọi là \( a \), rồi \( a \) này lại tiếp tục đi qua function \( \phi_q \), các bước đó được minh họa theo hình dưới đây:
<figure style="text-align: center;">
    <img src="{{ site.baseurl }}/assets/images/KanExample.png" alt="MLP Example" style="max-width: 100%; height: auto;">
    <figcaption><i>Nguồn: https://www.digitalocean.com/community/tutorials/kolmogorov-arnold-networks-kan-revolutionizing-deep-learning</i></figcaption>
</figure>
Từ ảnh ở phía trên chắc các bạn cũng đã hiểu được ý nghĩa của ảnh đó rồi. Tiếp theo, để cho thân thuộc hơn với MLP, tác giả đã cố gắng đưa biểu thức toán học này biểu diễn dưới dạng ma trận như sau:
$$
f(x) = \Phi_{\text{out}} \circ \Phi_{\text{in}} \circ \mathbf{x}
$$
trong đó:
\[
\mathbf{\Phi}_{\text{in}} =
\begin{pmatrix}
\psi_{1,1}(\cdot) & \cdots & \psi_{1,n}(\cdot) \\
\vdots & \ddots & \vdots \\
\psi_{2n+1,1}(\cdot) & \cdots & \psi_{2n+1,n}(\cdot)
\end{pmatrix},
\quad
\mathbf{\Phi}_{\text{out}} =
\bigl( \phi_{1}(\cdot) \ \cdots \ \phi_{2n+1}(\cdot) \bigr)
\]
Bạn thấy đó, với cái này thì lúc nào mạng của chúng ta cũng chỉ có 3 layer là input, hidden layer và output layer, lúc này, khi ta không muốn nhiều các layer hơn, ta có thể biến đổi đi một chút khi đó, ta sẽ có một matrix mới như sau:
\[
\Phi =
\begin{pmatrix}
\psi_{1,1}(\cdot) & \cdots & \psi_{1,n_{\text{in}}}(\cdot) \\
\vdots & \ddots & \vdots \\
\psi_{n_{\text{out}},1}(\cdot) & \cdots & \psi_{n_{\text{out}},n_{\text{in}}}(\cdot)
\end{pmatrix}
\]
Đây sẽ là weight của mỗi KAN Layer, trong đó \( n_{\text{in}} \) là số lượng input features ở layers trước còn \( n_{\text{out}} \) là số lượng node ở layer sau.<br>
Lúc này ta có công thức của KAN là:
\[
\text{KAN}(\mathbf{x}) = 
\big( \Phi_{L-1} \circ \Phi_{L-2} \circ \cdots \circ \Phi_1 \circ \Phi_0 \big) \mathbf{x}
\]

So sánh với công thức của MLP

\[
\text{MLP}(\mathbf{x}) = 
\big( \mathbf{W}_{L-1} \circ \sigma \circ \mathbf{W}_{L-2} \circ \sigma \circ \cdots \circ 
\mathbf{W}_1 \circ \sigma \circ \mathbf{W}_0 \big) \mathbf{x}
\]
Ta thấy một \( \Phi_{i} \) ở bên KAN tương đương với phép \( \sigma \circ \mathbf{W}_i \) ở biểu thức của MLP.
<p style="color: red;"> Như vậy là ở trên mình đã nói về góc nhìn dưới Kolmogorov–Arnold representation theorem và biến đổi đi một chút để ta có thể biểu diễn được một deep networks, tiếp theo ta sẽ tìm hiểu xem KAN được học như thế nào? </p>

<div>
Như đã giới thiệu ở trên, khác với MLP được học thông qua trọng số weight nối giữa các node ở layer trước với node ở layer sau, rồi đi qua activation cố định từ đó là input cho layer sau nữa, còn đối với KAN, ta sẽ học các activation của nó, chứ không cố định như trong MLP, dưới đây là cách làm của tác giả:

Tác giả định nghĩa một activation trong KAN như sau:
\[
\phi(x) = w \big( b(x) + \text{spline}(x) \big)
\]
Trong đó:
\[
b(x) = \text{silu}(x) = \frac{x}{1 + e^{-x}}
\]
trong hầu hết các trường hợp \( \text{spline}(x) \) được tham số hóa như một tổ hợp tuyến tính (linear combination) của B-spline sao cho:
\[
\text{spline}(x) = \sum_i c_i B_i(x)
\]
với \( c_i \) là các tham số ta có thể huấn luyện được, và \( w \) ta có thể là thừa vì nó có thể biểu diễn trong các tham số học nhưng tác giả vẫn đưa vào để kiểm soát độ tổng thể của activation function. Khởi tạo ban đầu là \( \text{spline}(x) \approx 0 \) và \( w \) được khởi tạo dựa theo Xavier, cái mà mình hay làm ngay cả trong MLP luôn. (Nếu bạn chưa biết về cách khởi tạo về Xavier thì bạn có thể đọc ở nơi khác nhưng đại khái nó là một cách khởi tạo để mô hình có thể học mượt mà hơn)

<p style="color: red;"> Nãy giờ bạn có thể thấy rằng B-spline được nhắc đến, nếu bạn chưa có kiến thức về B-spline này mình sẽ minh họa một chút cho bạn cảm thấy dễ hiểu hơn </p>

<figure style="text-align: center;">
    <img src="{{ site.baseurl }}/assets/images/B_spline.png" alt="MLP Example" style="max-width: 100%; height: auto;">
</figure>
B-spline xuất hiện trong bối cảnh khi mà ta có một tập các điểm datapoint(các điểm có màu đỏ) như hình trên, bây giờ để biểu diễn hình này, ta có thể nối chúng lại với nhau, nhưng như vậy các bạn nhìn nó không được mượt cho lắm, có nhiều đoạn gấp khúc, lúc này đường B-spline sinh ra để biểu diễn lại một đường cong trông mượt mà hơn(là đường màu xanh). Điều này rất có ích trong các bài toán về AutoCAD.<br>
Ta gọi các điểm màu đỏ là control point(các điểm điều khiển), điểm mấu chốt trong bài toán B-spline này là để vẽ ra được đường spline(màu xanh) thì hình đó được gộp lại bởi các hình nhỏ. Ví dụ như range của các điểm theo trục Ox là [1,14] thì ta có thể chia nhỏ cái này ra [1,2],[2,3],... và trong mỗi đoạn [1,2] sẽ vẽ một đường, gộp với đường ở đoạn [2,3] để ra được đường liền mạch như đường màu xanh ở hình trên.

<p> Lưu ý, so với một phương pháp khác Bezier curve, Bezier curve là ảnh hưởng toàn cục, nghĩa là các điểm control point đều có đóng góp vào đường Bezier, còn đối với B-spline, là local vì chỉ có một số điểm mới tác động đến phần của đường đó thôi chứ các điểm ở xa không tác động vào vùng đó. </p>
<p> Từ định nghĩa của hàm cơ sở, ta thấy rằng nó sẽ bằng 1 nếu t nằm trong vùng mình đã chia, còn lại là sẽ bằng 0.</p>

<h2>Công thức toán học của B-spline</h2>

<p>Giả sử ta có:</p>

<ul>
  <li>Một tập control points:
    $$P_0, P_1, \dots, P_n$$
  </li>
  <li>Bậc spline:
    $$k \quad (\text{thường dùng } k=3 \;\; \Rightarrow \;\; \text{cubic B-spline})$$
  </li>
  <li>Một dãy knot vector:
    $$U = \{ u_0, u_1, \dots, u_m \}, \quad m = n + k + 1$$
  </li>
</ul>

<p>Knot vector xác định "phạm vi ảnh hưởng" của mỗi control point.</p>

<hr>

<h3>Định nghĩa đường cong B-spline</h3>

$$
C(t) = \sum_{i=0}^{n} N_{i,k}(t)\, P_i, 
\quad t \in [u_k, \, u_{m-k}]
$$

<p>Trong đó:</p>
<ul>
  <li>$N_{i,k}(t)$ là <b>hàm cơ sở B-spline</b> bậc $k$.</li>
  <li>$P_i$ là các <b>control point</b>.</li>
</ul>

<hr>

<h3>Định nghĩa đệ quy của hàm cơ sở</h3>

$$
N_{i,0}(t) =
\begin{cases}
1, & \text{nếu } u_i \leq t < u_{i+1} \\
0, & \text{ngược lại}
\end{cases}
$$

$$
N_{i,k}(t) = 
\frac{t - u_i}{u_{i+k} - u_i}\, N_{i,k-1}(t) 
+ \frac{u_{i+k+1} - t}{u_{i+k+1} - u_{i+1}}\, N_{i+1,k-1}(t)
$$

<hr>

<p>Nãy giờ mình đã giới thiệu về B-spline, quay trở lại với cách học của KAN.</p>

<hr>
<h4>Ý tưởng chính của Residual Activation Functions</h4>

Họ không chỉ dùng $spline(x)$ đơn thuần, mà định nghĩa activation như sau:

$$
\phi(x) = w \Big( b(x) + \text{spline}(x) \Big)
$$

Trong đó:<br>

- $b(x)$: là một hàm cơ sở cố định, thường lấy là SiLU:
$$
b(x) = \text{silu}(x) = \frac{x}{1 + e^{-x}}
$$

Đây giống như một "activation an toàn" để tránh việc spline học sai làm model hỏng.<br>

- $\text{spline}(x)$: là phần spline được học:
$$
\text{spline}(x) = \sum_i c_i B_i(x)
$$

- $w$: hệ số scale (hệ số nhân toàn bộ activation). Về lý thuyết có thể hấp thụ vào $c_i$, nhưng họ vẫn giữ lại để dễ kiểm soát biên độ (giống batch norm hoặc layer scale).

Thay vì dùng spline "chay" (dễ khó tối ưu), họ dùng residual (giống skip connection):

$$
\text{activation}(x) = \text{spline}(x) + b(x) = \text{spline}(x) + \text{silu}(x)
$$

Như vậy, lúc mới khởi tạo, activation gần giống SiLU (ổn định), sau đó spline sẽ học thêm để tinh chỉnh.

<hr>

<h4>Khởi tạo và huấn luyện</h4>

Lúc khởi tạo, ta có:
$$
\text{spline}(x) \approx 0
$$
(tức ban đầu không đóng góp nhiều).

Hệ số $w$ được khởi tạo theo Xavier initialization (chuẩn như trong MLP).  

Kết luận: Ở thời điểm bắt đầu huấn luyện, activation gần như:
$$
\phi(x) \approx \text{silu}(x)
$$

→ giúp mô hình dễ hội tụ. Sau đó spline sẽ "nổi lên" dần trong quá trình học.

</div>

<hr>
<div>
<div style="text-align: center; font-weight: bold; font-size: 20px"> IV. KAN vs MLP</div>
Mình đã giới thiệu về KAN, cách KAN được tác giả giới thiệu và áp dụng vào trong các mô hình để thay thế cho MLP. Vậy tại sao KAN có thể là tương lai thay thế MLP ở đây?
<ol>
<li>
Đầu tiên, KAN vượt trội hơn MLP ở chỗ có thể giải thích một vấn đề gì đó dựa vào công thức toán học. Ta có thể thấy từ MLP không có tính diễn giải(interpretability) tốt, đầu tiên là vì tính phi tuyến (non-linear) chồng chất, mỗi tầng là một ánh xạ phi tuyến, khi chồng nhiều tầng thì output trở thành một hàm cực kì phức tạp. Còn đối với KAN thì mình có thể biểu diễn được kết quả thông qua các biểu thức toán học, và có thể thấy được phần nào, đoạn nào có tác động nhiều hơn vào kết quả (theo tính chất của B-spline phân đoạn ra vẽ rồi gộp hình lại với nhau).<br>
Để rõ hơn về mặt Interpretability, tác giả trong paper có đưa ra một model Training bằng KAN để học một function \( \exp(\sin(\pi x) + y^2) \), mình gen data dựa trên function trên để xem KAN có học được biểu diễn này không.
<figure style="text-align: center;">
    <img src="{{ site.baseurl }}/assets/images/EP.png" alt="Example Interpretable" style="max-width: 100%; height: auto;">
    <figcaption><i>Nguồn: https://arxiv.org/html/2404.19756v1</i></figcaption>
</figure>
<ol type="I">
<li>
Bắt đầu từ một KAN fully-connected có cấu trúc 
\([2,5,1]\) (tức là \(2 \; \text{input} \to 5 \; \text{hidden} \to 1 \; \text{output}\)). <br>

- Khi huấn luyện kèm regularization dạng sparsification, nhiều kết nối trở nên thưa thớt. <br>

- Quan sát cho thấy \( \tfrac{4}{5} \) neuron ẩn không có đóng góp đáng kể 
\(\;\Rightarrow\;\) có thể bỏ đi.
</li>
<li>
- Tiến hành tự động cắt bỏ (prune) các neuron ẩn không hữu ích. <br>

- Kết quả, chỉ còn lại \(1\) neuron duy nhất trong hidden layer, cấu trúc biến thành \([2,1,1]\). <br>

- Các hàm kích hoạt trong mạng lúc này có thể được diễn giải trực tiếp thành các 
hàm biểu tượng (symbolic functions), thay vì những tổ hợp phi tuyến khó hiểu như trong MLP. 
</li>
<li>
Sau khi prune xong, mình sẽ giữ lại được các thành phần chính, chính là ảnh sau Step 2 đã được biểu diễn trong hình ảnh, tiếp theo, tác giả cũng có một như viện nhận định xem hàm đó là hàm nào
<figure style="text-align: center;">
    <img src="{{ site.baseurl }}/assets/images/RedZone.png" alt="Example Zone" style="max-width: 100%; height: auto;">
    <figcaption><i>Nguồn: https://arxiv.org/html/2404.19756v1</i></figcaption>
</figure>
Đối với vùng khoanh đỏ trên, từ hàm số đó ta có thể dự đoán được đó là một hàm số của hàm sin, thì lúc này tác giả có viết sẵn một function cho chúng ta để dự đoán được đó là dựa trên hàm nào(các bạn hãy xem trong github được công bố cùng với paper của tác giả nha). Thì Step 3 của chúng ta là dự đoán các cái hàm đó á, 
</li>
<li>

Sau khi đã \( \text{symbolify} \) (tức là thay thế các hàm spline trong KAN bằng các hàm toán học quen thuộc như 
$\sin$, $\cos$, $e^x$), thì mạng không còn nhiều tham số tự do nữa.  

Các tham số còn lại chỉ là các tham số affine (các hệ số tỉ lệ, cộng trừ nhân chia tuyến tính).  

Người ta tiếp tục huấn luyện các tham số affine này.  

Khi thấy \( \textit{loss} \) giảm tới mức \( \textit{machine precision} \)(nghĩa là gần bằng $10^{-15}$, sai số rất nhỏ), 
thì có thể kết luận rằng mạng đã khớp chính xác công thức toán học gốc.  

</li>
<li>

Dùng thư viện \( \texttt{Sympy} \) để “rút gọn” và xuất ra công thức ký hiệu từ mạng đã \textit{symbolify}.  

Kết quả nhận được là:
\[
f(x,y) = 1.0 \, e^{\,1.0 y^2 + 1.0 \, \sin(3.14x)}
\]

Thực chất, công thức sinh dữ liệu ban đầu là:
\[
f(x,y) = e^{y^2 + \sin(\pi x)}
\]

Trong đó tác giả hiển thị $\pi \approx 3.14$ và chỉ giữ $2$ chữ số thập phân.
</li>
</ol>
</li>
<li>
Tiếp theo, chắc hẳn các bạn đã quen với thuật ngữ continue learning, nghĩa là sau khi mình training với một bộ data, sau một thời gian, mình có một tập data mới muốn mô hình của mình học tiếp những đặc chưng của data đó. Điều này có thể làm cho mô hình trở nên quên những data đã học trước đó và chú trọng hơn vào data hiện tại. Đó là điều đang xảy ra với MLP, còn với KAN, dường như KAN xử lý được vấn đề này. Vậy tại sao KAN lại làm được điều đó?. Đó là dựa vào tính chất đặc biệt của B-spline. Như đã giới thiệu ở trên, ta tưởng tượng đơn giản rằng trong tọa độ Oxy đã có các điểm data ban đầu, bây giờ ta thêm một số điểm thì điểm đó sẽ tác động vào B-spline nằm trên gần đoạn đó, chứ không tác động vào toàn bộ đường splines mà mình vẽ. Một ví dụ rõ ràng hơn ở dưới đây:
<figure style="text-align: center;">
    <img src="{{ site.baseurl }}/assets/images/CL.png" alt="Continue Learning" style="max-width: 100%; height: auto;">
    <figcaption><i>Nguồn: https://arxiv.org/html/2404.19756v1</i></figcaption>
</figure>
Ta chia ra 5 phase để huấn luyện data, với MLP, các bạn có thể thấy rằng đến phase nào thì MLP chỉ tập chung chú trọng vào data của phase đó mà quên mất các data đã được học ở phase trước, còn với KAN thì không.
</li>
<li>
Nói đến điểm tốt của KAN rồi ta sẽ nói đến điểm chưa tốt của KAN:
<ul>
<li>
Trước hết thì MLP đã được sử dụng xuyên suốt bao lâu nay rồi nên KAN vẫn cần nhiều thời gian hơn, cần cộng đồng phát triển hơn nữa để có thể khai thác được hết về KAN.
</li>
<li>
Về mặt thời gian huấn luyện, thì việc tính toán sline dạng hàm phức tạp thì làm cho chi phí tính toán nặng hơn MLP.
</li>
<li>
Về mặt hỗ trợ thì Pytorch/TensorFlow đang cõ hỗ trợ rất tốt cho MLP, còn KAN mới phát triển gần đây nên chưa có nhiều bên support.
</li>
</ul>
</li>
</ol>
</div>
<hr>
<h2>Reference</h2>
<a href="https://arxiv.org/abs/2404.19756v1" target="_blank" rel="noopener">
  KAN Paper
</a><br>
<a href="https://arxiv.org/abs/2007.15884" target="_blank" rel="noopener">
  Kolmogorov-Arnold representation theorem
</a><br>
</div>