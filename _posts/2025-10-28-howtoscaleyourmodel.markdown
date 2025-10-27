---
layout: my-post
title: "How to scale your models?"
date: 2025-10-28

summary: "Let's learn about how TPUs/GPUs have high impact with your model's parameter."
---

<div id="birthday-section" class="center">
  <input type="date" id="birthday-input" />
  <button id="save-birthday">Send</button>
  <p id="birthday-message"></p>
</div>

<div id="main-content">
<div style="text-align: center; font-weight: bold; font-size: 20px"> Contents </div>
<div style="font-weight: bold; font-size: 18px">I. Introductions </div>
<div style="font-weight: bold; font-size: 18px">II. Intro to Rooflines </div>
<div style="font-weight: bold; font-size: 18px">III. All about TPUs </div>
<hr>

<div>
    <ol type="I">
        <li style="color:red;">Introductions</li>
            <p>
                Training LLMs is often more difficult than in the old days because the number of parameters is increasing day by day. With the growth of parameters, fine-tuning has also become more challenging since there often isn’t enough VRAM in TPUs or GPUs to store everything required for the training and inference processes. Therefore, understanding how hardware devices actually impact your model parameters is a very important part if you want to scale your model effectively.
                <br>
                <br>
                In this blog, I will dive into TPUs — one of Google’s products designed primarily for training models. In the next blog, I will cover GPUs. Let’s get started!
            </p>
        <hr>
        <li style="color:red"> Intro to Rooflines </li>
        <p>
            <ol>
                <li>
                    First of all, when we run algorithms on hardware, we need to care about three things: how fast our computer can do math (it's just like an complexity of the code for person who know data structure and algorithms), the bandwidth available for moving data around (bytes/s), and the total memory availabel to store data (bytes).
                </li>
                <li>
                    When you use your data structure, a lot of algorithms, you often try to make time do your problem is lowest as you can, one algorithms can take 50ms, or 50s or 5ms. What make you can estimate your time to run your algorithms?
                    <br>
                    <p><span style="font-weight:bold;"> Computation:</span> A deep learning model is effectively a bunch of matrix multiplication, each composed of floating-point multiplication and addition operations (FLOPs). Our accelerator speed determines how long these take to compute is</p>
                    \[
                        T_{\text{math}} = \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}}
                    \]
                    With <br>
                    <ul>
                        <li>\( T_{\text{math}} \) is time we need to run our algorithms</li>
                        <li> \( \text{Computation FLOPs} \) is the total FLOPs of your algorithms. </li>
                        <li>\( \text{Accelerator FLOPs/s} \) is the speed of hardware can process in one second.</li>
                    </ul>
                    <br>
                    For instance, an NVIDIA H100 can perform about 9.89e14 bfloat161 FLOPs/s while a TPU v6e can perform 9.1e14 FLOPs/s.2 That means doing 1e12 FLOPs on an H100 will take (roughly) 1e12 / 9.89e14 = 1.01ms and 1e12 / 9.1e14 = 1.1ms on a TPU v6e.
                    <p><span style="font-weight:bold;"> Communication within a chip:</span> Within and accelerator, tensors need to be transferred between on-chip memory(HBM) and compute cores. Each device has difference of bytes can transfer per second </p>
                    <p><span style="font-weight:bold;"> Communication between chips:</span> As you know, the TPUs/GPUs not only have one chip, it has many chips in one device, in this case, when we distribute a model across multiple accelerators, tensors frequently need to transferred between them. There are often a few options to transfer (ICI, DCN, PCIe), each option with different bandwidths. </p>
                    \[
                        T_{\text{comms}} = \frac{\text{Communication Bytes}}{\text{Network/Memory Bandwidth Bytes/s}}
                    \]
                    With <br>
                    <ul>
                        <li>\( T_{\text{comms}} \) is time we need to transfer your data.</li>
                        <li> \( \text{Communication Bytes} \) is the total bytes we need to transfer. </li>
                        <li>\( \text{Network/Memory Bandwidth Bytes/s} \) is the speed of hardware can transfer data in one second.</li>
                    </ul>
                    <br>
                    Let's limit the range of time we need to run an algorithms. Generally:
                    \[
                        T_{\text{lower}} = \max(T_{\text{math}}, T_{\text{comms}}) \tag{1}
                    \]
                    \[
                        T_{\text{upper}} = T_{\text{math}} + T_{\text{comms}} \tag{2}
                    \]
                    Computation within a single chip can be overlapped with communication within a chip and between chips.
                    In case overlapped, we can estimaste the lower time we need is equation (1). Besides, the maximum time is both process is actually do sequentily, so we have upper time is equation (2).
                    <br>
                    We assume that we can perfectly overlap communication and computaion, when \( T_\text{math}  > T_\text{comms} \), we call this being <span style="font-weight:bold;">"compute-bound"</span>. The opposite case is called <span style="font-weight:bold;">"communication-bound"</span>. 
                    <p><span style="font-weight:bold;"> Definition:</span> We will introduce the arithmetic intensity of an algorithm is given by the ratio of the total FLOPs it performs to the number of bytes it needs to communicate — either within a chip or between chips.  </p>
                    \[
                        \text{Arithmetic Intensity} = \frac{\text{Computation FLOPs}}{\text{Communication Bytes}}
                    \]
                    Arithmetic intensity represents the number of FLOPs performed per byte of data transferred. When arithmetic intensity is high, \( T_\text{math} \) dominates \( T_\text{comms} \), meaning most of the available computational capacity is effectively utilized. Conversely, when arithmetic intensity is low, more time is spent on communication, leading to underutilized FLOPs.
                    \[
                        T_{\text{math}} > T_{\text{comms}}
                        \iff 
                        \frac{\text{Computation FLOPs}}{\text{Accelerator FLOPs/s}} 
                        > 
                        \frac{\text{Communication Bytes}}{\text{Bandwidth Bytes/s}}
                    \]
                    \[
                        \iff 
                        \frac{\text{Computation FLOPs}}{\text{Communication Bytes}} 
                        > 
                        \frac{\text{Accelerator FLOPs/s}}{\text{Bandwidth Bytes/s}}
                    \]
                    \[
                        \iff 
                        \text{Intensity(Computation)} > \text{Intensity(Accelerator)}
                    \]
                    The quantity \( \text{Intensity(Accelerator)} \) is the arithmetic intensity at which our accelerator achieves its peak FLOPs/s. Suppose in TPU v5e MXU, TPU can perform 1.97e16 FLOPs/s and load 8.2e11 bytes/s from HBM, it mean \( \text{Intensity(Accelerator)} \) = 240 FLOPs/byte. That means if an algorithm has lower arithmetics intensity than 240 FLOPs/byte, it mean the most time hardware (TPU/GPU) wait in transfer data process than calculation (in other words, it's bandwidth-bound).
                    <br>
                    Let's take a look an example:
                    <br>
                    To compute the dot product of two vectors in bfloat16, x . y: bf16[N] . bf16[N] -> bf16[1], we need to load x and y from memory, each of which has 2 * n = 2N Bytes, perform N multiplication and N - 1 additions, last we write 2 bytes back to HBM:
                    \[
                        \text{Intensity(dot product)} 
                        = 
                        \frac{\text{Total FLOPs}}{\text{Total Bytes}}
                        = 
                        \frac{N + N - 1}{2N + 2N + 2}
                        = 
                        \frac{2N - 1}{4N + 2}
                        \rightarrow 
                        \frac{1}{2}
                    \]
                    <br>
                    We can see that \( \frac{1}{2} \) is lower than 240 => this operation is limit by transfer data (bandwidth-bound).
                </li>
                <li>
                    Network communication
                </li>
                <p>
                    All the things we've discussed so far all within a single chip. In fact most of case we care about involve communication between chips: usually matrix multiplications that involve matrices sharded across multiple TPUs.
                    <br>
                    For instance, we want to multiply two big matrices X = bfloat16[B, D] @ Y = bfloat16[D, F]. Suppose we have 2 TPUs/GPUs. we can multiply half of each matrix on each TPU
                    \[
                    A = X[:,\, :D/2] \; @ \; Y[:D/2,\, :] \quad \text{(on TPU 0)}
                    \]
                    \[
                    B = X[:,\, D/2:] \; @ \; Y[D/2:,\, :] \quad \text{(on TPU 1)}
                    \]
                    then copy the result to the other TPU and add them together. Suppose our TPU can copy 4.5e10 bytes in each direction and perform 1.97e14 FLOPs/s on each chip. So \( T_\text{math} \) in this case will be:
                    \[
                    T_{\text{math}} = 
                    \frac{2BDF}{2 \cdot \text{Accelerator FLOPs/s}}
                    = 
                    \frac{BDF}{1.97 \times 10^{14}}
                    \]
                    Now, \( T_\text{comms} \) is equal:
                    \[
                    T_{\text{comms}} = 
                    \frac{2BF}{\text{Network Bandwidth}}
                    = 
                    \frac{2BF}{4.5 \times 10^{10}}
                    \]
                    We become "Compute-bound" when D > 8755 depends on two above equations.
                    <br>
                </p>
            </ol>
        </p>
        <hr>
        <li style="color:red;">All about TPUs</li>
        <p>
            In this part, let's take a look for TPU structure. Moreover, TPU is basically a compute core that specializes in matrix multiplication. Here is diagram:
            <figure style="text-align: center;">
                <img src="{{ site.baseurl }}/assets/images/TPUStructure.png" alt="Example TPU Structure" style="max-width: 100%; height: auto;">
                <figcaption><i>Nguồn: https://jax-ml.github.io/scaling-book/tpus/</i></figcaption>
            </figure>
            As you see the picture above, let's dive into TPUs structure. In one TPU core has three key units:
            <ul>
                <li>
                    The MXU (Matrix Multiply Unit) the core of TPU. It performs multiply matrix with support of systolic array. In each TPUs, MXU has difference performance, eg: it about 5e13 bf16 FLOPs/s MXU at 1.5GHz on TPU v5e.
                </li>
                <li>
                    The VPU (Vector Processing Unit) it carries out basic mathematical operations such as ReLU activations, pointwise addition, or multiplication between vectors. It also performs reduction operations like summation.
                </li>
                <li>
                    VMEM (Vector Memory) it communications with VPU and MXU, it so close to the compute units. This size is smaller than HBM, but it has higher bandwidth to the MXU. VMEM it like and L1/L2 cache on CPUs.
                </li>
                <li>
                    HBM (High Bandwidth Memory) is used to stores tensors for use by the TensorCore. HBM usually has capacity on the order of tens of gigabytes
                    <ul>
                        <li>
                            When we performs a computations, tensors are streamed out of HBM through VMEM into the MXU and the result is written back to HBM
                            <figure style="text-align: center;">
                                <img src="{{ site.baseurl }}/assets/images/pointwise-product.gif" alt="Example TPU Structure" style="max-width: 100%; height: auto;">
                                <figcaption><i>Nguồn: https://jax-ml.github.io/scaling-book/tpus/</i></figcaption>
                            </figure>
                        </li>
                    </ul>
                </li>
            </ul>
            As introduce above, it is one core of TPUs, but actually, in single chip, it usually has more than one core in chip.
            <figure style="text-align: center;">
                <img src="{{ site.baseurl }}/assets/images/cores.png" alt="Example TPU Structure More Than One Core" style="max-width: 100%; height: auto;">
                <figcaption><i>Nguồn: https://jax-ml.github.io/scaling-book/tpus/</i></figcaption>
            </figure>
            Let's explore about TPU Networking. <br>
            In fact, it does'nt only have one chip, we have more than one chip, it was called a Pod. In a Pod, Chips are connected to each other through the ICI (inter-chip interconnects) network. In the image below, ICI connects the 4 nearest neightbors (with edge links to form a 2D torus).
            <figure style="text-align: center;">
                <img src="{{ site.baseurl }}/assets/images/ici-wraparound.png" alt="ICI with 2D torus" style="max-width: 100%; height: auto;">
                <figcaption><i>Nguồn: https://jax-ml.github.io/scaling-book/tpus/</i></figcaption>
            </figure>
            In conclusion, in TPU (Tensor Processing Unit), about hardware structure, the levels hierarchical ascending is:
            <ol>
                <li>
                    Core
                </li>
                The core is the smallest computational unit of a TPU. <br>
                Each core includes:
                <ul>
                    <li>
                        A Matrix Multiply Unit (MXU) for large-scale matrix operations (the heart of neural network computation).
                    </li>
                    <li>
                        Vector and Scalar Units for smaller mathematical operations.
                    </li>
                    <li>
                        A Unified Buffer (on-chip memory) for temporary tensor storage.
                    </li>
                </ul>
                All these components communicate through an on-chip data bus, enabling extremely low-latency data movement within the core.
                <li>
                    Chip
                </li>
                A TPU chip integrates multiple cores on a single silicon die. <br>
                Cores within a chip are connected through a high-speed on-die interconnect, allowing them to share tensors efficiently.
                <li>
                    Device / Board
                </li>
                A TPU board (or device) contains several TPU chips. <br>
                For example:
                <ul>
                    <li>
                        TPU v3 board: 4 chips → 8 cores total.
                    </li>
                    <li>
                        Each board has its own power delivery, cooling system (liquid cooling from v3 onward), and a PCIe or NVLink connection to a host CPU.
                    </li>
                </ul>
                The host CPU handles:
                <ul>
                    <li>
                        Loading models and data into TPU memory,
                    </li>
                    <li>
                        Scheduling computations,
                    </li>
                    <li>
                        Collecting and aggregating results.
                    </li>
                </ul>
            </ol>
            <table>
                <thead>
                    <tr>
                        <th>Level</th>
                        <th>Contains</th>
                        <th>Connected</th>
                        <th>Purpose</th>
                    </tr>
                </thead>
                <tbody>
                    <tr>
                        <td><strong>Core</strong></td>
                        <td>MXU, Vector/Scalar Units, Buffer</td>
                        <td>On-chip bus</td>
                        <td>Basic tensor computation</td>
                    </tr>
                    <tr>
                        <td><strong>Chip</strong></td>
                        <td>Multiple cores</td>
                        <td>On-die interconnect</td>
                        <td>Coordinate computation between cores</td>
                    </tr>
                    <tr>
                        <td><strong>Board</strong></td>
                        <td>Multiple chips</td>
                        <td>High-speed internal links</td>
                        <td>Interface with CPU host</td>
                    </tr>
                    <tr>
                        <td><strong>Pod</strong></td>
                        <td>Many boards</td>
                        <td>ICI (Inter-Chip Interconnect)</td>
                        <td>Large-scale parallel training</td>
                    </tr>
                    <tr>
                        <td><strong>SuperPod / Data Center</strong></td>
                        <td>Multiple Pods</td>
                        <td>DCN (Data Center Network)</td>
                        <td>Global-scale distributed training</td>
                    </tr>
                </tbody>
            </table>
            <p>
                Communication performance is constrained by different network bandwidths, ranked by speed as follows:
            </p>
            <ul>
                <li><strong>HBM bandwidth:</strong> The data transfer rate between a TensorCore and its connected High Bandwidth Memory (HBM).</li>
                <li><strong>ICI bandwidth:</strong> The communication link between a TPU chip and its nearest four or six neighboring chips.</li>
                <li><strong>PCIe bandwidth:</strong> The connection speed between a CPU host and its attached tray or set of chips.</li>
                <li><strong>DCN bandwidth:</strong> The bandwidth across multiple CPU hosts, generally those not linked directly through ICI.</li>
            </ul>
            <p>
                Weight matrices need to be padded to at least size 128 (256 on TPU v6) in both dimensions to fill up the MXU (in fact, smaller axes are padded to 128).
            </p>
            <p>
                Lower precision matrix multiplication tends to be faster. TPUs can do int8 or int4 FLOPs roughly 2x/4x faster than bfloat16 FLOPs for generations that support it. VPU operations are still performed in fp32.
            </p>
        </p>
        In next blog, i will how to share data in multiple TPUs, the cost of each ways and how to appli it in transformer and training process.
        <hr>
            Reference <br>
        <a href="https://jax-ml.github.io/scaling-book/" target="_blank" rel="noopener">
            How To Scale Your Model
        </a><br>
    </ol>
</div>
</div>

<script type="module">
import { initializeApp } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-app.js";
import { getFirestore, doc, getDoc } from "https://www.gstatic.com/firebasejs/11.0.1/firebase-firestore.js";

const firebaseConfig = {
  apiKey: "AIzaSyC9gr2sjKXdqDNS8s2WUat-EObLeUUEAyM",
  authDomain: "my-secret-text.firebaseapp.com",
  projectId: "my-secret-text",
}

const app = initializeApp(firebaseConfig);
const db = getFirestore(app);

document.addEventListener("DOMContentLoaded", function () {
  const input = document.getElementById("birthday-input");
  const button = document.getElementById("save-birthday");
  const message = document.getElementById("birthday-message");
  const mainContent = document.getElementById("main-content");
  const birthdaySection = document.getElementById("birthday-section");

  button.addEventListener("click", async function () {
    const birthday = input.value;
    if (!birthday) {
      message.textContent = "Vui lòng nhập ngày sinh hợp lệ!";
      return;
    }

    const [year, month, day] = birthday.split("-").map(Number);

    console.log("Year:", year, "Month:", month, "Day:", day);

    if (day === 20 && month === 2 && year === 2004) {
      const notice = document.createElement("div");
      notice.style = "text-align:center; font-weight:bold; color:#ff6600; margin:20px;";

      // 🔥 Lấy dữ liệu từ Firestore
      try {
        const ref = doc(db, "secrets", "text");
        const snap = await getDoc(ref);
        if (snap.exists()) {
          notice.textContent = snap.data().message;
        } else {
          notice.textContent = "Không tìm thấy dữ liệu!";
        }
      } catch (err) {
        console.error("Lỗi khi đọc Firestore:", err);
        notice.textContent = "Không thể tải dữ liệu!";
      }

      mainContent.prepend(notice);
    }

    birthdaySection.style.display = "none";
    mainContent.style.display = "block";
  });
});
</script>
