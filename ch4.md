Linear Regression of an Indicator Matrix
----------------------------------------

若我們說，𝒢 有 *K* 個類別，為了表示該筆資料是何種類別，我們可以建立一個
*Y* 矩陣,
*Y* = (*Y*<sub>1</sub>, ..., *Y*<sub>*k*</sub>)<sub>*N* × *K*</sub>，也就是說*Y*矩陣中有*N*個*K*維的
row vectors。而根據線性迴歸模型，我們可得對 *Y*的估計為：

*Ŷ* = *X*(*X*<sup>*T*</sup>*X*)<sup> − 1</sup>*X*<sup>*T*</sup>*Y* = *X**B̂*

*B*<sub>(*p* + 1) × *K*</sub>：*p* 個 inputs 然後加上截距項。

或者是從另外一個觀點，也就是我們希望極小化*ŷ*跟 *y*的距離。
$$\\min\_{\\bf{B}}\\sum\_{i=1}^N\|\|y\_i-\[(1,x\_i^T) \\textbf{B} \]^T \|\|^2$$
也就是說，*f̂*(*x*)會分類到最接近的目標群(
*y*<sub>*i*</sub> = *t*<sub>*k*</sub>, *i**f* *g*<sub>*i*</sub> = *k*
)：

$$\\hat{G}(x)=\\mathop{\\arg\\min}\\limits\_{k}\|\|\\hat{f}(x)-t\_k\|\|^2$$
\#模擬Fig. 4.3¶
<a href="https://esl.hohoweiya.xyz/notes/LDA/sim-4-3/index.html" class="uri">https://esl.hohoweiya.xyz/notes/LDA/sim-4-3/index.html</a>

Linear Discriminant Analysis(LDA)
---------------------------------

在課本的2.4節，我們知道在做分類決策時，我們是在極大化某種後驗機率(posrerior
probability)，也就給定*X* = *x*時，找一個最大可能性的分類當作分析的結果。
也就是說： 我們令分類所做的預測 *Ĝ*(*x*) = 𝒢<sub>*k*</sub>
(𝒢<sub>*k*</sub>為其中一種分類)；
也就表示，當給定*X* = *x*的條件機率之下，𝒢<sub>*k*</sub>是最有可能的出象，換成數學的語言就是：
*P*(𝒢<sub>*k*</sub>\|*X* = *x*) = max<sub>*l*</sub>*P*(*G* = 𝒢<sub>*l*</sub>\|*X* = *x*)
（或者說是  = max<sub>*l*</sub>*P*(*G* = *l*\|*X* = *x*)）

有這個基本觀念後，我們設定資料屬於類別*k*的先驗機率為：*π*<sub>*k*</sub> = *P*(*G* = *k*)，而當然
$\\sum\_{k=1}^K\\pi\_k =1$。

而透過貝氏定理，我們可以得到以下關係：

$$P(G=k\|X=x)=\\frac{{\\color{Red} f\_k(x)}{\\color{Blue} \\pi\_k}}{{\\color{DarkOrange} \\sum\_{l=1}^k f\_l(x)\\pi\_l}}=\\frac{{\\color{Red} P(X\|G=k)}{\\color{Blue} P(G=k)} }{{\\color{DarkOrange} \\sum\_l P(X\|G=l)P(G=l)}}$$

根據上式，我們必須對*f*<sub>*k*</sub>(*x*)做一些假設，這也就是當資料是第*k*類時，*X*的機率密度函數。我們設定各分類的機率密度服從「**多變量常態分配(Multivariate
Normal Distribution)**」。

$$f\_k(x)=\\frac{1}{(2\\pi)^{\\frac{p}{2}}\|\\Sigma\_k\|^\\frac{1}{2}}e^{-\\frac{1}{2}(x-\\mu\_k)^T\\Sigma\_k^{-1}(x-\\mu\_k)}$$
要注意的是，在LDA的架構之下，所有類別的pdf均享有相同的共變數矩陣，也就是
*Σ*<sub>*k*</sub> = *Σ*, ∀*k*。

接著，我們就可以以去比較兩兩類別之間的後驗發生機率*P*(*G*\|*X*)，我們在此利用對數的良好性質來分析兩者關係，假設我們現在要探討類別*k*和類別*l*，誰的發生機率大呢？我們用下列關係式來表達：

$$\\begin{aligned}
\\ln\\frac{P(G=k\|X=x)}{P(G=l\|X=x)}&=\\ln \\frac{f\_k(x)\\pi\_k}{f\_l(x)\\pi\_l}=\\ln\\frac{\\pi\_k}{\\pi\_l}+\\ln {f\_k(x)}-\\ln{f\_l(x)}\\\\
&=\\ln\\frac{\\pi\_k}{\\pi\_l}-\\frac{1}{2}(x-\\mu\_k)^T\\Sigma^{-1}(x-\\mu\_k)+\\frac{1}{2}(x-\\mu\_l)^T\\Sigma^{-1}(x-\\mu\_l)\\\\
&=\\ln\\frac{\\pi\_k}{\\pi\_l}-\\frac{1}{2}(\\mu\_k+\\mu\_l)^T\\Sigma^{-1}(\\mu\_k-\\mu\_l)+x^T\\Sigma^{-1}(\\mu\_k-\\mu\_l)
\\end{aligned}$$

注意!這樣良好的線性性質是來自於我們假設兩個分類具有**相同的**共變異數矩陣。若我們對類別*k*和類別*l*的分界線感興趣，其分界線就是位在兩者機率密度相等之處，也就是當上式**等於0**時。

透過相同的想法，我們可以建立一個**線性判別函數(linear discriminant
function)***δ*<sub>*k*</sub>(*x*)，來決定該資料應被分配到哪一個類別，也就是$G(x)=\\mathop{\\arg\\max}\\limits\_{k}\\delta\_k(x)$。
線性判別函數如下所示：
$$\\delta\_k(x)=x^T\\Sigma^{-1}\\mu\_k-\\frac{1}{2}\\mu\_k^T \\Sigma^{-1}\\mu\_k +\\ln \\pi\_k$$

線性判別函數的推導來自以下的成比例關係：
$$\\begin{aligned}
P(G=k\|X=x)&\\propto f\_k(x)\\pi\_k \\\\
&\\propto -\\frac{1}{2}(x-\\mu\_k)^T \\Sigma^{-1} (x-\\mu\_k)+\\ln \\pi\_k = -\\frac{1}{2}x^T\\Sigma^{-1}x+x^T\\Sigma^{-1}\\mu\_k-\\frac{1}{2}\\mu\_k^T \\Sigma^{-1}\\mu\_k +\\ln \\pi\_k\\\\
&\\propto  x^T\\Sigma^{-1}\\mu\_k-\\frac{1}{2}\\mu\_k^T \\Sigma^{-1}\\mu\_k +\\ln \\pi\_k \\equiv  \\delta\_k(x)
\\end{aligned}$$

正如古典的統計分析一樣，我們並無法知道母體分配的參數，故我們在進行統計學習時，則選擇使用**訓練集的資料來估計母體參數**。

$$\\begin{aligned}
&\\hat{\\pi\_k}=N\_k/N \\\\
 &\\hat{\\mu\_k}=\\sum\_{g\_i=k}x\_i/N\_k \\\\
 &\\hat{\\Sigma}=\\sum\_{k=1}^K\\sum\_{g\_i=k}(x\_i-\\hat{\\mu\_k})(x\_i-\\hat{\\mu\_k})^T/(N-K)
\\end{aligned}$$
其中 *N*<sub>*k*</sub> 是類別*k*在訓練集中的數量(observations)。

### 2-class LDA (from Ex. 4.2)

> Suppose we have features *x* ∈ ℝ<sup>*p*</sup>, a two-class response,
> with class sizes *N*<sub>1</sub>, *N*<sub>2</sub>, and the target
> coded as  − *N*/*N*<sub>1</sub>, *N*/*N*<sub>2</sub>.

> 1.  Show that the LDA rule classifies to class 2 if
>     $$x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1}) \> \\frac{1}{2}(\\hat{\\mu\_2}+\\hat{\\mu\_1})^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1})-\\ln(N\_2/N\_1)$$
>     and class 1 otherwise.

*Sol:*

在二元的分類中，我們可以回溯到先前講到的log-odds的觀念，也就是建立此式來做比較(改寫自課本式4.9)：
$$\\begin{aligned}
\\ln\\frac{P(G=2\|X=x)}{P(G=1\|X=x)}
=\\ln\\frac{\\pi\_2}{\\pi\_1}-\\frac{1}{2}(\\mu\_2+\\mu\_1)^T\\Sigma^{-1}(\\mu\_2-\\mu\_1)+x^T\\Sigma^{-1}(\\mu\_2-\\mu\_1)
\\end{aligned}$$
根據對數性質，若此式**大於0**，就表示*P*(*G* = 2\|*X* = *x*)的機率相對於*P*(*G* = 1\|*X* = *x*)來得高，故我們自然會將其分類到
class 2。
而我們根據訓練集對上式做估計並且做些許代數運算即可得到決策的函數：
$$x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1}) \> \\frac{1}{2}(\\hat{\\mu\_2}+\\hat{\\mu\_1})\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1})-\\ln(N\_2/N\_1)$$

> 1.  Consider minimization of the least squares criterion
>     $$\\sum^N\_{i=1}(y\_i-\\beta\_0-x\_i^T\\beta)^2$$
>     Show that the solution *β̂* satisfies

$$\[(N-2)\\hat{\\Sigma}+N\\hat{\\Sigma}\_B\]\\beta=N(\\hat{\\mu\_2}-\\hat{\\mu\_1})$$
where,
$$\\hat{\\Sigma}\_B=\\frac{N\_1N\_2}{N^2}(\\hat{\\mu\_2}-\\hat{\\mu\_1})(\\hat{\\mu\_2}-\\hat{\\mu\_1})^T$$

*Sol:*

首先，我們知道 *N* = *N*<sub>1</sub> + *N*<sub>2</sub>。
而這個線性迴歸方程包含常數項，則根據 Normal equation
可以得到對$\\hat{\\beta\_0}$和$\\hat{\\beta\_1}$的估計：

$$ \\begin{bmatrix}
\\hat{\\beta\_0}\\\\ 
\\hat{\\beta}
\\end{bmatrix}=(X^TX)^{-1}X^Ty $$

其中 **designed matrix** 可這樣表示：(*X*裡面有$\\bf{1}$向量)

$$X^TX=\\begin{bmatrix}
N & \\sum\_{i=1}^N x\_i^T\\\\ 
\\sum\_{i=1}^N x\_i & \\sum\_{i=1}^Nx\_ix\_i^T 
\\end{bmatrix}$$
其中
$$\\sum\_{i=1}^Nx\_i =\\sum\_{i=1}^{N\_1}x\_i+\\sum\_{i=1}^{N\_2}x\_i = N\_1\\hat{\\mu\_1}+ N\_2\\hat{\\mu\_2}$$
又知
$$\\begin{aligned}
\\hat{\\Sigma} &= \\frac{1}{N-2}\\sum\_{k=1}^2\\sum\_{g\_i=k}(x\_i-\\hat{\\mu\_k})(x\_i-\\hat{\\mu\_k})^T \\\\
&=\\frac{1}{N-2}\[\\sum\_{g\_i=1}x\_ix\_i^T-N\_1 \\mu\_1 \\mu\_1^T+\\sum\_{g\_i=2}x\_ix\_i^T-N\_2\\mu\_2 \\mu\_2^T\]
\\end{aligned}$$
因此
$$\\sum\_{i=1}^Nx\_ix\_i^T = (N-2)\\hat{\\Sigma}+N\_1 \\mu\_1 \\mu\_1^T+N\_2\\mu\_2 \\mu\_2^T$$
若我們將屬於類別1的資料放在矩陣的前*N*<sub>1</sub>個，而類別2的資料*N*<sub>2</sub>個緊接在後，*X*<sup>*T*</sup>*y*可以寫成：

$$ X^Ty= \\begin{bmatrix}
1 &\\cdots  & 1 &1  & \\cdots &1 \\\\ 
x\_1 &\\cdots  &x\_{N\_1}  &x\_{N\_1+1}  &\\cdots  & x\_{N\_1+N\_2}
\\end{bmatrix}
\\begin{bmatrix}
-N/N\_1\\\\ 
\\vdots\\\\ 
-N/N\_1\\\\ 
N/N\_2\\\\ 
\\vdots\\\\ 
N/N\_2
\\end{bmatrix} =\\begin{bmatrix}
0 \\\\
-N\\mu\_1+N\\mu\_2
\\end{bmatrix}$$

整理先前計算出的這些東西代入 Normal equation，整理後可得：

(註：$\\beta\_0=(-\\frac{N\_1}{N}\\mu\_1^T-\\frac{N\_2}{N}\\mu\_2^T)\\beta$)

$$(N\_1\\mu\_1+N\_2\\mu\_2)(-\\frac{N\_1}{N}\\mu\_1^T-\\frac{N\_2}{N}\\mu\_2^T)\\beta+((N-2)\\hat{\\Sigma} +N\_1\\mu\_1\\mu\_1^T+N\_2\\mu\_2\\mu\_2^T)\\beta=N(\\mu\_2-\\mu\_1)$$

經過一些運算，並引入題目對 $\\hat{\\Sigma\_B}$的定義，即可得：
$$\[(N-2)\\hat{\\Sigma}+N\\hat{\\Sigma}\_B\]\\beta=N(\\hat{\\mu\_2}-\\hat{\\mu\_1})$$

> 1.  Hence show that *Σ̂*<sub>*B*</sub>*β* is in the direction
>     $(\\hat{\\mu\_2} - \\hat{\\mu\_1})$ and thus
>     $$\\hat{\\beta}\\propto \\hat{\\Sigma}^{-1} (\\hat{\\mu\_2} - \\hat{\\mu\_1})$$
>     Therefore the least-squares regression coefficient is identical to
>     the LDA coefficient, up to a scalar multiple.

*Sol:*

若我們直接計算 *Σ̂*<sub>*B*</sub>*β*，由於
$(\\hat{\\mu\_2}-\\hat{\\mu\_1})^T \\beta$ 內積後是一個scalar。也就是

$$\\hat{\\Sigma}\_B \\beta=\\frac{N\_1N\_2}{N^2}(\\hat{\\mu\_2}-\\hat{\\mu\_1}){\\color{Red} (\\hat{\\mu\_2}-\\hat{\\mu\_1})^T \\beta}=\\frac{N\_1N\_2}{N^2}(\\hat{\\mu\_2}-\\hat{\\mu\_1}){\\color{Red} c}$$

換句話說，*Σ̂*<sub>*B*</sub>*β*和$(\\hat{\\mu\_2}-\\hat{\\mu\_1})$差了${\\color{Red} \\frac{N\_1N\_2}{N^2}c}$倍。又這些常數都為正，故得知：
$$ \\hat{\\beta} \\propto \\hat{\\Sigma}^{-1} (\\hat{\\mu\_2}-\\hat{\\mu\_1}) $$

此即表示線性迴歸與LDA其實是有相像的運算邏輯。

> 1.  Show that this result holds for any (distinct) coding of the two
>     classes.

*Sol:*

在上一題中，我們並沒有特別對 *t*<sub>*k*</sub>
有特別假設，這樣的性質是來自於一開始對*Σ̂*<sub>*B*</sub> 的設計，並且與
*β* 內積後出現純數而推演出這樣的比例關係。

> 1.  Find the solution $\\hat{\\beta\_0}$ (up to the same scalar
>     multiple as in (c), and hence the predicted value
>     $\\hat{f}(x)=\\hat{\\beta\_0}+x^T\\hat{\\beta}$. Consider the
>     following rule: classify to class 2 if *f*(*x*) \> 0and class 1
>     otherwise. Show this is **not** the same as the LDA rule unless
>     the classes have equal numbers of observations.

*Sol:*

在先前的討論，我們知道：
$$\\beta\_0 = -\\frac{1}{N}(N\_1\\hat{\\mu}^T\_1+N\_2\\hat{\\mu}^T\_2)\\hat{\\beta}$$

我們可改寫 *f̂*(*x*) 如下：

$$\\begin{aligned}
\\hat{f}(x)&=-\\frac{1}{N}(N\_1\\hat{\\mu}^T\_1+N\_2\\hat{\\mu}^T\_2-Nx^T)    \\hat{\\beta }\\\\
&=-\\frac{1}{N}(N\_1\\hat{\\mu}^T\_1+N\_2\\hat{\\mu}^T\_2-Nx^T) {\\color{Red} \\lambda \\hat{\\Sigma}^{-1}(\\hat{\\mu\_2}-\\hat{\\mu\_1})}
\\end{aligned}$$

經過展開，並重整，則可得到下式：

$$x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1}) \> \\frac{1}{N}(N\_2\\hat{\\mu\_2}+N\_1\\hat{\\mu\_1})^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1})$$
若 *N*<sub>1</sub> = *N*<sub>2</sub>
，則*N*<sub>1</sub>/*N* = *N*<sub>2</sub>/*N* = 1/2，而ln (*N*<sub>2</sub>/*N*<sub>1</sub>) = ln 1 = 0

這樣的決策準則就跟LDA一樣了！

Quadratic Discriminant Analysis (QDA)
-------------------------------------

還記得先前在推導 LDA
時除了常態假設之外，還有一個變異數矩陣相等的假設，若我們放寬這個假設，這個方法就變成
QDA ，Quadratic
的部分就是來自於多變量常態分配中在*e*上的*x*二次式不會被對消。

二次判別函數(Quadratic Discriminant function)如下：

$$\\delta\_k(x)=-\\frac{1}{2}\\ln\|\\Sigma\_k\|-\\frac{1}{2}(x-\\mu\_k)^T\\Sigma\_k^{-1}(x-\\mu\_k)+\\ln \\pi\_k$$

個類別的分界線可以這樣表示

{*x*\|*δ*<sub>*k*</sub>(*x*) = *δ*<sub>*l*</sub>(*x*)}
QDA 有參數太多的缺點。

Regulized Discriminant Analysis (RDA)
-------------------------------------

![](https://esl.hohoweiya.xyz/img/04/fig4.7.png)

123
