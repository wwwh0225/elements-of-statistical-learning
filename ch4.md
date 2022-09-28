---
title: "Linear_Methods_for_Classification"
author: "Eric Huang"
output: 
  html_document:
    keep_md: true
#output: #會是.md
 # md_document:
  #  variant: markdown_github
#會變成圖片
#output:  
 # github_document:
#    pandoc_args: --webtex 
---



## Linear Regression of an Indicator Matrix
若我們說，$\mathcal{G}$ 有 $K$ 個類別，為了表示該筆資料是何種類別，我們可以建立一個 $Y$ 矩陣, $Y=(Y_1,...,Y_k)_{N \times K}$，也就是說 $Y$ 矩陣中有 $N$ 個 $K$ 維的 row vectors。而根據線性迴歸模型，我們可得對 $Y$ 的估計為：

$$ \hat{Y}  = X(X^TX)^{-1}X^TY=X \hat{B} $$
 
$B_{(p+1)\times K}$：$p$ 個 inputs 然後加上截距項。

或者是從另外一個觀點，也就是我們希望極小化 $\hat{y}$ 跟 $y$ 的距離。
$$\min_{\bf{B}}\sum_{i=1}^N||y_i-[(1,x_i^T) \textbf{B} ]^T ||^2$$
也就是說， $\hat{f}(x)$ 會分類到最接近的目標群( $y_i=t_k,\ if\ g_i = k$ )：


$$\hat{G}(x)=\mathop{\arg\min}\limits_{k}||\hat{f}(x)-t_k||^2$$
#模擬Fig. 4.3¶
https://esl.hohoweiya.xyz/notes/LDA/sim-4-3/index.html

## Linear Discriminant Analysis(LDA)

在課本的2.4節，我們知道在做分類決策時，我們是在極大化某種後驗機率(posrerior probability)，也就給定 $X=x$ 時，找一個最大可能性的分類當作分析的結果。
也就是說：
我們令分類所做的預測 $\hat{G}(x)=\mathcal{G}_k$  ( $\mathcal{G}_k$ 為其中一種分類)；
也就表示，當給定 $X=x$ 的條件機率之下， $\mathcal{G}_k$ 是最有可能的出象，換成數學的語言就是：
 $P(\mathcal{G}_k|X=x)=\max_{l}P(G=\mathcal{G}_l|X=x)$  
（或者說是 $=\max_{l}P(G=l|X=x)$ ）

有這個基本觀念後，我們設定資料屬於類別 $k$ 的先驗機率為： $\pi_k=P(G=k)$ ，而當然  $\sum_{k=1}^K\pi_k =1$。

而透過貝氏定理，我們可以得到以下關係：

$$P(G=k|X=x)=\frac{{\color{Red} f_k(x)}{\color{Blue} \pi_k}}{{\color{DarkOrange} \Sigma_{l=1}^k f_l(x)\pi_l}}=\frac{{\color{Red} P(X|G=k)}{\color{Blue} P(G=k)} }{{\color{DarkOrange} \Sigma_l P(X|G=l)P(G=l)}}$$

根據上式，我們必須對 $f_k(x)$ 做一些假設，這也就是當資料是第 $k$ 類時， $X$ 的機率密度函數。我們設定各分類的機率密度服從「**多變量常態分配(Multivariate Normal Distribution)**」。

$$f_k(x)=\frac{1}{(2\pi)^{\frac{p}{2}}|\Sigma_k|^\frac{1}{2}}e^{-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)}$$
要注意的是，在LDA的架構之下，所有類別的pdf均享有相同的共變數矩陣，也就是 $\Sigma_k=\Sigma,\forall k$。

接著，我們就可以以去比較兩兩類別之間的後驗發生機率 $P(G|X)$ ，我們在此利用對數的良好性質來分析兩者關係，假設我們現在要探討類別 $k$ 和類別 $l$ ，誰的發生機率大呢？我們用下列關係式來表達：

$$\begin{aligned}
\ln\frac{P(G=k|X=x)}{P(G=l|X=x)}&=\ln \frac{f_k(x)\pi_k}{f_l(x)\pi_l}=\ln\frac{\pi_k}{\pi_l}+\ln {f_k(x)}-\ln{f_l(x)}\\
&=\ln\frac{\pi_k}{\pi_l}-\frac{1}{2}(x-\mu_k)^T\Sigma^{-1}(x-\mu_k)+\frac{1}{2}(x-\mu_l)^T\Sigma^{-1}(x-\mu_l)\\
&=\ln\frac{\pi_k}{\pi_l}-\frac{1}{2}(\mu_k+\mu_l)^T\Sigma^{-1}(\mu_k-\mu_l)+x^T\Sigma^{-1}(\mu_k-\mu_l)
\end{aligned}$$

注意!這樣良好的線性性質是來自於我們假設兩個分類具有**相同的**共變異數矩陣。若我們對類別 $k$ 和類別 $l$ 的分界線感興趣，其分界線就是位在兩者機率密度相等之處，也就是當上式**等於0**時。

透過相同的想法，我們可以建立一個**線性判別函數(linear discriminant function)** $\delta_k(x)$ ，來決定該資料應被分配到哪一個類別，也就是 $G(x)=\mathop{\arg\max}\limits_{k}\delta_k(x)$ 。
線性判別函數如下所示：
$$\delta_k(x)=x^T\Sigma^{-1}\mu_k-\frac{1}{2}\mu_k^T \Sigma^{-1}\mu_k +\ln \pi_k$$

線性判別函數的推導來自以下的成比例關係：
$$\begin{aligned}
P(G=k|X=x)&\propto f_k(x)\pi_k \\
&\propto -\frac{1}{2}(x-\mu_k)^T \Sigma^{-1} (x-\mu_k)+\ln \pi_k = -\frac{1}{2}x^T\Sigma^{-1}x+x^T\Sigma^{-1}\mu_k-\frac{1}{2}\mu_k^T \Sigma^{-1}\mu_k +\ln \pi_k\\
&\propto  x^T\Sigma^{-1}\mu_k-\frac{1}{2}\mu_k^T \Sigma^{-1}\mu_k +\ln \pi_k \equiv  \delta_k(x)
\end{aligned}$$

正如古典的統計分析一樣，我們並無法知道母體分配的參數，故我們在進行統計學習時，則選擇使用**訓練集的資料來估計母體參數**。

 $$\begin{aligned}
&\hat{\pi_k}=N_k/N \\
 &\hat{\mu_k}=\sum_{g_i=k}x_i/N_k \\
 &\hat{\Sigma}=\sum_{k=1}^K\sum_{g_i=k}(x_i-\hat{\mu_k})(x_i-\hat{\mu_k})^T/(N-K)
\end{aligned}$$
其中 $N_k$ 是類別$k$在訓練集中的數量(observations)。

### 2-class LDA (from Ex. 4.2)

**Suppose we have features $x \in \mathbb{R}^p$, a two-class response, with class sizes $N_1$, $N_2$, and the target coded as $−N/N_1$, $N/N_2$.**

**(a) Show that the LDA rule classifies to class 2 if**
$$x^T\hat{\Sigma^{-1}}(\hat{\mu_2}-\hat{\mu_1}) > \frac{1}{2}(\hat{\mu_2}+\hat{\mu_1})^T\hat{\Sigma^{-1}}(\hat{\mu_2}-\hat{\mu_1})-\ln(N_2/N_1)$$ 
**and class 1 otherwise.**

*Sol:*

在二元的分類中，我們可以回溯到先前講到的log-odds的觀念，也就是建立此式來做比較(改寫自課本式4.9)：
$$\begin{aligned}
\ln\frac{P(G=2|X=x)}{P(G=1|X=x)}
=\ln\frac{\pi_2}{\pi_1}-\frac{1}{2}(\mu_2+\mu_1)^T\Sigma^{-1}(\mu_2-\mu_1)+x^T\Sigma^{-1}(\mu_2-\mu_1)
\end{aligned}$$
根據對數性質，若此式**大於0**，就表示 $P(G=2|X=x)$ 的機率相對於 $P(G=1|X=x)$ 來得高，故我們自然會將其分類到 class 2。
而我們根據訓練集對上式做估計並且做些許代數運算即可得到決策的函數：
$$x^T\hat{\Sigma^{-1}}(\hat{\mu_2}-\hat{\mu_1}) > \frac{1}{2}(\hat{\mu_2}+\hat{\mu_1})\hat{\Sigma^{-1}}(\hat{\mu_2}-\hat{\mu_1})-\ln(N_2/N_1)$$

**(b) Consider minimization of the least squares criterion**
$$\sum^N_{i=1}(y_i-\beta_0-x_i^T\beta)^2$$
**Show that the solution $\hat{\beta}$ satisfies**

$$[(N-2)\hat{\Sigma}+N\hat{\Sigma}_B]\beta=N(\hat{\mu_2}-\hat{\mu_1})$$
where,
$$\hat{\Sigma}_B=\frac{N_1N_2}{N^2}(\hat{\mu_2}-\hat{\mu_1})(\hat{\mu_2}-\hat{\mu_1})^T$$

*Sol:*

首先，我們知道 $N=N_1+N_2$。
而這個線性迴歸方程包含常數項，則根據 Normal equation 可以得到對 $\hat{\beta_0}$ 和 $\hat{\beta_1}$ 的估計：


$$ \begin{bmatrix}
\hat{\beta_0}\\ 
\hat{\beta}
\end{bmatrix}=(X^TX)^{-1}X^Ty $$

其中 **designed matrix** 可這樣表示：($X$裡面有$\bf{1}$向量)

$$X^TX=\begin{bmatrix}
N & \sum_{i=1}^N x_i^T\\ 
\sum_{i=1}^N x_i & \sum_{i=1}^Nx_ix_i^T 
\end{bmatrix}$$
其中
$$\sum_{i=1}^Nx_i =\sum_{i=1}^{N_1}x_i+\sum_{i=1}^{N_2}x_i = N_1\hat{\mu_1}+ N_2\hat{\mu_2}$$
又知
$$\begin{aligned}
\hat{\Sigma} &= \frac{1}{N-2}\sum_{k=1}^2\sum_{g_i=k}(x_i-\hat{\mu_k})(x_i-\hat{\mu_k})^T \\
&=\frac{1}{N-2}[\sum_{g_i=1}x_ix_i^T-N_1 \mu_1 \mu_1^T+\sum_{g_i=2}x_ix_i^T-N_2\mu_2 \mu_2^T]
\end{aligned}$$
因此
$$\sum_{i=1}^Nx_ix_i^T = (N-2)\hat{\Sigma}+N_1 \mu_1 \mu_1^T+N_2\mu_2 \mu_2^T$$
若我們將屬於類別1的資料放在矩陣的前 $N_1$ 個，而類別2的資料 $N_2$ 個緊接在後，$X^Ty$ 可以寫成：

$$ X^Ty= \begin{bmatrix}
1 &\cdots  & 1 &1  & \cdots &1 \\ 
x_1 &\cdots  &x_{N_1}  &x_{N_1+1}  &\cdots  & x_{N_1+N_2}
\end{bmatrix}
\begin{bmatrix}
-N/N_1\\ 
\vdots\\ 
-N/N_1\\ 
N/N_2\\ 
\vdots\\ 
N/N_2
\end{bmatrix} =\begin{bmatrix}
0 \\
-N\mu_1+N\mu_2
\end{bmatrix}$$

整理先前計算出的這些東西代入 Normal equation，整理後可得：

(註：$\beta_0=(-\frac{N_1}{N}\mu_1^T-\frac{N_2}{N}\mu_2^T)\beta$)

$$(N_1\mu_1+N_2\mu_2)(-\frac{N_1}{N}\mu_1^T-\frac{N_2}{N}\mu_2^T)\beta+((N-2)\hat{\Sigma} +N_1\mu_1\mu_1^T+N_2\mu_2\mu_2^T)\beta=N(\mu_2-\mu_1)$$

經過一些運算，並引入題目對 $\hat{\Sigma_B}$ 的定義，即可得：
$$[(N-2)\hat{\Sigma}+N\hat{\Sigma}_B]\beta=N(\hat{\mu_2}-\hat{\mu_1})$$


**(c) Hence show that $\hat{\Sigma}_B \beta$ is in the direction $(\hat{\mu_2} - \hat{\mu_1})$ and thus**
$$\hat{\beta}\propto \hat{\Sigma}^{-1} (\hat{\mu_2} - \hat{\mu_1})$$
**Therefore the least-squares regression coefficient is identical to the LDA coefficient, up to a scalar multiple.**

*Sol:*

若我們直接計算 $\hat{\Sigma}_B \beta$，由於 $(\hat{\mu_2}-\hat{\mu_1})^T \beta$ 內積後是一個scalar。也就是

$$\hat{\Sigma}_B \beta=\frac{N_1N_2}{N^2}(\hat{\mu_2}-\hat{\mu_1}){\color{Red} (\hat{\mu_2}-\hat{\mu_1})^T \beta}=\frac{N_1N_2}{N^2}(\hat{\mu_2}-\hat{\mu_1}){\color{Red} c}$$

換句話說， $\hat{\Sigma}_B \beta$ 和 $(\hat{\mu_2}-\hat{\mu_1})$ 差了 ${\color{Red} \frac{N_1N_2}{N^2}c}$ 倍。又這些常數都為正，故得知：
$$ \hat{\beta} \propto \hat{\Sigma}^{-1} (\hat{\mu_2}-\hat{\mu_1}) $$

此即表示線性迴歸與LDA其實是有相像的運算邏輯。

**(d) Show that this result holds for any (distinct) coding of the two classes.**

*Sol:*

在上一題中，我們並沒有特別對 $t_k$ 有特別假設，這樣的性質是來自於一開始對 $\hat{\Sigma}_B$ 的設計，並且與 $\beta$ 內積後出現純數而推演出這樣的比例關係。


**(e) Find the solution $\hat{\beta_0}$ (up to the same scalar multiple as in (c), and hence the predicted value $\hat{f}(x)=\hat{\beta_0}+x^T\hat{\beta}$. Consider the following rule: classify to class 2 if $f(x) > 0$and class 1 otherwise. Show this is not the same as the LDA rule unless the classes have equal numbers of observations.**

*Sol:*

在先前的討論，我們知道：
$$\beta_0 = -\frac{1}{N}(N_1\hat{\mu}^T_1+N_2\hat{\mu}^T_2)\hat{\beta}$$

我們可改寫 $\hat{f}(x)$ 如下：

$$\begin{aligned}
\hat{f}(x)&=-\frac{1}{N}(N_1\hat{\mu}^T_1+N_2\hat{\mu}^T_2-Nx^T)    \hat{\beta }\\
&=-\frac{1}{N}(N_1\hat{\mu}^T_1+N_2\hat{\mu}^T_2-Nx^T) {\color{Red} \lambda \hat{\Sigma}^{-1}(\hat{\mu_2}-\hat{\mu_1})}
\end{aligned}$$

經過展開，並重整，則可得到下式：

$$x^T\hat{\Sigma^{-1}}(\hat{\mu_2}-\hat{\mu_1}) > \frac{1}{N}(N_2\hat{\mu_2}+N_1\hat{\mu_1})^T\hat{\Sigma^{-1}}(\hat{\mu_2}-\hat{\mu_1})$$
若 $N_1=N_2$ ，則 $N_1/N=N_2/N=1/2$，而 $\ln(N_2/N_1)=\ln 1=0$

這樣的決策準則就跟LDA一樣了！

## Quadratic Discriminant Analysis (QDA)

還記得先前在推導 LDA 時除了常態假設之外，還有一個變異數矩陣相等的假設，若我們放寬這個假設，這個方法就變成 QDA ，Quadratic 的部分就是來自於多變量常態分配中在 $e$ 上的 $x$ 二次式不會被對消。

二次判別函數(Quadratic Discriminant function)如下：

$$\delta_k(x)=-\frac{1}{2}\ln|\Sigma_k|-\frac{1}{2}(x-\mu_k)^T\Sigma_k^{-1}(x-\mu_k)+\ln \pi_k$$

個類別的分界線可以這樣表示

$$\{x|\delta_k(x)=\delta_l(x) \}$$
QDA 有參數太多的缺點。

## Regulized Discriminant Analysis (RDA)  
![](https://esl.hohoweiya.xyz/img/04/fig4.7.png)


Friedman(1989) 提出了介於LDA與QDA之間的方法，稱之為 **Regulized Discriminant Analysis (RDA)**。正則化的共變異數矩陣如下所示：

$$\hat{\Sigma}_k(\alpha)=\alpha \hat{\Sigma}_k+(1-\alpha)\hat{\Sigma},\ \alpha \in[0,1]$$
在此， $\hat{\Sigma}$ 是LDA所做的共變數矩陣，而 $\hat{\Sigma}_k$ 為QDA的共變數矩陣。

by 課本 
$$\hat{\Sigma}(\gamma)=\gamma\hat{\Sigma}+(1-\gamma)\hat{\sigma}^2I$$

## LDA 的計算


## Reduced-Rank LDA


## LDA 和 linear regression 的關係 (from Ex. 4.3)

**Suppose that we transform the original predictors $\mathbf{X}$ to $\mathbf{\hat{Y}}$ by taking the predicted values under linear regression. Show that LDA using $\bf{\hat{Y}}$ is identical to using LDA in the original space.**

*Sol:*
根據題意，我們知道 $x \in \mathbb{R}^p$ 和 $y \in \mathbb{R}^k$ ，且：
$$\begin{aligned} 
& \hat{y}=\hat{\mathbf{B}}x \\
& \mathbf{\hat{Y}} = \mathbf{X} \hat{\mathbf{B}}= \mathbf{X}(\mathbf{X}^T\mathbf{X})^{-1}\mathbf{X}^T\mathbf{Y}
\end{aligned}$$

對於任意的類別 $k$ ，我們可以建構出該類別的平均數，我們定義類別 $k$ 對應其 $X$ 以及 $Y$ 的平均數為：

$$\begin{aligned} 
& \mu_k = \frac{1}{N_k}\sum_{g_i=k}x^T_i \\
& \hat{\mu_k} = \mathbf{B}^T\mu_k
\end{aligned}$$

若我們定義 $\mathbf{X}$ 的共變數矩陣為 ${\Sigma}$ ，則 $\hat{\mathbf{Y}}$的共變數矩陣則為：
$$\hat{\Sigma} = \mathbf{B}^T \Sigma \mathbf{B}$$
其中

$$\begin{aligned} 
& \Sigma =\frac{1}{N-K}X^T(I-YD^{-1}Y^T)X\\
&,where\  D =\begin{bmatrix}
n_1 & 0  & \cdots &0 \\ 
0 & n_2 &\cdots  &0 \\ 
 \vdots&\vdots  & \vdots & \vdots\\ 
0 &0  & \cdots & n_k
\end{bmatrix}
\end{aligned}$$

若我們直接用 $\hat{\mathbf{Y}}$ 的資料做 LDA，則判別函數為：

$$\delta_k(\hat{Y})= \hat{\mathbf{Y}}\hat{\Sigma}^{-1}\hat{\mu_k}-\frac{1}{2}\hat{\mu_k}^T\hat{\Sigma}^{-1}\hat{\mu_k}+\ln \pi_k$$

第一項可以改寫如下(全部寫成矩陣的樣子)：( $B=(X^TX)^{-1}X^TY$ )

$$\begin{aligned} 
\hat{\mathbf{Y}}\hat{\Sigma}^{-1}\hat{\mu}&=(XB)(B^T\Sigma B)^{-1}(B^TX^TYD^{-1}) \\
&=X\hat{\Sigma}^{-1}\mu
\end{aligned}$$

這就是使用 $\mathbf{X}$ 進行LDA的其中一部分。

而第二項，若將全部 $K$ 種類的平均搜集起來，寫成矩陣 $\hat{\mu}$ 可改寫為：

$$\begin{aligned} 
\hat{\mu_k}^T\hat{\Sigma}^{-1}\hat{\mu}&=(B^T \mu_k)^T\hat{\Sigma}^{-1}(B^TX^TYD^{-1})\\
&=\mu_k^TB\hat{\Sigma}^{-1}(B^TX^TYD^{-1})\\
&=\mu_k^TB(B^T\Sigma B)^{-1}B^TX^TYD^{-1}\\
&=\mu_k^T\Sigma^{-1}\mu
\end{aligned}$$

此即使用 $\mathbf{X}$ 做LDA的另外一部分。
所以我們可以知道，若我們透過迴歸方程的模式找出 $\hat{\mathbf{Y}}$ ，我們便可從對 $\hat{\mathbf{Y}}$ 做LDA轉換成對 $\mathbf{X}$ 做LDA。

 


## Logistic Regression

羅吉斯迴歸(Logistic Regression)是針對 $K$ 個類別的後驗機率做一個線性的的模型，而跟一般線性迴歸不同的地方是在於，羅吉斯迴歸透過對數的轉換將機率的取值保留在$[0,1]$之間。若我們的資料有 $K$ 類別，則羅吉斯回歸透過建立 $K-1$ 個 **log-odds** 來建構我們所要的模型，少的那一個類別就拿來作為對照組之用。
以 $K$ 作為基準，Logistic Regression 會建立這些模型關係：

$$\begin{aligned} 
\ln \frac{P(G=1|X=x)}{P(G=K|X=x)} &= \beta_{10}+\beta_1^Tx\\
\ln \frac{P(G=2|X=x)}{P(G=K|X=x)} &= \beta_{20}+\beta_2^Tx \\
& \vdots \\
\ln \frac{P(G=K-1|X=x)}{P(G=K|X=x)} &= \beta_{(K-1)0}+\beta_{K-1}^Tx
\end{aligned}$$

我們可以對這些式子取指數回來，可得：

$$P(G=l|X=x) =P(G=K|X=x) e^{\beta_{l0}+\beta_l^Tx}, l=1,2,\cdots,N-1$$

又

$$\begin{aligned} 
P(G=K|X=x) &= 1-\sum_{l=1}^{K-1}P(G=l|X=x)\\&=1-P(G=K|X=x)\sum_{l=1}^{K-1}e^{\beta_{l0}+\beta_l^Tx}
\end{aligned} $$

故可求解出：
$$P(G=K|X=x) = \frac{1}{1+\sum_{l=1}^{K-1}e^{\beta_{l0}+\beta_l^Tx} } $$
以及, 
$$P(G=k|X=x) = \frac{e^{\beta_{k0}+\beta_k^Tx} }{1+\sum_{l=1}^{K-1}e^{\beta_{l0}+\beta_l^Tx} }$$

若我們簡化表示，將這些參數定義為：

$$\theta = \{\beta_{10},\beta_1^T,\cdots,\beta_{(K-1)0},\beta_{(K-1)}^T  \}$$
則類別$k$機率可以表示成： $P(G=k|X=x)=p_k(x;\theta)$

### 配適 Logistic Regression 的參數

在共有 $K$ 類的資料中，可以把資料想成是從**多項分配(multinomial)**取出來的。因此 **likelihood** 可以想成是：

$$L = \prod _{i=1}^Np_{g_i}(x_i)$$

(只有其中一項的指數會是 $1$ ，其他都是 $0$ )
將之轉為 **log-likelihood**：

$$l=\sum _{i=1}^N \ln p_{g_i}(x_i)$$

若我們只先看 $K=2$ 的情形，則：(可參考課本4.4.1)
$$l(\beta) = \sum_{i=1}^N\{y_i\beta^Tx_i -\ln(1+e^{\beta^Tx_i})  \}$$

我們求解 log-likelihood 的極值，另一階導數為0：
$$\frac{\partial l(\beta)}{\partial \beta} = \sum_{i=1}^N x_i(y_i-p(x_i;\beta))=0$$

其中 $p(x_i;\beta) = \frac{e^{\beta^Tx}}{1+e^{\beta^Tx}}$ 。

我們將一階條件，擴寫成矩陣形式：

$$\frac{\partial l(\beta)}{\partial \beta} = \mathbf{X}^T(y-p)$$

二階條件的 **Hessian matrix** 如下：

$$\frac{\partial^2 l(\beta)}{\partial \beta\beta^T} = -\mathbf{X}^T\mathbf{WX}$$

- \mathbf{W} 是第 $i$ 個元素是 $p(x_i;\beta^{old})(1-p(x_i;\beta^{old}))$ 的對角矩陣。

由於此問題的一階條件沒有 closed-form，因此我們使用**牛頓法**找根：
$$\begin{aligned} 
\beta^{new} &= \beta^{old} + (\mathbf{W}^T\mathbf{WX})^{-1}\mathbf{X}^T(\mathbf{y}-\mathbf{p})\\
&= (\mathbf{W}^T\mathbf{WX})^{-1}\mathbf{X}^T\mathbf{W}(\mathbf{X}\beta^{old}+\mathbf{W}^{-1}(\mathbf{y}-\mathbf{p}))\\
&=(\mathbf{W}^T\mathbf{WX})^{-1}\mathbf{X}^T\mathbf{W}\mathbf{z}
\end{aligned}$$

觀察上式，我們令 $\mathbf{z}=\mathbf{X}\beta^{old}+\mathbf{W}^{-1}(\mathbf{y}-\mathbf{p})$，則我們每進行一次迭代，都是彷彿在進行一次 *weighted least squared*，故稱之為 **iteratively reweighted least squared(IRLS)**。在迭代的過程中，我們可以選擇從 $\beta=0$ 開始。


$$\beta^{new} \leftarrow \mathop{\arg\min}_\beta (\mathbf{z}-\mathbf{X}\beta)^T\mathbf{W}(\mathbf{z}-\mathbf{X}\beta)$$

### 一個 $K=2$ 的例子，從 $x \in \mathbb{R}$ 到 $x \in \mathbb{R}^p$ (from Ex. 4.5)

![](https://esl.hohoweiya.xyz/img/04/fig4.16.png)

**Consider a two-class logistic regression problem with $x \in \mathbb{R}$. Characterize the maximum-likelihood estimates of the slope and intercept parameter if the sample $x_i$ for the two classes are separated by a point $x \in \mathbb{R}$. Generalize this result to (a) $x \in \mathbb{R}^p$ (see Figure 4.16), and (b) more than two classes.**

在先前的討論中，我們知道當 $K=2$ 時，log-likelihood function 為

$$l(\beta) = \sum_{i=1}^N\{y_i\beta^Tx_i -\ln(1+e^{\beta^Tx_i})  \}$$

而若 $x \in \mathbb{R}$(一維)，則

$$y_i =\left\{\begin{matrix}
0,\ x_i \leq x_0\\ 
1,\ x_i > x_0
\end{matrix}\right.$$

我們考慮最基本的一維羅吉斯回歸模型，的對數概似函數：
$$l(\beta) = \sum_{i=1}^N\{y_i\beta^Tx_i -\ln(1+e^{\beta^Tx_i})  \}$$
故
$$\begin{aligned}
l(\beta) &= \sum_{i=1}^N\{y_i(\beta_0+\beta_1x_i) -\ln(1+e^{(\beta_0+\beta_1x_i)})  \} \\
&=\sum_{y_i=0}[-\ln(1+e^{(\beta_0+\beta_1x_i)})]+\sum_{y_i=1}[(\beta_0+\beta_1x_i)-\ln(1+e^{(\beta_0+\beta_1x_i)})]
 \end{aligned}$$

若我們想要 **maximize log-likelihood**，很明顯地，當 $\beta \rightarrow \infty$ 時就會發散，這導致我們無法找到一個 closed-form 的解。

但很直觀地，當$x \in \mathbb{R}$，由於資料只有一維，那我們在 x 軸上找到 $x=x_0$ ，並在此處畫上一條垂直於 x 軸的線，這樣就能夠明確地劃分出兩個分類。

- 當 $x \in \mathbb{R}^p$ 時，該如何處理呢？(p>1)
  
我們透過一樣的邏輯擴展 $l(\beta)$ 用矩陣向量表示：

$$\begin{aligned}
l(\beta) &= \sum_{i=1}^N\{y_i\beta^Tx_i -\ln(1+e^{\beta^Tx_i})  \} \\
&=\sum_{y_i=0}[-\ln(1+e^{\beta^Tx_i})]+\sum_{y_i=1}[\beta^Tx_i-\ln(1+e^{\beta^Tx_i})]\end{aligned}$$

同樣地，我們同樣會發現 $l(\beta)$ 會發散。

- 當 $K>2$ 時，要如何處理呢？

直觀上來說，當 $K>2$ 則需要超過一個超平面來協助我們分類，我們同樣能寫出這種情況的 log-likelihood function：

$$\begin{aligned}
l(\beta) &= \sum_{i=1}^N [\sum_{k=1}^{K-1}\mathbf{1}_{y_i=k} \beta_k^Tx_i-\ln(1+\sum_{l=1}^{K-1}e^{\beta_l^Tx_i}]\\
&=\sum_{k=1}^{K-1} \sum_{g_k}[\beta_k^Tx_i-\ln(1+\sum_{l=1}^{K-1}\beta_l^Tx_i) ]
+\sum_{g_k}[-\ln(1+\sum_{l=1}^{K-1}e^{\beta_l^Tx_i})]
\end{aligned}$$

而這條式子同樣也找不到 \beta 的 closed-form。

### Example: South African Heart Disease


### Logistic v.s. LDA


## Perceptron Learning Algorithm (PLA)











