Linear\_Methods\_for\_Classification
================
Eric Huang

## Linear Regression of an Indicator Matrix

若我們說，![\\mathcal{G}](https://latex.codecogs.com/png.latex?%5Cmathcal%7BG%7D
"\\mathcal{G}") 有 ![K](https://latex.codecogs.com/png.latex?K "K")
個類別，為了表示該筆資料是何種類別，我們可以建立一個
![Y](https://latex.codecogs.com/png.latex?Y "Y") 矩陣,
![Y=(Y\_1,...,Y\_k)\_{N \\times
K}](https://latex.codecogs.com/png.latex?Y%3D%28Y_1%2C...%2CY_k%29_%7BN%20%5Ctimes%20K%7D
"Y=(Y_1,...,Y_k)_{N \\times K}")，也就是說![Y](https://latex.codecogs.com/png.latex?Y
"Y")矩陣中有![N](https://latex.codecogs.com/png.latex?N
"N")個![K](https://latex.codecogs.com/png.latex?K "K")維的 row
vectors。而根據線性迴歸模型，我們可得對
![Y](https://latex.codecogs.com/png.latex?Y "Y")的估計為：

  
![ \\hat{Y} = X(X^TX)^{-1}X^TY=X \\hat{B}
](https://latex.codecogs.com/png.latex?%20%5Chat%7BY%7D%20%20%3D%20X%28X%5ETX%29%5E%7B-1%7DX%5ETY%3DX%20%5Chat%7BB%7D%20
" \\hat{Y}  = X(X^TX)^{-1}X^TY=X \\hat{B} ")  

![B\_{(p+1)\\times
K}](https://latex.codecogs.com/png.latex?B_%7B%28p%2B1%29%5Ctimes%20K%7D
"B_{(p+1)\\times K}")：![p](https://latex.codecogs.com/png.latex?p "p") 個
inputs 然後加上截距項。

或者是從另外一個觀點，也就是我們希望極小化![\\hat{y}](https://latex.codecogs.com/png.latex?%5Chat%7By%7D
"\\hat{y}")跟 ![y](https://latex.codecogs.com/png.latex?y "y")的距離。   
![\\min\_{\\bf{B}}\\sum\_{i=1}^N||y\_i-\[(1,x\_i^T) \\textbf{B} \]^T
||^2](https://latex.codecogs.com/png.latex?%5Cmin_%7B%5Cbf%7BB%7D%7D%5Csum_%7Bi%3D1%7D%5EN%7C%7Cy_i-%5B%281%2Cx_i%5ET%29%20%5Ctextbf%7BB%7D%20%5D%5ET%20%7C%7C%5E2
"\\min_{\\bf{B}}\\sum_{i=1}^N||y_i-[(1,x_i^T) \\textbf{B} ]^T ||^2")  
也就是說，![\\hat{f}(x)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28x%29
"\\hat{f}(x)")會分類到最接近的目標群( ![y\_i=t\_k,\\ if\\ g\_i =
k](https://latex.codecogs.com/png.latex?y_i%3Dt_k%2C%5C%20if%5C%20g_i%20%3D%20k
"y_i=t_k,\\ if\\ g_i = k") )：

  
![\\hat{G}(x)=\\mathop{\\arg\\min}\\limits\_{k}||\\hat{f}(x)-t\_k||^2](https://latex.codecogs.com/png.latex?%5Chat%7BG%7D%28x%29%3D%5Cmathop%7B%5Carg%5Cmin%7D%5Climits_%7Bk%7D%7C%7C%5Chat%7Bf%7D%28x%29-t_k%7C%7C%5E2
"\\hat{G}(x)=\\mathop{\\arg\\min}\\limits_{k}||\\hat{f}(x)-t_k||^2")  
\#模擬Fig. 4.3¶ <https://esl.hohoweiya.xyz/notes/LDA/sim-4-3/index.html>

## Linear Discriminant Analysis(LDA)

在課本的2.4節，我們知道在做分類決策時，我們是在極大化某種後驗機率(posrerior
probability)，也就給定![X=x](https://latex.codecogs.com/png.latex?X%3Dx
"X=x")時，找一個最大可能性的分類當作分析的結果。 也就是說： 我們令分類所做的預測
![\\hat{G}(x)=\\mathcal{G}\_k](https://latex.codecogs.com/png.latex?%5Chat%7BG%7D%28x%29%3D%5Cmathcal%7BG%7D_k
"\\hat{G}(x)=\\mathcal{G}_k")
(![\\mathcal{G}\_k](https://latex.codecogs.com/png.latex?%5Cmathcal%7BG%7D_k
"\\mathcal{G}_k")為其中一種分類)；
也就表示，當給定![X=x](https://latex.codecogs.com/png.latex?X%3Dx
"X=x")的條件機率之下，![\\mathcal{G}\_k](https://latex.codecogs.com/png.latex?%5Cmathcal%7BG%7D_k
"\\mathcal{G}_k")是最有可能的出象，換成數學的語言就是：
![P(\\mathcal{G}\_k|X=x)=\\max\_{l}P(G=\\mathcal{G}\_l|X=x)](https://latex.codecogs.com/png.latex?P%28%5Cmathcal%7BG%7D_k%7CX%3Dx%29%3D%5Cmax_%7Bl%7DP%28G%3D%5Cmathcal%7BG%7D_l%7CX%3Dx%29
"P(\\mathcal{G}_k|X=x)=\\max_{l}P(G=\\mathcal{G}_l|X=x)") （或者說是
![=\\max\_{l}P(G=l|X=x)](https://latex.codecogs.com/png.latex?%3D%5Cmax_%7Bl%7DP%28G%3Dl%7CX%3Dx%29
"=\\max_{l}P(G=l|X=x)")）

有這個基本觀念後，我們設定資料屬於類別![k](https://latex.codecogs.com/png.latex?k
"k")的先驗機率為：![\\pi\_k=P(G=k)](https://latex.codecogs.com/png.latex?%5Cpi_k%3DP%28G%3Dk%29
"\\pi_k=P(G=k)")，而當然 ![\\sum\_{k=1}^K\\pi\_k
=1](https://latex.codecogs.com/png.latex?%5Csum_%7Bk%3D1%7D%5EK%5Cpi_k%20%3D1
"\\sum_{k=1}^K\\pi_k =1")。

而透過貝氏定理，我們可以得到以下關係：

  
![P(G=k|X=x)=\\frac{{\\color{Red} f\_k(x)}{\\color{Blue}
\\pi\_k}}{{\\color{DarkOrange} \\sum\_{l=1}^k
f\_l(x)\\pi\_l}}=\\frac{{\\color{Red} P(X|G=k)}{\\color{Blue} P(G=k)}
}{{\\color{DarkOrange} \\sum\_l
P(X|G=l)P(G=l)}}](https://latex.codecogs.com/png.latex?P%28G%3Dk%7CX%3Dx%29%3D%5Cfrac%7B%7B%5Ccolor%7BRed%7D%20f_k%28x%29%7D%7B%5Ccolor%7BBlue%7D%20%5Cpi_k%7D%7D%7B%7B%5Ccolor%7BDarkOrange%7D%20%5Csum_%7Bl%3D1%7D%5Ek%20f_l%28x%29%5Cpi_l%7D%7D%3D%5Cfrac%7B%7B%5Ccolor%7BRed%7D%20P%28X%7CG%3Dk%29%7D%7B%5Ccolor%7BBlue%7D%20P%28G%3Dk%29%7D%20%7D%7B%7B%5Ccolor%7BDarkOrange%7D%20%5Csum_l%20P%28X%7CG%3Dl%29P%28G%3Dl%29%7D%7D
"P(G=k|X=x)=\\frac{{\\color{Red} f_k(x)}{\\color{Blue} \\pi_k}}{{\\color{DarkOrange} \\sum_{l=1}^k f_l(x)\\pi_l}}=\\frac{{\\color{Red} P(X|G=k)}{\\color{Blue} P(G=k)} }{{\\color{DarkOrange} \\sum_l P(X|G=l)P(G=l)}}")  

根據上式，我們必須對![f\_k(x)](https://latex.codecogs.com/png.latex?f_k%28x%29
"f_k(x)")做一些假設，這也就是當資料是第![k](https://latex.codecogs.com/png.latex?k
"k")類時，![X](https://latex.codecogs.com/png.latex?X
"X")的機率密度函數。我們設定各分類的機率密度服從「**多變量常態分配(Multivariate
Normal Distribution)**」。

  
![f\_k(x)=\\frac{1}{(2\\pi)^{\\frac{p}{2}}|\\Sigma\_k|^\\frac{1}{2}}e^{-\\frac{1}{2}(x-\\mu\_k)^T\\Sigma\_k^{-1}(x-\\mu\_k)}](https://latex.codecogs.com/png.latex?f_k%28x%29%3D%5Cfrac%7B1%7D%7B%282%5Cpi%29%5E%7B%5Cfrac%7Bp%7D%7B2%7D%7D%7C%5CSigma_k%7C%5E%5Cfrac%7B1%7D%7B2%7D%7De%5E%7B-%5Cfrac%7B1%7D%7B2%7D%28x-%5Cmu_k%29%5ET%5CSigma_k%5E%7B-1%7D%28x-%5Cmu_k%29%7D
"f_k(x)=\\frac{1}{(2\\pi)^{\\frac{p}{2}}|\\Sigma_k|^\\frac{1}{2}}e^{-\\frac{1}{2}(x-\\mu_k)^T\\Sigma_k^{-1}(x-\\mu_k)}")  
要注意的是，在LDA的架構之下，所有類別的pdf均享有相同的共變數矩陣，也就是 ![\\Sigma\_k=\\Sigma,\\forall
k](https://latex.codecogs.com/png.latex?%5CSigma_k%3D%5CSigma%2C%5Cforall%20k
"\\Sigma_k=\\Sigma,\\forall k")。

接著，我們就可以以去比較兩兩類別之間的後驗發生機率![P(G|X)](https://latex.codecogs.com/png.latex?P%28G%7CX%29
"P(G|X)")，我們在此利用對數的良好性質來分析兩者關係，假設我們現在要探討類別![k](https://latex.codecogs.com/png.latex?k
"k")和類別![l](https://latex.codecogs.com/png.latex?l
"l")，誰的發生機率大呢？我們用下列關係式來表達：

  
![\\begin{aligned}&#10;\\ln\\frac{P(G=k|X=x)}{P(G=l|X=x)}&=\\ln
\\frac{f\_k(x)\\pi\_k}{f\_l(x)\\pi\_l}=\\ln\\frac{\\pi\_k}{\\pi\_l}+\\ln
{f\_k(x)}-\\ln{f\_l(x)}\\\\&#10;&=\\ln\\frac{\\pi\_k}{\\pi\_l}-\\frac{1}{2}(x-\\mu\_k)^T\\Sigma^{-1}(x-\\mu\_k)+\\frac{1}{2}(x-\\mu\_l)^T\\Sigma^{-1}(x-\\mu\_l)\\\\&#10;&=\\ln\\frac{\\pi\_k}{\\pi\_l}-\\frac{1}{2}(\\mu\_k+\\mu\_l)^T\\Sigma^{-1}(\\mu\_k-\\mu\_l)+x^T\\Sigma^{-1}(\\mu\_k-\\mu\_l)&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cln%5Cfrac%7BP%28G%3Dk%7CX%3Dx%29%7D%7BP%28G%3Dl%7CX%3Dx%29%7D%26%3D%5Cln%20%5Cfrac%7Bf_k%28x%29%5Cpi_k%7D%7Bf_l%28x%29%5Cpi_l%7D%3D%5Cln%5Cfrac%7B%5Cpi_k%7D%7B%5Cpi_l%7D%2B%5Cln%20%7Bf_k%28x%29%7D-%5Cln%7Bf_l%28x%29%7D%5C%5C%0A%26%3D%5Cln%5Cfrac%7B%5Cpi_k%7D%7B%5Cpi_l%7D-%5Cfrac%7B1%7D%7B2%7D%28x-%5Cmu_k%29%5ET%5CSigma%5E%7B-1%7D%28x-%5Cmu_k%29%2B%5Cfrac%7B1%7D%7B2%7D%28x-%5Cmu_l%29%5ET%5CSigma%5E%7B-1%7D%28x-%5Cmu_l%29%5C%5C%0A%26%3D%5Cln%5Cfrac%7B%5Cpi_k%7D%7B%5Cpi_l%7D-%5Cfrac%7B1%7D%7B2%7D%28%5Cmu_k%2B%5Cmu_l%29%5ET%5CSigma%5E%7B-1%7D%28%5Cmu_k-%5Cmu_l%29%2Bx%5ET%5CSigma%5E%7B-1%7D%28%5Cmu_k-%5Cmu_l%29%0A%5Cend%7Baligned%7D
"\\begin{aligned}
\\ln\\frac{P(G=k|X=x)}{P(G=l|X=x)}&=\\ln \\frac{f_k(x)\\pi_k}{f_l(x)\\pi_l}=\\ln\\frac{\\pi_k}{\\pi_l}+\\ln {f_k(x)}-\\ln{f_l(x)}\\\\
&=\\ln\\frac{\\pi_k}{\\pi_l}-\\frac{1}{2}(x-\\mu_k)^T\\Sigma^{-1}(x-\\mu_k)+\\frac{1}{2}(x-\\mu_l)^T\\Sigma^{-1}(x-\\mu_l)\\\\
&=\\ln\\frac{\\pi_k}{\\pi_l}-\\frac{1}{2}(\\mu_k+\\mu_l)^T\\Sigma^{-1}(\\mu_k-\\mu_l)+x^T\\Sigma^{-1}(\\mu_k-\\mu_l)
\\end{aligned}")  

注意\!這樣良好的線性性質是來自於我們假設兩個分類具有**相同的**共變異數矩陣。若我們對類別![k](https://latex.codecogs.com/png.latex?k
"k")和類別![l](https://latex.codecogs.com/png.latex?l
"l")的分界線感興趣，其分界線就是位在兩者機率密度相等之處，也就是當上式**等於0**時。

透過相同的想法，我們可以建立一個**線性判別函數(linear discriminant
function)**![\\delta\_k(x)](https://latex.codecogs.com/png.latex?%5Cdelta_k%28x%29
"\\delta_k(x)")，來決定該資料應被分配到哪一個類別，也就是![G(x)=\\mathop{\\arg\\max}\\limits\_{k}\\delta\_k(x)](https://latex.codecogs.com/png.latex?G%28x%29%3D%5Cmathop%7B%5Carg%5Cmax%7D%5Climits_%7Bk%7D%5Cdelta_k%28x%29
"G(x)=\\mathop{\\arg\\max}\\limits_{k}\\delta_k(x)")。 線性判別函數如下所示：   
![\\delta\_k(x)=x^T\\Sigma^{-1}\\mu\_k-\\frac{1}{2}\\mu\_k^T
\\Sigma^{-1}\\mu\_k +\\ln
\\pi\_k](https://latex.codecogs.com/png.latex?%5Cdelta_k%28x%29%3Dx%5ET%5CSigma%5E%7B-1%7D%5Cmu_k-%5Cfrac%7B1%7D%7B2%7D%5Cmu_k%5ET%20%5CSigma%5E%7B-1%7D%5Cmu_k%20%2B%5Cln%20%5Cpi_k
"\\delta_k(x)=x^T\\Sigma^{-1}\\mu_k-\\frac{1}{2}\\mu_k^T \\Sigma^{-1}\\mu_k +\\ln \\pi_k")  

線性判別函數的推導來自以下的成比例關係：   
![\\begin{aligned}&#10;P(G=k|X=x)&\\propto f\_k(x)\\pi\_k
\\\\&#10;&\\propto -\\frac{1}{2}(x-\\mu\_k)^T \\Sigma^{-1}
(x-\\mu\_k)+\\ln \\pi\_k =
-\\frac{1}{2}x^T\\Sigma^{-1}x+x^T\\Sigma^{-1}\\mu\_k-\\frac{1}{2}\\mu\_k^T
\\Sigma^{-1}\\mu\_k +\\ln \\pi\_k\\\\&#10;&\\propto
x^T\\Sigma^{-1}\\mu\_k-\\frac{1}{2}\\mu\_k^T \\Sigma^{-1}\\mu\_k +\\ln
\\pi\_k \\equiv
\\delta\_k(x)&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0AP%28G%3Dk%7CX%3Dx%29%26%5Cpropto%20f_k%28x%29%5Cpi_k%20%5C%5C%0A%26%5Cpropto%20-%5Cfrac%7B1%7D%7B2%7D%28x-%5Cmu_k%29%5ET%20%5CSigma%5E%7B-1%7D%20%28x-%5Cmu_k%29%2B%5Cln%20%5Cpi_k%20%3D%20-%5Cfrac%7B1%7D%7B2%7Dx%5ET%5CSigma%5E%7B-1%7Dx%2Bx%5ET%5CSigma%5E%7B-1%7D%5Cmu_k-%5Cfrac%7B1%7D%7B2%7D%5Cmu_k%5ET%20%5CSigma%5E%7B-1%7D%5Cmu_k%20%2B%5Cln%20%5Cpi_k%5C%5C%0A%26%5Cpropto%20%20x%5ET%5CSigma%5E%7B-1%7D%5Cmu_k-%5Cfrac%7B1%7D%7B2%7D%5Cmu_k%5ET%20%5CSigma%5E%7B-1%7D%5Cmu_k%20%2B%5Cln%20%5Cpi_k%20%5Cequiv%20%20%5Cdelta_k%28x%29%0A%5Cend%7Baligned%7D
"\\begin{aligned}
P(G=k|X=x)&\\propto f_k(x)\\pi_k \\\\
&\\propto -\\frac{1}{2}(x-\\mu_k)^T \\Sigma^{-1} (x-\\mu_k)+\\ln \\pi_k = -\\frac{1}{2}x^T\\Sigma^{-1}x+x^T\\Sigma^{-1}\\mu_k-\\frac{1}{2}\\mu_k^T \\Sigma^{-1}\\mu_k +\\ln \\pi_k\\\\
&\\propto  x^T\\Sigma^{-1}\\mu_k-\\frac{1}{2}\\mu_k^T \\Sigma^{-1}\\mu_k +\\ln \\pi_k \\equiv  \\delta_k(x)
\\end{aligned}")  

正如古典的統計分析一樣，我們並無法知道母體分配的參數，故我們在進行統計學習時，則選擇使用**訓練集的資料來估計母體參數**。

  
![\\begin{aligned}&#10;&\\hat{\\pi\_k}=N\_k/N \\\\&#10;
&\\hat{\\mu\_k}=\\sum\_{g\_i=k}x\_i/N\_k \\\\&#10;
&\\hat{\\Sigma}=\\sum\_{k=1}^K\\sum\_{g\_i=k}(x\_i-\\hat{\\mu\_k})(x\_i-\\hat{\\mu\_k})^T/(N-K)&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%26%5Chat%7B%5Cpi_k%7D%3DN_k%2FN%20%5C%5C%0A%20%26%5Chat%7B%5Cmu_k%7D%3D%5Csum_%7Bg_i%3Dk%7Dx_i%2FN_k%20%5C%5C%0A%20%26%5Chat%7B%5CSigma%7D%3D%5Csum_%7Bk%3D1%7D%5EK%5Csum_%7Bg_i%3Dk%7D%28x_i-%5Chat%7B%5Cmu_k%7D%29%28x_i-%5Chat%7B%5Cmu_k%7D%29%5ET%2F%28N-K%29%0A%5Cend%7Baligned%7D
"\\begin{aligned}
&\\hat{\\pi_k}=N_k/N \\\\
 &\\hat{\\mu_k}=\\sum_{g_i=k}x_i/N_k \\\\
 &\\hat{\\Sigma}=\\sum_{k=1}^K\\sum_{g_i=k}(x_i-\\hat{\\mu_k})(x_i-\\hat{\\mu_k})^T/(N-K)
\\end{aligned}")  
其中 ![N\_k](https://latex.codecogs.com/png.latex?N_k "N_k")
是類別![k](https://latex.codecogs.com/png.latex?k
"k")在訓練集中的數量(observations)。

### 2-class LDA (from Ex. 4.2)

**Suppose we have features ![x \\in
\\mathbb{R}^p](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5Ep
"x \\in \\mathbb{R}^p"), a two-class response, with class sizes
![N\_1](https://latex.codecogs.com/png.latex?N_1 "N_1"),
![N\_2](https://latex.codecogs.com/png.latex?N_2 "N_2"), and the target
coded as
![−N/N\_1](https://latex.codecogs.com/png.latex?%E2%88%92N%2FN_1
"−N/N_1"), ![N/N\_2](https://latex.codecogs.com/png.latex?N%2FN_2
"N/N_2").**

**(a) Show that the LDA rule classifies to class 2 if**   
![x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1}) \>
\\frac{1}{2}(\\hat{\\mu\_2}+\\hat{\\mu\_1})^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1})-\\ln(N\_2/N\_1)](https://latex.codecogs.com/png.latex?x%5ET%5Chat%7B%5CSigma%5E%7B-1%7D%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%20%3E%20%5Cfrac%7B1%7D%7B2%7D%28%5Chat%7B%5Cmu_2%7D%2B%5Chat%7B%5Cmu_1%7D%29%5ET%5Chat%7B%5CSigma%5E%7B-1%7D%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29-%5Cln%28N_2%2FN_1%29
"x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu_2}-\\hat{\\mu_1}) \> \\frac{1}{2}(\\hat{\\mu_2}+\\hat{\\mu_1})^T\\hat{\\Sigma^{-1}}(\\hat{\\mu_2}-\\hat{\\mu_1})-\\ln(N_2/N_1)")  
**and class 1 otherwise.**

*Sol:*

在二元的分類中，我們可以回溯到先前講到的log-odds的觀念，也就是建立此式來做比較(改寫自課本式4.9)：   
![\\begin{aligned}&#10;\\ln\\frac{P(G=2|X=x)}{P(G=1|X=x)}&#10;=\\ln\\frac{\\pi\_2}{\\pi\_1}-\\frac{1}{2}(\\mu\_2+\\mu\_1)^T\\Sigma^{-1}(\\mu\_2-\\mu\_1)+x^T\\Sigma^{-1}(\\mu\_2-\\mu\_1)&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Cln%5Cfrac%7BP%28G%3D2%7CX%3Dx%29%7D%7BP%28G%3D1%7CX%3Dx%29%7D%0A%3D%5Cln%5Cfrac%7B%5Cpi_2%7D%7B%5Cpi_1%7D-%5Cfrac%7B1%7D%7B2%7D%28%5Cmu_2%2B%5Cmu_1%29%5ET%5CSigma%5E%7B-1%7D%28%5Cmu_2-%5Cmu_1%29%2Bx%5ET%5CSigma%5E%7B-1%7D%28%5Cmu_2-%5Cmu_1%29%0A%5Cend%7Baligned%7D
"\\begin{aligned}
\\ln\\frac{P(G=2|X=x)}{P(G=1|X=x)}
=\\ln\\frac{\\pi_2}{\\pi_1}-\\frac{1}{2}(\\mu_2+\\mu_1)^T\\Sigma^{-1}(\\mu_2-\\mu_1)+x^T\\Sigma^{-1}(\\mu_2-\\mu_1)
\\end{aligned}")  
根據對數性質，若此式**大於0**，就表示
![P(G=2|X=x)](https://latex.codecogs.com/png.latex?P%28G%3D2%7CX%3Dx%29
"P(G=2|X=x)") 的機率相對於
![P(G=1|X=x)](https://latex.codecogs.com/png.latex?P%28G%3D1%7CX%3Dx%29
"P(G=1|X=x)") 來得高，故我們自然會將其分類到 class 2。 而我們根據訓練集對上式做估計並且做些許代數運算即可得到決策的函數：
  
![x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1}) \>
\\frac{1}{2}(\\hat{\\mu\_2}+\\hat{\\mu\_1})\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1})-\\ln(N\_2/N\_1)](https://latex.codecogs.com/png.latex?x%5ET%5Chat%7B%5CSigma%5E%7B-1%7D%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%20%3E%20%5Cfrac%7B1%7D%7B2%7D%28%5Chat%7B%5Cmu_2%7D%2B%5Chat%7B%5Cmu_1%7D%29%5Chat%7B%5CSigma%5E%7B-1%7D%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29-%5Cln%28N_2%2FN_1%29
"x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu_2}-\\hat{\\mu_1}) \> \\frac{1}{2}(\\hat{\\mu_2}+\\hat{\\mu_1})\\hat{\\Sigma^{-1}}(\\hat{\\mu_2}-\\hat{\\mu_1})-\\ln(N_2/N_1)")  

**(b) Consider minimization of the least squares criterion**   
![\\sum^N\_{i=1}(y\_i-\\beta\_0-x\_i^T\\beta)^2](https://latex.codecogs.com/png.latex?%5Csum%5EN_%7Bi%3D1%7D%28y_i-%5Cbeta_0-x_i%5ET%5Cbeta%29%5E2
"\\sum^N_{i=1}(y_i-\\beta_0-x_i^T\\beta)^2")  
**Show that the solution
![\\hat{\\beta}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta%7D
"\\hat{\\beta}") satisfies**

  
![\[(N-2)\\hat{\\Sigma}+N\\hat{\\Sigma}\_B\]\\beta=N(\\hat{\\mu\_2}-\\hat{\\mu\_1})](https://latex.codecogs.com/png.latex?%5B%28N-2%29%5Chat%7B%5CSigma%7D%2BN%5Chat%7B%5CSigma%7D_B%5D%5Cbeta%3DN%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29
"[(N-2)\\hat{\\Sigma}+N\\hat{\\Sigma}_B]\\beta=N(\\hat{\\mu_2}-\\hat{\\mu_1})")  
where,   
![\\hat{\\Sigma}\_B=\\frac{N\_1N\_2}{N^2}(\\hat{\\mu\_2}-\\hat{\\mu\_1})(\\hat{\\mu\_2}-\\hat{\\mu\_1})^T](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D_B%3D%5Cfrac%7BN_1N_2%7D%7BN%5E2%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%5ET
"\\hat{\\Sigma}_B=\\frac{N_1N_2}{N^2}(\\hat{\\mu_2}-\\hat{\\mu_1})(\\hat{\\mu_2}-\\hat{\\mu_1})^T")  

*Sol:*

首先，我們知道
![N=N\_1+N\_2](https://latex.codecogs.com/png.latex?N%3DN_1%2BN_2
"N=N_1+N_2")。 而這個線性迴歸方程包含常數項，則根據 Normal equation 可以得到對
![\\hat{\\beta\_0}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta_0%7D
"\\hat{\\beta_0}") 和
![\\hat{\\beta\_1}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta_1%7D
"\\hat{\\beta_1}") 的估計：

  
![ \\begin{bmatrix}&#10;\\hat{\\beta\_0}\\\\
&#10;\\hat{\\beta}&#10;\\end{bmatrix}=(X^TX)^{-1}X^Ty
](https://latex.codecogs.com/png.latex?%20%5Cbegin%7Bbmatrix%7D%0A%5Chat%7B%5Cbeta_0%7D%5C%5C%20%0A%5Chat%7B%5Cbeta%7D%0A%5Cend%7Bbmatrix%7D%3D%28X%5ETX%29%5E%7B-1%7DX%5ETy%20
" \\begin{bmatrix}
\\hat{\\beta_0}\\\\ 
\\hat{\\beta}
\\end{bmatrix}=(X^TX)^{-1}X^Ty ")  

其中 **designed matrix**
可這樣表示：(![X](https://latex.codecogs.com/png.latex?X
"X")裡面有![\\bf{1}](https://latex.codecogs.com/png.latex?%5Cbf%7B1%7D
"\\bf{1}")向量)

  
![X^TX=\\begin{bmatrix}&#10;N & \\sum\_{i=1}^N x\_i^T\\\\
&#10;\\sum\_{i=1}^N x\_i & \\sum\_{i=1}^Nx\_ix\_i^T
&#10;\\end{bmatrix}](https://latex.codecogs.com/png.latex?X%5ETX%3D%5Cbegin%7Bbmatrix%7D%0AN%20%26%20%5Csum_%7Bi%3D1%7D%5EN%20x_i%5ET%5C%5C%20%0A%5Csum_%7Bi%3D1%7D%5EN%20x_i%20%26%20%5Csum_%7Bi%3D1%7D%5ENx_ix_i%5ET%20%0A%5Cend%7Bbmatrix%7D
"X^TX=\\begin{bmatrix}
N & \\sum_{i=1}^N x_i^T\\\\ 
\\sum_{i=1}^N x_i & \\sum_{i=1}^Nx_ix_i^T 
\\end{bmatrix}")  
其中   
![\\sum\_{i=1}^Nx\_i =\\sum\_{i=1}^{N\_1}x\_i+\\sum\_{i=1}^{N\_2}x\_i =
N\_1\\hat{\\mu\_1}+
N\_2\\hat{\\mu\_2}](https://latex.codecogs.com/png.latex?%5Csum_%7Bi%3D1%7D%5ENx_i%20%3D%5Csum_%7Bi%3D1%7D%5E%7BN_1%7Dx_i%2B%5Csum_%7Bi%3D1%7D%5E%7BN_2%7Dx_i%20%3D%20N_1%5Chat%7B%5Cmu_1%7D%2B%20N_2%5Chat%7B%5Cmu_2%7D
"\\sum_{i=1}^Nx_i =\\sum_{i=1}^{N_1}x_i+\\sum_{i=1}^{N_2}x_i = N_1\\hat{\\mu_1}+ N_2\\hat{\\mu_2}")  
又知   
![\\begin{aligned}&#10;\\hat{\\Sigma} &=
\\frac{1}{N-2}\\sum\_{k=1}^2\\sum\_{g\_i=k}(x\_i-\\hat{\\mu\_k})(x\_i-\\hat{\\mu\_k})^T
\\\\&#10;&=\\frac{1}{N-2}\[\\sum\_{g\_i=1}x\_ix\_i^T-N\_1 \\mu\_1
\\mu\_1^T+\\sum\_{g\_i=2}x\_ix\_i^T-N\_2\\mu\_2
\\mu\_2^T\]&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Chat%7B%5CSigma%7D%20%26%3D%20%5Cfrac%7B1%7D%7BN-2%7D%5Csum_%7Bk%3D1%7D%5E2%5Csum_%7Bg_i%3Dk%7D%28x_i-%5Chat%7B%5Cmu_k%7D%29%28x_i-%5Chat%7B%5Cmu_k%7D%29%5ET%20%5C%5C%0A%26%3D%5Cfrac%7B1%7D%7BN-2%7D%5B%5Csum_%7Bg_i%3D1%7Dx_ix_i%5ET-N_1%20%5Cmu_1%20%5Cmu_1%5ET%2B%5Csum_%7Bg_i%3D2%7Dx_ix_i%5ET-N_2%5Cmu_2%20%5Cmu_2%5ET%5D%0A%5Cend%7Baligned%7D
"\\begin{aligned}
\\hat{\\Sigma} &= \\frac{1}{N-2}\\sum_{k=1}^2\\sum_{g_i=k}(x_i-\\hat{\\mu_k})(x_i-\\hat{\\mu_k})^T \\\\
&=\\frac{1}{N-2}[\\sum_{g_i=1}x_ix_i^T-N_1 \\mu_1 \\mu_1^T+\\sum_{g_i=2}x_ix_i^T-N_2\\mu_2 \\mu_2^T]
\\end{aligned}")  
因此   
![\\sum\_{i=1}^Nx\_ix\_i^T = (N-2)\\hat{\\Sigma}+N\_1 \\mu\_1
\\mu\_1^T+N\_2\\mu\_2
\\mu\_2^T](https://latex.codecogs.com/png.latex?%5Csum_%7Bi%3D1%7D%5ENx_ix_i%5ET%20%3D%20%28N-2%29%5Chat%7B%5CSigma%7D%2BN_1%20%5Cmu_1%20%5Cmu_1%5ET%2BN_2%5Cmu_2%20%5Cmu_2%5ET
"\\sum_{i=1}^Nx_ix_i^T = (N-2)\\hat{\\Sigma}+N_1 \\mu_1 \\mu_1^T+N_2\\mu_2 \\mu_2^T")  
若我們將屬於類別1的資料放在矩陣的前 ![N\_1](https://latex.codecogs.com/png.latex?N_1
"N_1") 個，而類別2的資料 ![N\_2](https://latex.codecogs.com/png.latex?N_2 "N_2")
個緊接在後，![X^Ty](https://latex.codecogs.com/png.latex?X%5ETy "X^Ty") 可以寫成：

  
![ X^Ty= \\begin{bmatrix}&#10;1 &\\cdots & 1 &1 & \\cdots &1 \\\\
&#10;x\_1 &\\cdots \&x\_{N\_1} \&x\_{N\_1+1} &\\cdots &
x\_{N\_1+N\_2}&#10;\\end{bmatrix}&#10;\\begin{bmatrix}&#10;-N/N\_1\\\\
&#10;\\vdots\\\\ &#10;-N/N\_1\\\\ &#10;N/N\_2\\\\ &#10;\\vdots\\\\
&#10;N/N\_2&#10;\\end{bmatrix} =\\begin{bmatrix}&#10;0
\\\\&#10;-N\\mu\_1+N\\mu\_2&#10;\\end{bmatrix}](https://latex.codecogs.com/png.latex?%20X%5ETy%3D%20%5Cbegin%7Bbmatrix%7D%0A1%20%26%5Ccdots%20%20%26%201%20%261%20%20%26%20%5Ccdots%20%261%20%5C%5C%20%0Ax_1%20%26%5Ccdots%20%20%26x_%7BN_1%7D%20%20%26x_%7BN_1%2B1%7D%20%20%26%5Ccdots%20%20%26%20x_%7BN_1%2BN_2%7D%0A%5Cend%7Bbmatrix%7D%0A%5Cbegin%7Bbmatrix%7D%0A-N%2FN_1%5C%5C%20%0A%5Cvdots%5C%5C%20%0A-N%2FN_1%5C%5C%20%0AN%2FN_2%5C%5C%20%0A%5Cvdots%5C%5C%20%0AN%2FN_2%0A%5Cend%7Bbmatrix%7D%20%3D%5Cbegin%7Bbmatrix%7D%0A0%20%5C%5C%0A-N%5Cmu_1%2BN%5Cmu_2%0A%5Cend%7Bbmatrix%7D
" X^Ty= \\begin{bmatrix}
1 &\\cdots  & 1 &1  & \\cdots &1 \\\\ 
x_1 &\\cdots  &x_{N_1}  &x_{N_1+1}  &\\cdots  & x_{N_1+N_2}
\\end{bmatrix}
\\begin{bmatrix}
-N/N_1\\\\ 
\\vdots\\\\ 
-N/N_1\\\\ 
N/N_2\\\\ 
\\vdots\\\\ 
N/N_2
\\end{bmatrix} =\\begin{bmatrix}
0 \\\\
-N\\mu_1+N\\mu_2
\\end{bmatrix}")  

整理先前計算出的這些東西代入 Normal equation，整理後可得：

(註：![\\beta\_0=(-\\frac{N\_1}{N}\\mu\_1^T-\\frac{N\_2}{N}\\mu\_2^T)\\beta](https://latex.codecogs.com/png.latex?%5Cbeta_0%3D%28-%5Cfrac%7BN_1%7D%7BN%7D%5Cmu_1%5ET-%5Cfrac%7BN_2%7D%7BN%7D%5Cmu_2%5ET%29%5Cbeta
"\\beta_0=(-\\frac{N_1}{N}\\mu_1^T-\\frac{N_2}{N}\\mu_2^T)\\beta"))

  
![(N\_1\\mu\_1+N\_2\\mu\_2)(-\\frac{N\_1}{N}\\mu\_1^T-\\frac{N\_2}{N}\\mu\_2^T)\\beta+((N-2)\\hat{\\Sigma}
+N\_1\\mu\_1\\mu\_1^T+N\_2\\mu\_2\\mu\_2^T)\\beta=N(\\mu\_2-\\mu\_1)](https://latex.codecogs.com/png.latex?%28N_1%5Cmu_1%2BN_2%5Cmu_2%29%28-%5Cfrac%7BN_1%7D%7BN%7D%5Cmu_1%5ET-%5Cfrac%7BN_2%7D%7BN%7D%5Cmu_2%5ET%29%5Cbeta%2B%28%28N-2%29%5Chat%7B%5CSigma%7D%20%2BN_1%5Cmu_1%5Cmu_1%5ET%2BN_2%5Cmu_2%5Cmu_2%5ET%29%5Cbeta%3DN%28%5Cmu_2-%5Cmu_1%29
"(N_1\\mu_1+N_2\\mu_2)(-\\frac{N_1}{N}\\mu_1^T-\\frac{N_2}{N}\\mu_2^T)\\beta+((N-2)\\hat{\\Sigma} +N_1\\mu_1\\mu_1^T+N_2\\mu_2\\mu_2^T)\\beta=N(\\mu_2-\\mu_1)")  

經過一些運算，並引入題目對
![\\hat{\\Sigma\_B}](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma_B%7D
"\\hat{\\Sigma_B}") 的定義，即可得：   
![\[(N-2)\\hat{\\Sigma}+N\\hat{\\Sigma}\_B\]\\beta=N(\\hat{\\mu\_2}-\\hat{\\mu\_1})](https://latex.codecogs.com/png.latex?%5B%28N-2%29%5Chat%7B%5CSigma%7D%2BN%5Chat%7B%5CSigma%7D_B%5D%5Cbeta%3DN%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29
"[(N-2)\\hat{\\Sigma}+N\\hat{\\Sigma}_B]\\beta=N(\\hat{\\mu_2}-\\hat{\\mu_1})")  

**(c) Hence show that ![\\hat{\\Sigma}\_B
\\beta](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D_B%20%5Cbeta
"\\hat{\\Sigma}_B \\beta") is in the direction ![(\\hat{\\mu\_2} -
\\hat{\\mu\_1})](https://latex.codecogs.com/png.latex?%28%5Chat%7B%5Cmu_2%7D%20-%20%5Chat%7B%5Cmu_1%7D%29
"(\\hat{\\mu_2} - \\hat{\\mu_1})") and thus**   
![\\hat{\\beta}\\propto \\hat{\\Sigma}^{-1} (\\hat{\\mu\_2} -
\\hat{\\mu\_1})](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta%7D%5Cpropto%20%5Chat%7B%5CSigma%7D%5E%7B-1%7D%20%28%5Chat%7B%5Cmu_2%7D%20-%20%5Chat%7B%5Cmu_1%7D%29
"\\hat{\\beta}\\propto \\hat{\\Sigma}^{-1} (\\hat{\\mu_2} - \\hat{\\mu_1})")  
**Therefore the least-squares regression coefficient is identical to the
LDA coefficient, up to a scalar multiple.**

*Sol:*

若我們直接計算 ![\\hat{\\Sigma}\_B
\\beta](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D_B%20%5Cbeta
"\\hat{\\Sigma}_B \\beta")，由於 ![(\\hat{\\mu\_2}-\\hat{\\mu\_1})^T
\\beta](https://latex.codecogs.com/png.latex?%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%5ET%20%5Cbeta
"(\\hat{\\mu_2}-\\hat{\\mu_1})^T \\beta") 內積後是一個scalar。也就是

  
![\\hat{\\Sigma}\_B
\\beta=\\frac{N\_1N\_2}{N^2}(\\hat{\\mu\_2}-\\hat{\\mu\_1}){\\color{Red}
(\\hat{\\mu\_2}-\\hat{\\mu\_1})^T
\\beta}=\\frac{N\_1N\_2}{N^2}(\\hat{\\mu\_2}-\\hat{\\mu\_1}){\\color{Red}
c}](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D_B%20%5Cbeta%3D%5Cfrac%7BN_1N_2%7D%7BN%5E2%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%7B%5Ccolor%7BRed%7D%20%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%5ET%20%5Cbeta%7D%3D%5Cfrac%7BN_1N_2%7D%7BN%5E2%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%7B%5Ccolor%7BRed%7D%20c%7D
"\\hat{\\Sigma}_B \\beta=\\frac{N_1N_2}{N^2}(\\hat{\\mu_2}-\\hat{\\mu_1}){\\color{Red} (\\hat{\\mu_2}-\\hat{\\mu_1})^T \\beta}=\\frac{N_1N_2}{N^2}(\\hat{\\mu_2}-\\hat{\\mu_1}){\\color{Red} c}")  

換句話說， ![\\hat{\\Sigma}\_B
\\beta](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D_B%20%5Cbeta
"\\hat{\\Sigma}_B \\beta") 和
![(\\hat{\\mu\_2}-\\hat{\\mu\_1})](https://latex.codecogs.com/png.latex?%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29
"(\\hat{\\mu_2}-\\hat{\\mu_1})") 差了 ![{\\color{Red}
\\frac{N\_1N\_2}{N^2}c}](https://latex.codecogs.com/png.latex?%7B%5Ccolor%7BRed%7D%20%5Cfrac%7BN_1N_2%7D%7BN%5E2%7Dc%7D
"{\\color{Red} \\frac{N_1N_2}{N^2}c}") 倍。又這些常數都為正，故得知：   
![ \\hat{\\beta} \\propto \\hat{\\Sigma}^{-1}
(\\hat{\\mu\_2}-\\hat{\\mu\_1})
](https://latex.codecogs.com/png.latex?%20%5Chat%7B%5Cbeta%7D%20%5Cpropto%20%5Chat%7B%5CSigma%7D%5E%7B-1%7D%20%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%20
" \\hat{\\beta} \\propto \\hat{\\Sigma}^{-1} (\\hat{\\mu_2}-\\hat{\\mu_1}) ")  

此即表示線性迴歸與LDA其實是有相像的運算邏輯。

**(d) Show that this result holds for any (distinct) coding of the two
classes.**

*Sol:*

在上一題中，我們並沒有特別對 ![t\_k](https://latex.codecogs.com/png.latex?t_k "t_k")
有特別假設，這樣的性質是來自於一開始對
![\\hat{\\Sigma}\_B](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D_B
"\\hat{\\Sigma}_B") 的設計，並且與
![\\beta](https://latex.codecogs.com/png.latex?%5Cbeta "\\beta")
內積後出現純數而推演出這樣的比例關係。

**(e) Find the solution
![\\hat{\\beta\_0}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cbeta_0%7D
"\\hat{\\beta_0}") (up to the same scalar multiple as in (c), and hence
the predicted value
![\\hat{f}(x)=\\hat{\\beta\_0}+x^T\\hat{\\beta}](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28x%29%3D%5Chat%7B%5Cbeta_0%7D%2Bx%5ET%5Chat%7B%5Cbeta%7D
"\\hat{f}(x)=\\hat{\\beta_0}+x^T\\hat{\\beta}"). Consider the following
rule: classify to class 2 if ![f(x)
\> 0](https://latex.codecogs.com/png.latex?f%28x%29%20%3E%200
"f(x) \> 0")and class 1 otherwise. Show this is not the same as the LDA
rule unless the classes have equal numbers of observations.**

*Sol:*

在先前的討論，我們知道：   
![\\beta\_0 =
-\\frac{1}{N}(N\_1\\hat{\\mu}^T\_1+N\_2\\hat{\\mu}^T\_2)\\hat{\\beta}](https://latex.codecogs.com/png.latex?%5Cbeta_0%20%3D%20-%5Cfrac%7B1%7D%7BN%7D%28N_1%5Chat%7B%5Cmu%7D%5ET_1%2BN_2%5Chat%7B%5Cmu%7D%5ET_2%29%5Chat%7B%5Cbeta%7D
"\\beta_0 = -\\frac{1}{N}(N_1\\hat{\\mu}^T_1+N_2\\hat{\\mu}^T_2)\\hat{\\beta}")  

我們可改寫
![\\hat{f}(x)](https://latex.codecogs.com/png.latex?%5Chat%7Bf%7D%28x%29
"\\hat{f}(x)") 如下：

  
![\\begin{aligned}&#10;\\hat{f}(x)&=-\\frac{1}{N}(N\_1\\hat{\\mu}^T\_1+N\_2\\hat{\\mu}^T\_2-Nx^T)
\\hat{\\beta
}\\\\&#10;&=-\\frac{1}{N}(N\_1\\hat{\\mu}^T\_1+N\_2\\hat{\\mu}^T\_2-Nx^T)
{\\color{Red} \\lambda
\\hat{\\Sigma}^{-1}(\\hat{\\mu\_2}-\\hat{\\mu\_1})}&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0A%5Chat%7Bf%7D%28x%29%26%3D-%5Cfrac%7B1%7D%7BN%7D%28N_1%5Chat%7B%5Cmu%7D%5ET_1%2BN_2%5Chat%7B%5Cmu%7D%5ET_2-Nx%5ET%29%20%20%20%20%5Chat%7B%5Cbeta%20%7D%5C%5C%0A%26%3D-%5Cfrac%7B1%7D%7BN%7D%28N_1%5Chat%7B%5Cmu%7D%5ET_1%2BN_2%5Chat%7B%5Cmu%7D%5ET_2-Nx%5ET%29%20%7B%5Ccolor%7BRed%7D%20%5Clambda%20%5Chat%7B%5CSigma%7D%5E%7B-1%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%7D%0A%5Cend%7Baligned%7D
"\\begin{aligned}
\\hat{f}(x)&=-\\frac{1}{N}(N_1\\hat{\\mu}^T_1+N_2\\hat{\\mu}^T_2-Nx^T)    \\hat{\\beta }\\\\
&=-\\frac{1}{N}(N_1\\hat{\\mu}^T_1+N_2\\hat{\\mu}^T_2-Nx^T) {\\color{Red} \\lambda \\hat{\\Sigma}^{-1}(\\hat{\\mu_2}-\\hat{\\mu_1})}
\\end{aligned}")  

經過展開，並重整，則可得到下式：

  
![x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1}) \>
\\frac{1}{N}(N\_2\\hat{\\mu\_2}+N\_1\\hat{\\mu\_1})^T\\hat{\\Sigma^{-1}}(\\hat{\\mu\_2}-\\hat{\\mu\_1})](https://latex.codecogs.com/png.latex?x%5ET%5Chat%7B%5CSigma%5E%7B-1%7D%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29%20%3E%20%5Cfrac%7B1%7D%7BN%7D%28N_2%5Chat%7B%5Cmu_2%7D%2BN_1%5Chat%7B%5Cmu_1%7D%29%5ET%5Chat%7B%5CSigma%5E%7B-1%7D%7D%28%5Chat%7B%5Cmu_2%7D-%5Chat%7B%5Cmu_1%7D%29
"x^T\\hat{\\Sigma^{-1}}(\\hat{\\mu_2}-\\hat{\\mu_1}) \> \\frac{1}{N}(N_2\\hat{\\mu_2}+N_1\\hat{\\mu_1})^T\\hat{\\Sigma^{-1}}(\\hat{\\mu_2}-\\hat{\\mu_1})")  
若 ![N\_1=N\_2](https://latex.codecogs.com/png.latex?N_1%3DN_2 "N_1=N_2")
，則
![N\_1/N=N\_2/N=1/2](https://latex.codecogs.com/png.latex?N_1%2FN%3DN_2%2FN%3D1%2F2
"N_1/N=N_2/N=1/2")，而
![\\ln(N\_2/N\_1)=\\ln 1=0](https://latex.codecogs.com/png.latex?%5Cln%28N_2%2FN_1%29%3D%5Cln%201%3D0
"\\ln(N_2/N_1)=\\ln 1=0")

這樣的決策準則就跟LDA一樣了！

## Quadratic Discriminant Analysis (QDA)

還記得先前在推導 LDA 時除了常態假設之外，還有一個變異數矩陣相等的假設，若我們放寬這個假設，這個方法就變成 QDA ，Quadratic
的部分就是來自於多變量常態分配中在 ![e](https://latex.codecogs.com/png.latex?e "e")
上的 ![x](https://latex.codecogs.com/png.latex?x "x") 二次式不會被對消。

二次判別函數(Quadratic Discriminant function)如下：

  
![\\delta\_k(x)=-\\frac{1}{2}\\ln|\\Sigma\_k|-\\frac{1}{2}(x-\\mu\_k)^T\\Sigma\_k^{-1}(x-\\mu\_k)+\\ln
\\pi\_k](https://latex.codecogs.com/png.latex?%5Cdelta_k%28x%29%3D-%5Cfrac%7B1%7D%7B2%7D%5Cln%7C%5CSigma_k%7C-%5Cfrac%7B1%7D%7B2%7D%28x-%5Cmu_k%29%5ET%5CSigma_k%5E%7B-1%7D%28x-%5Cmu_k%29%2B%5Cln%20%5Cpi_k
"\\delta_k(x)=-\\frac{1}{2}\\ln|\\Sigma_k|-\\frac{1}{2}(x-\\mu_k)^T\\Sigma_k^{-1}(x-\\mu_k)+\\ln \\pi_k")  

個類別的分界線可以這樣表示

  
![\\{x|\\delta\_k(x)=\\delta\_l(x)
\\}](https://latex.codecogs.com/png.latex?%5C%7Bx%7C%5Cdelta_k%28x%29%3D%5Cdelta_l%28x%29%20%5C%7D
"\\{x|\\delta_k(x)=\\delta_l(x) \\}")  
QDA 有參數太多的缺點。

## Regulized Discriminant Analysis (RDA)

![](https://esl.hohoweiya.xyz/img/04/fig4.7.png)

Friedman(1989) 提出了介於LDA與QDA之間的方法，稱之為 **Regulized Discriminant Analysis
(RDA)**。正則化的共變異數矩陣如下所示：

  
![\\hat{\\Sigma}\_k(\\alpha)=\\alpha
\\hat{\\Sigma}\_k+(1-\\alpha)\\hat{\\Sigma},\\ \\alpha
\\in\[0,1\]](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D_k%28%5Calpha%29%3D%5Calpha%20%5Chat%7B%5CSigma%7D_k%2B%281-%5Calpha%29%5Chat%7B%5CSigma%7D%2C%5C%20%5Calpha%20%5Cin%5B0%2C1%5D
"\\hat{\\Sigma}_k(\\alpha)=\\alpha \\hat{\\Sigma}_k+(1-\\alpha)\\hat{\\Sigma},\\ \\alpha \\in[0,1]")  
在此，
![\\hat{\\Sigma}](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D
"\\hat{\\Sigma}") 是LDA所做的共變數矩陣，而
![\\hat{\\Sigma}\_k](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D_k
"\\hat{\\Sigma}_k") 為QDA的共變數矩陣。

by 課本   
![\\hat{\\Sigma}(\\gamma)=\\gamma\\hat{\\Sigma}+(1-\\gamma)\\hat{\\sigma}^2I](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D%28%5Cgamma%29%3D%5Cgamma%5Chat%7B%5CSigma%7D%2B%281-%5Cgamma%29%5Chat%7B%5Csigma%7D%5E2I
"\\hat{\\Sigma}(\\gamma)=\\gamma\\hat{\\Sigma}+(1-\\gamma)\\hat{\\sigma}^2I")  

## LDA 的計算

## Reduced-Rank LDA

## LDA 和 linear regression 的關係 (from Ex. 4.3)

**Suppose that we transform the original predictors
![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D
"\\mathbf{X}") to
![\\mathbf{\\hat{Y}}](https://latex.codecogs.com/png.latex?%5Cmathbf%7B%5Chat%7BY%7D%7D
"\\mathbf{\\hat{Y}}") by taking the predicted values under linear
regression. Show that LDA using
![\\bf{\\hat{Y}}](https://latex.codecogs.com/png.latex?%5Cbf%7B%5Chat%7BY%7D%7D
"\\bf{\\hat{Y}}") is identical to using LDA in the original space.**

*Sol:* 根據題意，我們知道 ![x \\in
\\mathbb{R}^p](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5Ep
"x \\in \\mathbb{R}^p") 和 ![y \\in
\\mathbb{R}^k](https://latex.codecogs.com/png.latex?y%20%5Cin%20%5Cmathbb%7BR%7D%5Ek
"y \\in \\mathbb{R}^k") ，且：   
![\\begin{aligned} &#10;& \\hat{y}=\\hat{\\mathbf{B}}x \\\\&#10;&
\\mathbf{\\hat{Y}} = \\mathbf{X} \\hat{\\mathbf{B}}=
\\mathbf{X}(\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{Y}&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%0A%26%20%5Chat%7By%7D%3D%5Chat%7B%5Cmathbf%7BB%7D%7Dx%20%5C%5C%0A%26%20%5Cmathbf%7B%5Chat%7BY%7D%7D%20%3D%20%5Cmathbf%7BX%7D%20%5Chat%7B%5Cmathbf%7BB%7D%7D%3D%20%5Cmathbf%7BX%7D%28%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BX%7D%29%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BY%7D%0A%5Cend%7Baligned%7D
"\\begin{aligned} 
& \\hat{y}=\\hat{\\mathbf{B}}x \\\\
& \\mathbf{\\hat{Y}} = \\mathbf{X} \\hat{\\mathbf{B}}= \\mathbf{X}(\\mathbf{X}^T\\mathbf{X})^{-1}\\mathbf{X}^T\\mathbf{Y}
\\end{aligned}")  

對於任意的類別 ![k](https://latex.codecogs.com/png.latex?k "k")
，我們可以建構出該類別的平均數，我們定義類別
![k](https://latex.codecogs.com/png.latex?k "k") 對應其
![X](https://latex.codecogs.com/png.latex?X "X") 以及
![Y](https://latex.codecogs.com/png.latex?Y "Y") 的平均數為：

  
![\\begin{aligned} &#10;& \\mu\_k = \\frac{1}{N\_k}\\sum\_{g\_i=k}x^T\_i
\\\\&#10;& \\hat{\\mu\_k} =
\\mathbf{B}^T\\mu\_k&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%0A%26%20%5Cmu_k%20%3D%20%5Cfrac%7B1%7D%7BN_k%7D%5Csum_%7Bg_i%3Dk%7Dx%5ET_i%20%5C%5C%0A%26%20%5Chat%7B%5Cmu_k%7D%20%3D%20%5Cmathbf%7BB%7D%5ET%5Cmu_k%0A%5Cend%7Baligned%7D
"\\begin{aligned} 
& \\mu_k = \\frac{1}{N_k}\\sum_{g_i=k}x^T_i \\\\
& \\hat{\\mu_k} = \\mathbf{B}^T\\mu_k
\\end{aligned}")  

若我們定義
![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D
"\\mathbf{X}") 的共變數矩陣為
![{\\Sigma}](https://latex.codecogs.com/png.latex?%7B%5CSigma%7D
"{\\Sigma}") ，則
![\\hat{\\mathbf{Y}}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cmathbf%7BY%7D%7D
"\\hat{\\mathbf{Y}}")的共變數矩陣則為：   
![\\hat{\\Sigma} = \\mathbf{B}^T \\Sigma
\\mathbf{B}](https://latex.codecogs.com/png.latex?%5Chat%7B%5CSigma%7D%20%3D%20%5Cmathbf%7BB%7D%5ET%20%5CSigma%20%5Cmathbf%7BB%7D
"\\hat{\\Sigma} = \\mathbf{B}^T \\Sigma \\mathbf{B}")  
其中

  
![\\begin{aligned} &#10;& \\Sigma
=\\frac{1}{N-K}X^T(I-YD^{-1}Y^T)X\\\\&#10;&,where\\ D
=\\begin{bmatrix}&#10;n\_1 & 0 & \\cdots &0 \\\\ &#10;0 & n\_2 &\\cdots
&0 \\\\ &#10; \\vdots&\\vdots & \\vdots & \\vdots\\\\ &#10;0 &0 &
\\cdots &
n\_k&#10;\\end{bmatrix}&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%0A%26%20%5CSigma%20%3D%5Cfrac%7B1%7D%7BN-K%7DX%5ET%28I-YD%5E%7B-1%7DY%5ET%29X%5C%5C%0A%26%2Cwhere%5C%20%20D%20%3D%5Cbegin%7Bbmatrix%7D%0An_1%20%26%200%20%20%26%20%5Ccdots%20%260%20%5C%5C%20%0A0%20%26%20n_2%20%26%5Ccdots%20%20%260%20%5C%5C%20%0A%20%5Cvdots%26%5Cvdots%20%20%26%20%5Cvdots%20%26%20%5Cvdots%5C%5C%20%0A0%20%260%20%20%26%20%5Ccdots%20%26%20n_k%0A%5Cend%7Bbmatrix%7D%0A%5Cend%7Baligned%7D
"\\begin{aligned} 
& \\Sigma =\\frac{1}{N-K}X^T(I-YD^{-1}Y^T)X\\\\
&,where\\  D =\\begin{bmatrix}
n_1 & 0  & \\cdots &0 \\\\ 
0 & n_2 &\\cdots  &0 \\\\ 
 \\vdots&\\vdots  & \\vdots & \\vdots\\\\ 
0 &0  & \\cdots & n_k
\\end{bmatrix}
\\end{aligned}")  

若我們直接用
![\\hat{\\mathbf{Y}}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cmathbf%7BY%7D%7D
"\\hat{\\mathbf{Y}}") 的資料做 LDA，則判別函數為：

  
![\\delta\_k(\\hat{Y})=
\\hat{\\mathbf{Y}}\\hat{\\Sigma}^{-1}\\hat{\\mu\_k}-\\frac{1}{2}\\hat{\\mu\_k}^T\\hat{\\Sigma}^{-1}\\hat{\\mu\_k}+\\ln
\\pi\_k](https://latex.codecogs.com/png.latex?%5Cdelta_k%28%5Chat%7BY%7D%29%3D%20%5Chat%7B%5Cmathbf%7BY%7D%7D%5Chat%7B%5CSigma%7D%5E%7B-1%7D%5Chat%7B%5Cmu_k%7D-%5Cfrac%7B1%7D%7B2%7D%5Chat%7B%5Cmu_k%7D%5ET%5Chat%7B%5CSigma%7D%5E%7B-1%7D%5Chat%7B%5Cmu_k%7D%2B%5Cln%20%5Cpi_k
"\\delta_k(\\hat{Y})= \\hat{\\mathbf{Y}}\\hat{\\Sigma}^{-1}\\hat{\\mu_k}-\\frac{1}{2}\\hat{\\mu_k}^T\\hat{\\Sigma}^{-1}\\hat{\\mu_k}+\\ln \\pi_k")  

第一項可以改寫如下(全部寫成矩陣的樣子)：(
![B=(X^TX)^{-1}X^TY](https://latex.codecogs.com/png.latex?B%3D%28X%5ETX%29%5E%7B-1%7DX%5ETY
"B=(X^TX)^{-1}X^TY") )

  
![\\begin{aligned}
&#10;\\hat{\\mathbf{Y}}\\hat{\\Sigma}^{-1}\\hat{\\mu}&=(XB)(B^T\\Sigma
B)^{-1}(B^TX^TYD^{-1})
\\\\&#10;&=X\\hat{\\Sigma}^{-1}\\mu&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%0A%5Chat%7B%5Cmathbf%7BY%7D%7D%5Chat%7B%5CSigma%7D%5E%7B-1%7D%5Chat%7B%5Cmu%7D%26%3D%28XB%29%28B%5ET%5CSigma%20B%29%5E%7B-1%7D%28B%5ETX%5ETYD%5E%7B-1%7D%29%20%5C%5C%0A%26%3DX%5Chat%7B%5CSigma%7D%5E%7B-1%7D%5Cmu%0A%5Cend%7Baligned%7D
"\\begin{aligned} 
\\hat{\\mathbf{Y}}\\hat{\\Sigma}^{-1}\\hat{\\mu}&=(XB)(B^T\\Sigma B)^{-1}(B^TX^TYD^{-1}) \\\\
&=X\\hat{\\Sigma}^{-1}\\mu
\\end{aligned}")  

這就是使用
![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D
"\\mathbf{X}") 進行LDA的其中一部分。

而第二項，若將全部 ![K](https://latex.codecogs.com/png.latex?K "K")
種類的平均搜集起來，寫成矩陣
![\\hat{\\mu}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cmu%7D
"\\hat{\\mu}") 可改寫為：

  
![\\begin{aligned}
&#10;\\hat{\\mu\_k}^T\\hat{\\Sigma}^{-1}\\hat{\\mu}&=(B^T
\\mu\_k)^T\\hat{\\Sigma}^{-1}(B^TX^TYD^{-1})\\\\&#10;&=\\mu\_k^TB\\hat{\\Sigma}^{-1}(B^TX^TYD^{-1})\\\\&#10;&=\\mu\_k^TB(B^T\\Sigma
B)^{-1}B^TX^TYD^{-1}\\\\&#10;&=\\mu\_k^T\\Sigma^{-1}\\mu&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%0A%5Chat%7B%5Cmu_k%7D%5ET%5Chat%7B%5CSigma%7D%5E%7B-1%7D%5Chat%7B%5Cmu%7D%26%3D%28B%5ET%20%5Cmu_k%29%5ET%5Chat%7B%5CSigma%7D%5E%7B-1%7D%28B%5ETX%5ETYD%5E%7B-1%7D%29%5C%5C%0A%26%3D%5Cmu_k%5ETB%5Chat%7B%5CSigma%7D%5E%7B-1%7D%28B%5ETX%5ETYD%5E%7B-1%7D%29%5C%5C%0A%26%3D%5Cmu_k%5ETB%28B%5ET%5CSigma%20B%29%5E%7B-1%7DB%5ETX%5ETYD%5E%7B-1%7D%5C%5C%0A%26%3D%5Cmu_k%5ET%5CSigma%5E%7B-1%7D%5Cmu%0A%5Cend%7Baligned%7D
"\\begin{aligned} 
\\hat{\\mu_k}^T\\hat{\\Sigma}^{-1}\\hat{\\mu}&=(B^T \\mu_k)^T\\hat{\\Sigma}^{-1}(B^TX^TYD^{-1})\\\\
&=\\mu_k^TB\\hat{\\Sigma}^{-1}(B^TX^TYD^{-1})\\\\
&=\\mu_k^TB(B^T\\Sigma B)^{-1}B^TX^TYD^{-1}\\\\
&=\\mu_k^T\\Sigma^{-1}\\mu
\\end{aligned}")  

此即使用
![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D
"\\mathbf{X}") 做LDA的另外一部分。 所以我們可以知道，若我們透過迴歸方程的模式找出
![\\hat{\\mathbf{Y}}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cmathbf%7BY%7D%7D
"\\hat{\\mathbf{Y}}") ，我們便可從對
![\\hat{\\mathbf{Y}}](https://latex.codecogs.com/png.latex?%5Chat%7B%5Cmathbf%7BY%7D%7D
"\\hat{\\mathbf{Y}}") 做LDA轉換成對
![\\mathbf{X}](https://latex.codecogs.com/png.latex?%5Cmathbf%7BX%7D
"\\mathbf{X}") 做LDA。

## Logistic Regression

羅吉斯迴歸(Logistic Regression)是針對
![K](https://latex.codecogs.com/png.latex?K "K")
個類別的後驗機率做一個線性的的模型，而跟一般線性迴歸不同的地方是在於，羅吉斯迴歸透過對數的轉換將機率的取值保留在![\[0,1\]](https://latex.codecogs.com/png.latex?%5B0%2C1%5D
"[0,1]")之間。若我們的資料有 ![K](https://latex.codecogs.com/png.latex?K "K")
類別，則羅吉斯回歸透過建立 ![K-1](https://latex.codecogs.com/png.latex?K-1
"K-1") 個 **log-odds** 來建構我們所要的模型，少的那一個類別就拿來作為對照組之用。 以
![K](https://latex.codecogs.com/png.latex?K "K") 作為基準，Logistic
Regression 會建立這些模型關係：

  
![\\begin{aligned} &#10;\\ln \\frac{P(G=1|X=x)}{P(G=K|X=x)} &=
\\beta\_{10}+\\beta\_1^Tx\\\\&#10;\\ln \\frac{P(G=2|X=x)}{P(G=K|X=x)} &=
\\beta\_{20}+\\beta\_2^Tx \\\\&#10;& \\vdots \\\\&#10;\\ln
\\frac{P(G=K-1|X=x)}{P(G=K|X=x)} &=
\\beta\_{(K-1)0}+\\beta\_{K-1}^Tx&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%0A%5Cln%20%5Cfrac%7BP%28G%3D1%7CX%3Dx%29%7D%7BP%28G%3DK%7CX%3Dx%29%7D%20%26%3D%20%5Cbeta_%7B10%7D%2B%5Cbeta_1%5ETx%5C%5C%0A%5Cln%20%5Cfrac%7BP%28G%3D2%7CX%3Dx%29%7D%7BP%28G%3DK%7CX%3Dx%29%7D%20%26%3D%20%5Cbeta_%7B20%7D%2B%5Cbeta_2%5ETx%20%5C%5C%0A%26%20%5Cvdots%20%5C%5C%0A%5Cln%20%5Cfrac%7BP%28G%3DK-1%7CX%3Dx%29%7D%7BP%28G%3DK%7CX%3Dx%29%7D%20%26%3D%20%5Cbeta_%7B%28K-1%290%7D%2B%5Cbeta_%7BK-1%7D%5ETx%0A%5Cend%7Baligned%7D
"\\begin{aligned} 
\\ln \\frac{P(G=1|X=x)}{P(G=K|X=x)} &= \\beta_{10}+\\beta_1^Tx\\\\
\\ln \\frac{P(G=2|X=x)}{P(G=K|X=x)} &= \\beta_{20}+\\beta_2^Tx \\\\
& \\vdots \\\\
\\ln \\frac{P(G=K-1|X=x)}{P(G=K|X=x)} &= \\beta_{(K-1)0}+\\beta_{K-1}^Tx
\\end{aligned}")  

我們可以對這些式子取指數回來，可得：

  
![P(G=l|X=x) =P(G=K|X=x) e^{\\beta\_{l0}+\\beta\_l^Tx},
l=1,2,\\cdots,N-1](https://latex.codecogs.com/png.latex?P%28G%3Dl%7CX%3Dx%29%20%3DP%28G%3DK%7CX%3Dx%29%20e%5E%7B%5Cbeta_%7Bl0%7D%2B%5Cbeta_l%5ETx%7D%2C%20l%3D1%2C2%2C%5Ccdots%2CN-1
"P(G=l|X=x) =P(G=K|X=x) e^{\\beta_{l0}+\\beta_l^Tx}, l=1,2,\\cdots,N-1")  

又

  
![\\begin{aligned} &#10;P(G=K|X=x)
&= 1-\\sum\_{l=1}^{K-1}P(G=l|X=x)\\\\&=1-P(G=K|X=x)\\sum\_{l=1}^{K-1}e^{\\beta\_{l0}+\\beta\_l^Tx}&#10;\\end{aligned}
](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%0AP%28G%3DK%7CX%3Dx%29%20%26%3D%201-%5Csum_%7Bl%3D1%7D%5E%7BK-1%7DP%28G%3Dl%7CX%3Dx%29%5C%5C%26%3D1-P%28G%3DK%7CX%3Dx%29%5Csum_%7Bl%3D1%7D%5E%7BK-1%7De%5E%7B%5Cbeta_%7Bl0%7D%2B%5Cbeta_l%5ETx%7D%0A%5Cend%7Baligned%7D%20
"\\begin{aligned} 
P(G=K|X=x) &= 1-\\sum_{l=1}^{K-1}P(G=l|X=x)\\\\&=1-P(G=K|X=x)\\sum_{l=1}^{K-1}e^{\\beta_{l0}+\\beta_l^Tx}
\\end{aligned} ")  

故可求解出：   
![P(G=K|X=x) =
\\frac{1}{1+\\sum\_{l=1}^{K-1}e^{\\beta\_{l0}+\\beta\_l^Tx} }
](https://latex.codecogs.com/png.latex?P%28G%3DK%7CX%3Dx%29%20%3D%20%5Cfrac%7B1%7D%7B1%2B%5Csum_%7Bl%3D1%7D%5E%7BK-1%7De%5E%7B%5Cbeta_%7Bl0%7D%2B%5Cbeta_l%5ETx%7D%20%7D%20
"P(G=K|X=x) = \\frac{1}{1+\\sum_{l=1}^{K-1}e^{\\beta_{l0}+\\beta_l^Tx} } ")  
以及,   
![P(G=k|X=x) = \\frac{e^{\\beta\_{k0}+\\beta\_k^Tx}
}{1+\\sum\_{l=1}^{K-1}e^{\\beta\_{l0}+\\beta\_l^Tx}
}](https://latex.codecogs.com/png.latex?P%28G%3Dk%7CX%3Dx%29%20%3D%20%5Cfrac%7Be%5E%7B%5Cbeta_%7Bk0%7D%2B%5Cbeta_k%5ETx%7D%20%7D%7B1%2B%5Csum_%7Bl%3D1%7D%5E%7BK-1%7De%5E%7B%5Cbeta_%7Bl0%7D%2B%5Cbeta_l%5ETx%7D%20%7D
"P(G=k|X=x) = \\frac{e^{\\beta_{k0}+\\beta_k^Tx} }{1+\\sum_{l=1}^{K-1}e^{\\beta_{l0}+\\beta_l^Tx} }")  

若我們簡化表示，將這些參數定義為：

  
![\\theta =
\\{\\beta\_{10},\\beta\_1^T,\\cdots,\\beta\_{(K-1)0},\\beta\_{(K-1)}^T
\\}](https://latex.codecogs.com/png.latex?%5Ctheta%20%3D%20%5C%7B%5Cbeta_%7B10%7D%2C%5Cbeta_1%5ET%2C%5Ccdots%2C%5Cbeta_%7B%28K-1%290%7D%2C%5Cbeta_%7B%28K-1%29%7D%5ET%20%20%5C%7D
"\\theta = \\{\\beta_{10},\\beta_1^T,\\cdots,\\beta_{(K-1)0},\\beta_{(K-1)}^T  \\}")  
則類別![k](https://latex.codecogs.com/png.latex?k "k")機率可以表示成：
![P(G=k|X=x)=p\_k(x;\\theta)](https://latex.codecogs.com/png.latex?P%28G%3Dk%7CX%3Dx%29%3Dp_k%28x%3B%5Ctheta%29
"P(G=k|X=x)=p_k(x;\\theta)")

### 配適 Logistic Regression 的參數

在共有 ![K](https://latex.codecogs.com/png.latex?K "K")
類的資料中，可以把資料想成是從**多項分配(multinomial)**取出來的。因此
**likelihood** 可以想成是：

  
![L = \\prod
\_{i=1}^Np\_{g\_i}(x\_i)](https://latex.codecogs.com/png.latex?L%20%3D%20%5Cprod%20_%7Bi%3D1%7D%5ENp_%7Bg_i%7D%28x_i%29
"L = \\prod _{i=1}^Np_{g_i}(x_i)")  

(只有其中一項的指數會是 ![1](https://latex.codecogs.com/png.latex?1 "1") ，其他都是
![0](https://latex.codecogs.com/png.latex?0 "0") ) 將之轉為
**log-likelihood**：

  
![l=\\sum \_{i=1}^N \\ln
p\_{g\_i}(x\_i)](https://latex.codecogs.com/png.latex?l%3D%5Csum%20_%7Bi%3D1%7D%5EN%20%5Cln%20p_%7Bg_i%7D%28x_i%29
"l=\\sum _{i=1}^N \\ln p_{g_i}(x_i)")  

若我們只先看 ![K=2](https://latex.codecogs.com/png.latex?K%3D2 "K=2")
的情形，則：(可參考課本4.4.1)   
![l(\\beta) = \\sum\_{i=1}^N\\{y\_i\\beta^Tx\_i
-\\ln(1+e^{\\beta^Tx\_i})
\\}](https://latex.codecogs.com/png.latex?l%28%5Cbeta%29%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%5C%7By_i%5Cbeta%5ETx_i%20-%5Cln%281%2Be%5E%7B%5Cbeta%5ETx_i%7D%29%20%20%5C%7D
"l(\\beta) = \\sum_{i=1}^N\\{y_i\\beta^Tx_i -\\ln(1+e^{\\beta^Tx_i})  \\}")  

我們求解 log-likelihood 的極值，另一階導數為0：   
![\\frac{\\partial l(\\beta)}{\\partial \\beta} = \\sum\_{i=1}^N
x\_i(y\_i-p(x\_i;\\beta))=0](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20l%28%5Cbeta%29%7D%7B%5Cpartial%20%5Cbeta%7D%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%20x_i%28y_i-p%28x_i%3B%5Cbeta%29%29%3D0
"\\frac{\\partial l(\\beta)}{\\partial \\beta} = \\sum_{i=1}^N x_i(y_i-p(x_i;\\beta))=0")  

其中 ![p(x\_i;\\beta) =
\\frac{e^{\\beta^Tx}}{1+e^{\\beta^Tx}}](https://latex.codecogs.com/png.latex?p%28x_i%3B%5Cbeta%29%20%3D%20%5Cfrac%7Be%5E%7B%5Cbeta%5ETx%7D%7D%7B1%2Be%5E%7B%5Cbeta%5ETx%7D%7D
"p(x_i;\\beta) = \\frac{e^{\\beta^Tx}}{1+e^{\\beta^Tx}}") 。

我們將一階條件，擴寫成矩陣形式：

  
![\\frac{\\partial l(\\beta)}{\\partial \\beta} =
\\mathbf{X}^T(y-p)](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%20l%28%5Cbeta%29%7D%7B%5Cpartial%20%5Cbeta%7D%20%3D%20%5Cmathbf%7BX%7D%5ET%28y-p%29
"\\frac{\\partial l(\\beta)}{\\partial \\beta} = \\mathbf{X}^T(y-p)")  

二階條件的 **Hessian matrix** 如下：

  
![\\frac{\\partial^2 l(\\beta)}{\\partial \\beta\\beta^T} =
-\\mathbf{X}^T\\mathbf{WX}](https://latex.codecogs.com/png.latex?%5Cfrac%7B%5Cpartial%5E2%20l%28%5Cbeta%29%7D%7B%5Cpartial%20%5Cbeta%5Cbeta%5ET%7D%20%3D%20-%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BWX%7D
"\\frac{\\partial^2 l(\\beta)}{\\partial \\beta\\beta^T} = -\\mathbf{X}^T\\mathbf{WX}")  

  -  是第 ![i](https://latex.codecogs.com/png.latex?i "i") 個元素是
    ![p(x\_i;\\beta^{old})(1-p(x\_i;\\beta^{old}))](https://latex.codecogs.com/png.latex?p%28x_i%3B%5Cbeta%5E%7Bold%7D%29%281-p%28x_i%3B%5Cbeta%5E%7Bold%7D%29%29
    "p(x_i;\\beta^{old})(1-p(x_i;\\beta^{old}))") 的對角矩陣。

由於此問題的一階條件沒有 closed-form，因此我們使用**牛頓法**找根：   
![\\begin{aligned} &#10;\\beta^{new} &= \\beta^{old} +
(\\mathbf{W}^T\\mathbf{WX})^{-1}\\mathbf{X}^T(\\mathbf{y}-\\mathbf{p})\\\\&#10;&=
(\\mathbf{W}^T\\mathbf{WX})^{-1}\\mathbf{X}^T\\mathbf{W}(\\mathbf{X}\\beta^{old}+\\mathbf{W}^{-1}(\\mathbf{y}-\\mathbf{p}))\\\\&#10;&=(\\mathbf{W}^T\\mathbf{WX})^{-1}\\mathbf{X}^T\\mathbf{W}\\mathbf{z}&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%20%0A%5Cbeta%5E%7Bnew%7D%20%26%3D%20%5Cbeta%5E%7Bold%7D%20%2B%20%28%5Cmathbf%7BW%7D%5ET%5Cmathbf%7BWX%7D%29%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%28%5Cmathbf%7By%7D-%5Cmathbf%7Bp%7D%29%5C%5C%0A%26%3D%20%28%5Cmathbf%7BW%7D%5ET%5Cmathbf%7BWX%7D%29%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BW%7D%28%5Cmathbf%7BX%7D%5Cbeta%5E%7Bold%7D%2B%5Cmathbf%7BW%7D%5E%7B-1%7D%28%5Cmathbf%7By%7D-%5Cmathbf%7Bp%7D%29%29%5C%5C%0A%26%3D%28%5Cmathbf%7BW%7D%5ET%5Cmathbf%7BWX%7D%29%5E%7B-1%7D%5Cmathbf%7BX%7D%5ET%5Cmathbf%7BW%7D%5Cmathbf%7Bz%7D%0A%5Cend%7Baligned%7D
"\\begin{aligned} 
\\beta^{new} &= \\beta^{old} + (\\mathbf{W}^T\\mathbf{WX})^{-1}\\mathbf{X}^T(\\mathbf{y}-\\mathbf{p})\\\\
&= (\\mathbf{W}^T\\mathbf{WX})^{-1}\\mathbf{X}^T\\mathbf{W}(\\mathbf{X}\\beta^{old}+\\mathbf{W}^{-1}(\\mathbf{y}-\\mathbf{p}))\\\\
&=(\\mathbf{W}^T\\mathbf{WX})^{-1}\\mathbf{X}^T\\mathbf{W}\\mathbf{z}
\\end{aligned}")  

觀察上式，我們令
![\\mathbf{z}=\\mathbf{X}\\beta^{old}+\\mathbf{W}^{-1}(\\mathbf{y}-\\mathbf{p})](https://latex.codecogs.com/png.latex?%5Cmathbf%7Bz%7D%3D%5Cmathbf%7BX%7D%5Cbeta%5E%7Bold%7D%2B%5Cmathbf%7BW%7D%5E%7B-1%7D%28%5Cmathbf%7By%7D-%5Cmathbf%7Bp%7D%29
"\\mathbf{z}=\\mathbf{X}\\beta^{old}+\\mathbf{W}^{-1}(\\mathbf{y}-\\mathbf{p})")，則我們每進行一次迭代，都是彷彿在進行一次
*weighted least squared*，故稱之為 **iteratively reweighted least
squared(IRLS)**。在迭代的過程中，我們可以選擇從
![\\beta=0](https://latex.codecogs.com/png.latex?%5Cbeta%3D0 "\\beta=0")
開始。

  
![\\beta^{new} \\leftarrow \\mathop{\\arg\\min}\_\\beta
(\\mathbf{z}-\\mathbf{X}\\beta)^T\\mathbf{W}(\\mathbf{z}-\\mathbf{X}\\beta)](https://latex.codecogs.com/png.latex?%5Cbeta%5E%7Bnew%7D%20%5Cleftarrow%20%5Cmathop%7B%5Carg%5Cmin%7D_%5Cbeta%20%28%5Cmathbf%7Bz%7D-%5Cmathbf%7BX%7D%5Cbeta%29%5ET%5Cmathbf%7BW%7D%28%5Cmathbf%7Bz%7D-%5Cmathbf%7BX%7D%5Cbeta%29
"\\beta^{new} \\leftarrow \\mathop{\\arg\\min}_\\beta (\\mathbf{z}-\\mathbf{X}\\beta)^T\\mathbf{W}(\\mathbf{z}-\\mathbf{X}\\beta)")  

### 一個 ![K=2](https://latex.codecogs.com/png.latex?K%3D2 "K=2") 的例子，從 ![x \\in \\mathbb{R}](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D "x \\in \\mathbb{R}") 到 ![x \\in \\mathbb{R}^p](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5Ep "x \\in \\mathbb{R}^p") (from Ex. 4.5)

![](https://esl.hohoweiya.xyz/img/04/fig4.16.png)

**Consider a two-class logistic regression problem with ![x \\in
\\mathbb{R}](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D
"x \\in \\mathbb{R}"). Characterize the maximum-likelihood estimates of
the slope and intercept parameter if the sample
![x\_i](https://latex.codecogs.com/png.latex?x_i "x_i") for the two
classes are separated by a point ![x \\in
\\mathbb{R}](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D
"x \\in \\mathbb{R}"). Generalize this result to (a) ![x \\in
\\mathbb{R}^p](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5Ep
"x \\in \\mathbb{R}^p") (see Figure 4.16), and (b) more than two
classes.**

在先前的討論中，我們知道當 ![K=2](https://latex.codecogs.com/png.latex?K%3D2 "K=2")
時，log-likelihood function 為

  
![l(\\beta) = \\sum\_{i=1}^N\\{y\_i\\beta^Tx\_i
-\\ln(1+e^{\\beta^Tx\_i})
\\}](https://latex.codecogs.com/png.latex?l%28%5Cbeta%29%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%5C%7By_i%5Cbeta%5ETx_i%20-%5Cln%281%2Be%5E%7B%5Cbeta%5ETx_i%7D%29%20%20%5C%7D
"l(\\beta) = \\sum_{i=1}^N\\{y_i\\beta^Tx_i -\\ln(1+e^{\\beta^Tx_i})  \\}")  

而若 ![x \\in
\\mathbb{R}](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D
"x \\in \\mathbb{R}")(一維)，則

  
![y\_i =\\left\\{\\begin{matrix}&#10;0,\\ x\_i \\leq x\_0\\\\ &#10;1,\\
x\_i \>
x\_0&#10;\\end{matrix}\\right.](https://latex.codecogs.com/png.latex?y_i%20%3D%5Cleft%5C%7B%5Cbegin%7Bmatrix%7D%0A0%2C%5C%20x_i%20%5Cleq%20x_0%5C%5C%20%0A1%2C%5C%20x_i%20%3E%20x_0%0A%5Cend%7Bmatrix%7D%5Cright.
"y_i =\\left\\{\\begin{matrix}
0,\\ x_i \\leq x_0\\\\ 
1,\\ x_i \> x_0
\\end{matrix}\\right.")  

我們考慮最基本的一維羅吉斯回歸模型，的對數概似函數：   
![l(\\beta) = \\sum\_{i=1}^N\\{y\_i\\beta^Tx\_i
-\\ln(1+e^{\\beta^Tx\_i})
\\}](https://latex.codecogs.com/png.latex?l%28%5Cbeta%29%20%3D%20%5Csum_%7Bi%3D1%7D%5EN%5C%7By_i%5Cbeta%5ETx_i%20-%5Cln%281%2Be%5E%7B%5Cbeta%5ETx_i%7D%29%20%20%5C%7D
"l(\\beta) = \\sum_{i=1}^N\\{y_i\\beta^Tx_i -\\ln(1+e^{\\beta^Tx_i})  \\}")  
故   
![\\begin{aligned}&#10;l(\\beta) &=
\\sum\_{i=1}^N\\{y\_i(\\beta\_0+\\beta\_1x\_i)
-\\ln(1+e^{(\\beta\_0+\\beta\_1x\_i)}) \\}
\\\\&#10;&=\\sum\_{y\_i=0}\[-\\ln(1+e^{(\\beta\_0+\\beta\_1x\_i)})\]+\\sum\_{y\_i=1}\[(\\beta\_0+\\beta\_1x\_i)-\\ln(1+e^{(\\beta\_0+\\beta\_1x\_i)})\]&#10;
\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Al%28%5Cbeta%29%20%26%3D%20%5Csum_%7Bi%3D1%7D%5EN%5C%7By_i%28%5Cbeta_0%2B%5Cbeta_1x_i%29%20-%5Cln%281%2Be%5E%7B%28%5Cbeta_0%2B%5Cbeta_1x_i%29%7D%29%20%20%5C%7D%20%5C%5C%0A%26%3D%5Csum_%7By_i%3D0%7D%5B-%5Cln%281%2Be%5E%7B%28%5Cbeta_0%2B%5Cbeta_1x_i%29%7D%29%5D%2B%5Csum_%7By_i%3D1%7D%5B%28%5Cbeta_0%2B%5Cbeta_1x_i%29-%5Cln%281%2Be%5E%7B%28%5Cbeta_0%2B%5Cbeta_1x_i%29%7D%29%5D%0A%20%5Cend%7Baligned%7D
"\\begin{aligned}
l(\\beta) &= \\sum_{i=1}^N\\{y_i(\\beta_0+\\beta_1x_i) -\\ln(1+e^{(\\beta_0+\\beta_1x_i)})  \\} \\\\
&=\\sum_{y_i=0}[-\\ln(1+e^{(\\beta_0+\\beta_1x_i)})]+\\sum_{y_i=1}[(\\beta_0+\\beta_1x_i)-\\ln(1+e^{(\\beta_0+\\beta_1x_i)})]
 \\end{aligned}")  

若我們想要 **maximize log-likelihood**，很明顯地，當 ![\\beta \\rightarrow
\\infty](https://latex.codecogs.com/png.latex?%5Cbeta%20%5Crightarrow%20%5Cinfty
"\\beta \\rightarrow \\infty") 時就會發散，這導致我們無法找到一個 closed-form 的解。

但很直觀地，當![x \\in
\\mathbb{R}](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D
"x \\in \\mathbb{R}")，由於資料只有一維，那我們在 x 軸上找到
![x=x\_0](https://latex.codecogs.com/png.latex?x%3Dx_0 "x=x_0")
，並在此處畫上一條垂直於 x 軸的線，這樣就能夠明確地劃分出兩個分類。

  - 當 ![x \\in
    \\mathbb{R}^p](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5Ep
    "x \\in \\mathbb{R}^p") 時，該如何處理呢？(p\>1)

我們透過一樣的邏輯擴展
![l(\\beta)](https://latex.codecogs.com/png.latex?l%28%5Cbeta%29
"l(\\beta)") 用矩陣向量表示：

  
![\\begin{aligned}&#10;l(\\beta) &= \\sum\_{i=1}^N\\{y\_i\\beta^Tx\_i
-\\ln(1+e^{\\beta^Tx\_i}) \\}
\\\\&#10;&=\\sum\_{y\_i=0}\[-\\ln(1+e^{\\beta^Tx\_i})\]+\\sum\_{y\_i=1}\[\\beta^Tx\_i-\\ln(1+e^{\\beta^Tx\_i})\]\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Al%28%5Cbeta%29%20%26%3D%20%5Csum_%7Bi%3D1%7D%5EN%5C%7By_i%5Cbeta%5ETx_i%20-%5Cln%281%2Be%5E%7B%5Cbeta%5ETx_i%7D%29%20%20%5C%7D%20%5C%5C%0A%26%3D%5Csum_%7By_i%3D0%7D%5B-%5Cln%281%2Be%5E%7B%5Cbeta%5ETx_i%7D%29%5D%2B%5Csum_%7By_i%3D1%7D%5B%5Cbeta%5ETx_i-%5Cln%281%2Be%5E%7B%5Cbeta%5ETx_i%7D%29%5D%5Cend%7Baligned%7D
"\\begin{aligned}
l(\\beta) &= \\sum_{i=1}^N\\{y_i\\beta^Tx_i -\\ln(1+e^{\\beta^Tx_i})  \\} \\\\
&=\\sum_{y_i=0}[-\\ln(1+e^{\\beta^Tx_i})]+\\sum_{y_i=1}[\\beta^Tx_i-\\ln(1+e^{\\beta^Tx_i})]\\end{aligned}")  

同樣地，我們同樣會發現
![l(\\beta)](https://latex.codecogs.com/png.latex?l%28%5Cbeta%29
"l(\\beta)") 會發散。

  - 當 ![K\>2](https://latex.codecogs.com/png.latex?K%3E2 "K\>2")
    時，要如何處理呢？

直觀上來說，當 ![K\>2](https://latex.codecogs.com/png.latex?K%3E2 "K\>2")
則需要超過一個超平面來協助我們分類，我們同樣能寫出這種情況的 log-likelihood function：

  
![\\begin{aligned}&#10;l(\\beta) &= \\sum\_{i=1}^N
\[\\sum\_{k=1}^{K-1}\\mathbf{1}\_{y\_i=k}
\\beta\_k^Tx\_i-\\ln(1+\\sum\_{l=1}^{K-1}e^{\\beta\_l^Tx\_i}\]\\\\&#10;&=\\sum\_{k=1}^{K-1}
\\sum\_{g\_k}\[\\beta\_k^Tx\_i-\\ln(1+\\sum\_{l=1}^{K-1}\\beta\_l^Tx\_i)
\]&#10;+\\sum\_{g\_k}\[-\\ln(1+\\sum\_{l=1}^{K-1}e^{\\beta\_l^Tx\_i})\]&#10;\\end{aligned}](https://latex.codecogs.com/png.latex?%5Cbegin%7Baligned%7D%0Al%28%5Cbeta%29%20%26%3D%20%5Csum_%7Bi%3D1%7D%5EN%20%5B%5Csum_%7Bk%3D1%7D%5E%7BK-1%7D%5Cmathbf%7B1%7D_%7By_i%3Dk%7D%20%5Cbeta_k%5ETx_i-%5Cln%281%2B%5Csum_%7Bl%3D1%7D%5E%7BK-1%7De%5E%7B%5Cbeta_l%5ETx_i%7D%5D%5C%5C%0A%26%3D%5Csum_%7Bk%3D1%7D%5E%7BK-1%7D%20%5Csum_%7Bg_k%7D%5B%5Cbeta_k%5ETx_i-%5Cln%281%2B%5Csum_%7Bl%3D1%7D%5E%7BK-1%7D%5Cbeta_l%5ETx_i%29%20%5D%0A%2B%5Csum_%7Bg_k%7D%5B-%5Cln%281%2B%5Csum_%7Bl%3D1%7D%5E%7BK-1%7De%5E%7B%5Cbeta_l%5ETx_i%7D%29%5D%0A%5Cend%7Baligned%7D
"\\begin{aligned}
l(\\beta) &= \\sum_{i=1}^N [\\sum_{k=1}^{K-1}\\mathbf{1}_{y_i=k} \\beta_k^Tx_i-\\ln(1+\\sum_{l=1}^{K-1}e^{\\beta_l^Tx_i}]\\\\
&=\\sum_{k=1}^{K-1} \\sum_{g_k}[\\beta_k^Tx_i-\\ln(1+\\sum_{l=1}^{K-1}\\beta_l^Tx_i) ]
+\\sum_{g_k}[-\\ln(1+\\sum_{l=1}^{K-1}e^{\\beta_l^Tx_i})]
\\end{aligned}")  

而這條式子同樣也找不到 的 closed-form。

### Example: South African Heart Disease

### Logistic v.s. LDA

## Perceptron Learning Algorithm (PLA)
