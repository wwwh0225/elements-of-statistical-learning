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

## Linear discriminant Analysis(LDA)

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

要注意，這樣良好的線性性質是來自於我們假設兩個分類具有**相同的**共變異數矩陣。若我們對類別![k](https://latex.codecogs.com/png.latex?k
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

<p class="comment">

Suppose we have features ![x \\in
\\mathbb{R}^p](https://latex.codecogs.com/png.latex?x%20%5Cin%20%5Cmathbb%7BR%7D%5Ep
"x \\in \\mathbb{R}^p"), a two-class response, with class sizes
![N\_1](https://latex.codecogs.com/png.latex?N_1 "N_1"),
![N\_2](https://latex.codecogs.com/png.latex?N_2 "N_2"), and the target
coded as
![−N/N\_1](https://latex.codecogs.com/png.latex?%E2%88%92N%2FN_1
"−N/N_1"), ![N/N\_2](https://latex.codecogs.com/png.latex?N%2FN_2
"N/N_2").

</p>

``` r
summary(cars)
```

    ##      speed           dist       
    ##  Min.   : 4.0   Min.   :  2.00  
    ##  1st Qu.:12.0   1st Qu.: 26.00  
    ##  Median :15.0   Median : 36.00  
    ##  Mean   :15.4   Mean   : 42.98  
    ##  3rd Qu.:19.0   3rd Qu.: 56.00  
    ##  Max.   :25.0   Max.   :120.00

## Including Plots

You can also embed plots, for example:

![](ch4_files/figure-gfm/pressure-1.png)<!-- -->

Note that the `echo = FALSE` parameter was added to the code chunk to
prevent printing of the R code that generated the plot.
