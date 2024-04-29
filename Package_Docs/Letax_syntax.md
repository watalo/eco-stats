
一般公式分为两种形式，行内公式和行间公式。

- 行内公式：$\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.$
- 行间公式：$$\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.$$

  对应的代码块为：python
$$ \begin{align*}  \overbrace{= some\ expression}^{description} \end{align*} $$

```python
$ \Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,. $
$$\Gamma(z) = \int_0^\infty t^{z-1}e^{-t}dt\,.$$
```

  行内公式是在公式代码块的基础上前面加上`$` ，后面加上`$`组成的，而行间公式则是在公式代码块前后使用``$$``
  下面主要介绍数学公式中常用的一些符号。

## **希腊字母**

| 名称      | 大写  | code    | 小写       | code     |
| ------- | --- | ------- | -------- | -------- |
| alpha   | A   | A       | α        | \alpha   |
| beta    | B   | B       | β        | \beta    |
| gamma   | Γ   | \Gamma  | γ        | \gamma   |
| delta   | Δ   | \Delta  | δ        | \delta   |
| epsilon | E   | E       | ϵ        | \epsilon |
| zeta    | Z   | Z       | ζ        | \zeta    |
| eta     | H   | H       | η        | \eta     |
| theta   | Θ   | \Theta  | θ        | \theta   |
| iota    | I   | I       | ι        | \iota    |
| kappa   | K   | K       | κ        | \kappa   |
| lambda  | Λ   | \Lambda | λ        | \lambda  |
| mu      | M   | M       | μ        | \mu      |
| nu      | N   | N       | ν        | \nu      |
| xi      | Ξ   | \Xi     | ξ        | \xi      |
| omicron | O   | O       | ο        | \omicron |
| pi      | Π   | \Pi     | π        | \pi      |
| rho     | P   | P       | ρ        | \rho     |
| sigma   | Σ   | \Sigma  | σ        |          |
| \sigma  | tau | T       | T        | τ        |
| upsilon | Υ   | υ       | \upsilon |          |
| phi     | Φ   | \Phi    | ϕ        | \phi     |
| chi     | X   | X       | χ        | \chi     |
| psi     | Ψ   | \Psi    | ψ        | \psi     |
| omega   | Ω   | \Omega  | ω        | \omega   |

## **上标与下标**

  上标和下标分别使用`^` 与`_` ，例如`$x_i^2$`表示的是：![x_i^2](https://math.jianshu.com/math?formula=x_i%5E2)。  
  默认情况下，上、下标符号仅仅对下一个组起作用。一个组即单个字符或者使用`{..}` 包裹起来的内容。如果使用`$10^10$` 表示的是![10^10](https://math.jianshu.com/math?formula=10%5E10)，而`$10^{10}$` 才是![10^{10}](https://math.jianshu.com/math?formula=10%5E%7B10%7D)。同时，大括号还能消除二义性，如`x^5^6` 将得到一个错误，必须使用大括号来界定^的结合性，如`${x^5}^6$` ：![{x^5}^6](https://math.jianshu.com/math?formula=%7Bx%5E5%7D%5E6)或者`$x^{5^6}$` ：![x^{5^6}](https://math.jianshu.com/math?formula=x%5E%7B5%5E6%7D)。

## **括号**

### 小括号与方括号

  使用原始的`( )` ，`[ ]` 即可，如`$(2+3)[4+4]$` ：![(2+3)](https://math.jianshu.com/math?formula=(2%2B3)) ![[4+4]](https://math.jianshu.com/math?formula=%5B4%2B4%5D)  
  使用\left(或\right)使符号大小与邻近的公式相适应（该语句适用于所有括号类型），
  如`$\left(\frac{x}{y}\right)$` ：![\left(\frac{x}{y}\right)](https://math.jianshu.com/math?formula=%5Cleft(%5Cfrac%7Bx%7D%7By%7D%5Cright))

### 大括号

  由于大括号`{}` 被用于分组，因此需要使用`\{`和`\}`表示大括号，也可以使用`\lbrace` 和`\rbrace`来表示。如`$\{a\*b\}:a\∗b$` 或`$\lbrace a\*b\rbrace :a\*b$` 表示![\{a*b\}:a∗b](https://math.jianshu.com/math?formula=%5C%7Ba*b%5C%7D%3Aa%E2%88%97b)。

### 尖括号

  区分于小于号和大于号，使用`\langle` 和`\rangle` 表示左尖括号和右尖括号。如`$\langle x \rangle$` 表示：![\langle x \rangle](https://math.jianshu.com/math?formula=%5Clangle%20x%20%5Crangle)。

### 上取整

  使用`\lceil` 和 `\rceil` 表示。 如，`$\lceil x \rceil$`：![\lceil x \rceil](https://math.jianshu.com/math?formula=%5Clceil%20x%20%5Crceil)。

### 下取整

  使用`\lfloor` 和 `\rfloor` 表示。如，`$\lfloor x \rfloor$`：![\lfloor x \rfloor](https://math.jianshu.com/math?formula=%5Clfloor%20x%20%5Crfloor)。

## **求和与积分**

### 求和

  `\sum` 用来表示求和符号，其下标表示求和下限，上标表示上限。如:  
  `$\sum_{r=1}^n$`表示：![\sum_{r=1}^n](https://math.jianshu.com/math?formula=%5Csum_%7Br%3D1%7D%5En)。  
  `$$\sum_{r=1}^n$$`表示：![\sum_{r=1}^n](https://math.jianshu.com/math?formula=%5Csum_%7Br%3D1%7D%5En)

### 积分

  `\int` 用来表示积分符号，同样地，其上下标表示积分的上下限。如，`$\int_{r=1}^\infty$`：![\int_{r=1}^\infty](https://math.jianshu.com/math?formula=%5Cint_%7Br%3D1%7D%5E%5Cinfty)。  
  多重积分同样使用 **int** ，通过 **i** 的数量表示积分导数：  
  `$\iint$` ：![\iint](https://math.jianshu.com/math?formula=%5Ciint)  
  `$\iiint$` ：![\iiint](https://math.jianshu.com/math?formula=%5Ciiint)  
  `$\iiiint$` ：![\iiiint](https://math.jianshu.com/math?formula=%5Ciiiint)

### 连乘

  `$\prod {a+b}$`，输出：![\prod {a+b}](https://math.jianshu.com/math?formula=%5Cprod%20%7Ba%2Bb%7D)。  
  `$\prod_{i=1}^{K}$`，输出：![\prod_{i=1}^{K}](https://math.jianshu.com/math?formula=%5Cprod_%7Bi%3D1%7D%5E%7BK%7D)。  
  `$$\prod_{i=1}^{K}$$`，输出：![\prod_{i=1}^{K}](https://math.jianshu.com/math?formula=%5Cprod_%7Bi%3D1%7D%5E%7BK%7D)。

### 其他

  与此类似的符号还有，  
  `$\prod$` ：![\prod](https://math.jianshu.com/math?formula=%5Cprod)  
  `$\bigcup$` ：![\bigcup](https://math.jianshu.com/math?formula=%5Cbigcup)  
  `$\bigcap$` ：![\bigcap](https://math.jianshu.com/math?formula=%5Cbigcap)  
  `$arg\,\max_{c_k}$`：![arg\,\max_{c_k}](https://math.jianshu.com/math?formula=arg%5C%2C%5Cmax_%7Bc_k%7D)  
  `$arg\,\min_{c_k}$`：![arg\,\min_{c_k}](https://math.jianshu.com/math?formula=arg%5C%2C%5Cmin_%7Bc_k%7D)  
  `$\mathop {argmin}_{c_k}$`：![\mathop {argmin}_{c_k}](https://math.jianshu.com/math?formula=%5Cmathop%20%7Bargmin%7D_%7Bc_k%7D)  
  `$\mathop {argmax}_{c_k}$`：![\mathop {argmax}_{c_k}](https://math.jianshu.com/math?formula=%5Cmathop%20%7Bargmax%7D_%7Bc_k%7D)  
  `$\max_{c_k}$`：![\max_{c_k}](https://math.jianshu.com/math?formula=%5Cmax_%7Bc_k%7D)  
  `$\min_{c_k}$`：![\min_{c_k}](https://math.jianshu.com/math?formula=%5Cmin_%7Bc_k%7D)

## **分式与根式**

### 分式

- 第一种，使用`\frac ab`，`\frac`作用于其后的两个组`a` ，`b` ，结果为![\frac ab](https://math.jianshu.com/math?formula=%5Cfrac%20ab)。如果你的分子或分母不是单个字符，请使用`{..}`来分组，比如`$\frac {a+c+1}{b+c+2}$`表示![\frac {a+c+1}{b+c+2}](https://math.jianshu.com/math?formula=%5Cfrac%20%7Ba%2Bc%2B1%7D%7Bb%2Bc%2B2%7D)。
- 第二种，使用`\over`来分隔一个组的前后两部分，如`{a+1\over b+1}`：![{a+1\over b+1}](https://math.jianshu.com/math?formula=%7Ba%2B1%5Cover%20b%2B1%7D)

### 连分数

  书写连分数表达式时，请使用`\cfrac`代替`\frac`或者`\over`两者效果对比如下：  
  `\frac` 表示如下：

```ruby
$$x=a_0 + \frac {1^2}{a_1 + \frac {2^2}{a_2 + \frac {3^2}{a_3 + \frac {4^2}{a_4 + ...}}}}$$
```

  显示如下：  
![x=a_0 + \frac {1^2}{a_1 + \frac {2^2}{a_2 + \frac {3^2}{a_3 + \frac {4^2}{a_4 + ...}}}}](https://math.jianshu.com/math?formula=x%3Da_0%20%2B%20%5Cfrac%20%7B1%5E2%7D%7Ba_1%20%2B%20%5Cfrac%20%7B2%5E2%7D%7Ba_2%20%2B%20%5Cfrac%20%7B3%5E2%7D%7Ba_3%20%2B%20%5Cfrac%20%7B4%5E2%7D%7Ba_4%20%2B%20...%7D%7D%7D%7D)  
  `\cfrac` 表示如下：

```ruby
$$x=a_0 + \cfrac {1^2}{a_1 + \cfrac {2^2}{a_2 + \cfrac {3^2}{a_3 + \cfrac {4^2}{a_4 + ...}}}}$$
```

  显示如下：  
![x=a_0 + \cfrac {1^2}{a_1 + \cfrac {2^2}{a_2 + \cfrac {3^2}{a_3 + \cfrac {4^2}{a_4 + ...}}}}](https://math.jianshu.com/math?formula=x%3Da_0%20%2B%20%5Ccfrac%20%7B1%5E2%7D%7Ba_1%20%2B%20%5Ccfrac%20%7B2%5E2%7D%7Ba_2%20%2B%20%5Ccfrac%20%7B3%5E2%7D%7Ba_3%20%2B%20%5Ccfrac%20%7B4%5E2%7D%7Ba_4%20%2B%20...%7D%7D%7D%7D)

### 根式

  根式使用`\sqrt` 来表示。  
  如开4次方：`$\sqrt[4]{\frac xy}$` ：![\sqrt[4]{\frac xy}](https://math.jianshu.com/math?formula=%5Csqrt%5B4%5D%7B%5Cfrac%20xy%7D)。  
  开平方：`$\sqrt {a+b}$`：![\sqrt {a+b}](https://math.jianshu.com/math?formula=%5Csqrt%20%7Ba%2Bb%7D)。

## **多行表达式**

### 分类表达式

  定义函数的时候经常需要分情况给出表达式，使用`\begin{cases}…\end{cases}` 。其中：

-   使用`\\` 来分类，
-   使用`&` 指示需要对齐的位置，
-   使用`\` +`空格`表示空格。

```ruby
$$
f(n)
\begin{cases}
\cfrac n2, &if\ n\ is\ even\\
3n + 1, &if\  n\ is\ odd
\end{cases}
$$
```

  表示:  
![f(n) \begin{cases} \cfrac n2, &if\ n\ is\ even\\ 3n + 1, &if\ n\ is\ odd \end{cases}](https://math.jianshu.com/math?formula=f(n)%20%5Cbegin%7Bcases%7D%20%5Ccfrac%20n2%2C%20%26if%5C%20n%5C%20is%5C%20even%5C%5C%203n%20%2B%201%2C%20%26if%5C%20n%5C%20is%5C%20odd%20%5Cend%7Bcases%7D)

```ruby
$$
L(Y,f(X)) =
\begin{cases}
0, & \text{Y = f(X)}  \\
1, & \text{Y $\neq$ f(X)}
\end{cases}
$$
```

  表示:  
![L(Y,f(X)) = \begin{cases} 0, & \text{Y = f(X)} \\ 1, & \text{Y $\neq$ f(X)} \end{cases}](https://math.jianshu.com/math?formula=L(Y%2Cf(X))%20%3D%20%5Cbegin%7Bcases%7D%200%2C%20%26%20%5Ctext%7BY%20%3D%20f(X)%7D%20%5C%5C%201%2C%20%26%20%5Ctext%7BY%20%24%5Cneq%24%20f(X)%7D%20%5Cend%7Bcases%7D)  
  如果想分类之间的垂直间隔变大，可以使用`\\[2ex]` 代替`\\` 来分隔不同的情况。(`3ex,4ex` 也可以用，`1ex` 相当于原始距离）。如下所示：

```ruby
$$
L(Y,f(X)) =
\begin{cases}
0, & \text{Y = f(X)} \\[5ex]
1, & \text{Y $\neq$ f(X)}
\end{cases}
$$
```

  表示：  
![L(Y,f(X)) = \begin{cases} 0, & \text{Y = f(X)} \\[5ex] 1, & \text{Y $\neq$ f(X)} \end{cases}](https://math.jianshu.com/math?formula=L(Y%2Cf(X))%20%3D%20%5Cbegin%7Bcases%7D%200%2C%20%26%20%5Ctext%7BY%20%3D%20f(X)%7D%20%5C%5C%5B5ex%5D%201%2C%20%26%20%5Ctext%7BY%20%24%5Cneq%24%20f(X)%7D%20%5Cend%7Bcases%7D)

### 多行表达式

  有时候需要将一行公式分多行进行显示。

```ruby
$$
\begin{equation}\begin{split} 
a&=b+c-d \\ 
&\quad +e-f\\ 
&=g+h\\ 
& =i 
\end{split}\end{equation}
$$
```

  表示：  
![\begin{equation}\begin{split} a&=b+c-d \\ &\quad +e-f\\ &=g+h\\ & =i \end{split}\end{equation}](https://math.jianshu.com/math?formula=%5Cbegin%7Bequation%7D%5Cbegin%7Bsplit%7D%20a%26%3Db%2Bc-d%20%5C%5C%20%26%5Cquad%20%2Be-f%5C%5C%20%26%3Dg%2Bh%5C%5C%20%26%20%3Di%20%5Cend%7Bsplit%7D%5Cend%7Bequation%7D)  
  其中`begin{equation}` 表示开始方程，`end{equation}` 表示方程结束；`begin{split}` 表示开始多行公式，`end{split}` 表示结束；公式中用`\\` 表示回车到下一行，`&` 表示对齐的位置。

### 方程组

  使用`\begin{array}...\end{array}` 与`\left \{` 与`\right.` 配合表示方程组:

```ruby
$$
\left \{ 
\begin{array}{c}
a_1x+b_1y+c_1z=d_1 \\ 
a_2x+b_2y+c_2z=d_2 \\ 
a_3x+b_3y+c_3z=d_3
\end{array}
\right.
$$
```

  表示：  
![\left \{ \begin{array}{c} a_1x+b_1y+c_1z=d_1 \\ a_2x+b_2y+c_2z=d_2 \\ a_3x+b_3y+c_3z=d_3 \end{array} \right.](https://math.jianshu.com/math?formula=%5Cleft%20%5C%7B%20%5Cbegin%7Barray%7D%7Bc%7D%20a_1x%2Bb_1y%2Bc_1z%3Dd_1%20%5C%5C%20a_2x%2Bb_2y%2Bc_2z%3Dd_2%20%5C%5C%20a_3x%2Bb_3y%2Bc_3z%3Dd_3%20%5Cend%7Barray%7D%20%5Cright.)  
  注意：通常MathJax通过内部策略自己管理公式内部的空间，因此`a…b` 与`a…….b` （`.`表示空格）都会显示为`ab` 。可以通过在`ab` 间加入`\` ,增加些许间隙，`\;` 增加较宽的间隙，`\quad` 与`\qquad` 会增加更大的间隙。

## **特殊函数与符号**

### 三角函数

  `$\sinx$` : $sinx$
  `\arctanx` : $arctanz$

### 比较运算符

  小于(`\lt` )：![\lt](https://math.jianshu.com/math?formula=%5Clt)  
  大于(`\gt` )：![\gt](https://math.jianshu.com/math?formula=%5Cgt)  
  小于等于(`\le` )：![\le](https://math.jianshu.com/math?formula=%5Cle)  
  大于等于(`\ge` )：![\ge](https://math.jianshu.com/math?formula=%5Cge)  
  不等于(`\ne` ) : ![\ne](https://math.jianshu.com/math?formula=%5Cne)  
  可以在这些运算符前面加上`\not` ，如`\not\lt` : ![\not\lt`](https://math.jianshu.com/math?formula=%5Cnot%5Clt%60)

### 集合关系与运算

  并集(`\cup` ): ![\cup](https://math.jianshu.com/math?formula=%5Ccup)  
  交集(`\cap` ): ![\cap](https://math.jianshu.com/math?formula=%5Ccap)  
  差集(`\setminus` ): ![\setminus](https://math.jianshu.com/math?formula=%5Csetminus)  
  子集(`\subset` ): ![\subset](https://math.jianshu.com/math?formula=%5Csubset)  
  子集(`\subseteq` ): ![\subseteq](https://math.jianshu.com/math?formula=%5Csubseteq)  
  非子集(`\subsetneq` ): ![\subsetneq](https://math.jianshu.com/math?formula=%5Csubsetneq)  
  父集(`\supset` ): ![\supset](https://math.jianshu.com/math?formula=%5Csupset)  
  属于(`\in` ): ![\in](https://math.jianshu.com/math?formula=%5Cin)  
  不属于(`\notin` ): ![\notin](https://math.jianshu.com/math?formula=%5Cnotin)  
  空集(`\emptyset` ): ![\emptyset](https://math.jianshu.com/math?formula=%5Cemptyset)  
  空(`\varnothing` ): ![\varnothing](https://math.jianshu.com/math?formula=%5Cvarnothing)

### 排列

  `\binom{n+1}{2k}` : ![\binom{n+1}{2k}](https://math.jianshu.com/math?formula=%5Cbinom%7Bn%2B1%7D%7B2k%7D)  
  `{n+1 \choose 2k}` : ![{n+1 \choose 2k}](https://math.jianshu.com/math?formula=%7Bn%2B1%20%5Cchoose%202k%7D)

### 箭头

  (`\to` ):![\to](https://math.jianshu.com/math?formula=%5Cto)  
  (`\rightarrow` ): ![\rightarrow](https://math.jianshu.com/math?formula=%5Crightarrow)  
  (`\leftarrow` ): ![\leftarrow](https://math.jianshu.com/math?formula=%5Cleftarrow)  
  (`\Rightarrow` ): ![\Rightarrow](https://math.jianshu.com/math?formula=%5CRightarrow)  
  (`\Leftarrow` ): ![\Leftarrow](https://math.jianshu.com/math?formula=%5CLeftarrow)  
  (`\mapsto` ): ![\mapsto](https://math.jianshu.com/math?formula=%5Cmapsto)

### 逻辑运算符

  (`\land` ): ![\land](https://math.jianshu.com/math?formula=%5Cland)  
  (`\lor` ): ![\lor](https://math.jianshu.com/math?formula=%5Clor)  
  (`\lnot` ): ![\lnot](https://math.jianshu.com/math?formula=%5Clnot)  
  (`\forall` ): ![\forall](https://math.jianshu.com/math?formula=%5Cforall)  
  (`\exists` ): ![\exists](https://math.jianshu.com/math?formula=%5Cexists)  
  (`\top` ): ![\top](https://math.jianshu.com/math?formula=%5Ctop)  
  (`\bot` ): ![\bot](https://math.jianshu.com/math?formula=%5Cbot)  
  (`\vdash` ): ![\vdash](https://math.jianshu.com/math?formula=%5Cvdash)  
  (`\vDash` ): ![\vDash](https://math.jianshu.com/math?formula=%5CvDash)

### 操作符

  (`\star` ): ![\star](https://math.jianshu.com/math?formula=%5Cstar)  
  (`\ast` ): ![\ast](https://math.jianshu.com/math?formula=%5Cast)  
  (`\oplus` ): ![\oplus](https://math.jianshu.com/math?formula=%5Coplus)  
  (`\circ` ): ![\circ](https://math.jianshu.com/math?formula=%5Ccirc)  
  (`\bullet` ): ![\bullet](https://math.jianshu.com/math?formula=%5Cbullet)

### 等于

  (`\approx` ): ![\approx](https://math.jianshu.com/math?formula=%5Capprox)  
  (`\sim` ): ![\sim](https://math.jianshu.com/math?formula=%5Csim)  
  (`\equiv` ): ![\equiv](https://math.jianshu.com/math?formula=%5Cequiv)  
  (`\prec` ): ![\prec](https://math.jianshu.com/math?formula=%5Cprec)

### 范围

  (`\infty` ): ![\infty](https://math.jianshu.com/math?formula=%5Cinfty)  
  (`\aleph_o` ): ![\aleph_o](https://math.jianshu.com/math?formula=%5Caleph_o)  
  (`\nabla` ): ![\nabla](https://math.jianshu.com/math?formula=%5Cnabla)  
  (`\Im` ): ![\Im](https://math.jianshu.com/math?formula=%5CIm)  
  (`\Re` ): ![\Re](https://math.jianshu.com/math?formula=%5CRe)

### 模运算

  (`\pmod` ): ![b \pmod n](https://math.jianshu.com/math?formula=b%20%5Cpmod%20n)  
  如`a \equiv b \pmod n` : ![a \equiv b \pmod n](https://math.jianshu.com/math?formula=a%20%5Cequiv%20b%20%5Cpmod%20n)

### 点

  (`\ldots` ): ![\ldots](https://math.jianshu.com/math?formula=%5Cldots)  
  (`\cdots` ): ![\cdots](https://math.jianshu.com/math?formula=%5Ccdots)  
  (`\cdot` ): ![\cdot](https://math.jianshu.com/math?formula=%5Ccdot)  
  其区别是点的位置不同，`\ldots` 位置稍低，`\cdots` 位置居中。

```ruby
$$
\begin{equation}
a_1+a_2+\ldots+a_n \\ 
a_1+a_2+\cdots+a_n
\end{equation}
$$
```

  表示：  
![\begin{equation} a_1+a_2+\ldots+a_n \\ a_1+a_2+\cdots+a_n \end{equation}](https://math.jianshu.com/math?formula=%5Cbegin%7Bequation%7D%20a_1%2Ba_2%2B%5Cldots%2Ba_n%20%5C%5C%20a_1%2Ba_2%2B%5Ccdots%2Ba_n%20%5Cend%7Bequation%7D)

## **顶部符号**

  对于单字符，`\hat x` ：![\hat x](https://math.jianshu.com/math?formula=%5Chat%20x)  
  多字符可以使用`\widehat {xy}` ：![\widehat {xy}](https://math.jianshu.com/math?formula=%5Cwidehat%20%7Bxy%7D)  
  类似的还有:  
  (`\overline x` ): ![\overline x](https://math.jianshu.com/math?formula=%5Coverline%20x)  
  矢量(`\vec` ): ![\vec x](https://math.jianshu.com/math?formula=%5Cvec%20x)  
  向量(`\overrightarrow {xy}` ): ![\overrightarrow {xy}](https://math.jianshu.com/math?formula=%5Coverrightarrow%20%7Bxy%7D)  
  (`\dot x` ): ![\dot x](https://math.jianshu.com/math?formula=%5Cdot%20x)  
  (`\ddot x` ): ![\ddot x](https://math.jianshu.com/math?formula=%5Cddot%20x)  
  (`\dot {\dot x}` ): ![\dot {\dot x}](https://math.jianshu.com/math?formula=%5Cdot%20%7B%5Cdot%20x%7D)
波浪上标（`\tilde x`） $\tilde x$
## **表格**

  使用`\begin{array}{列样式}…\end{array}` 这样的形式来创建表格，列样式可以是`clr` 表示居中，左，右对齐，还可以使用`|` 表示一条竖线。表格中各行使用`\\` 分隔，各列使用`&` 分隔。使用`\hline` 在本行前加入一条直线。 例如:

```ruby
$$
\begin{array}{c|lcr}
n & \text{Left} & \text{Center} & \text{Right} \\
\hline
1 & 0.24 & 1 & 125 \\
2 & -1 & 189 & -8 \\
3 & -20 & 2000 & 1+10i \\
\end{array}
$$
```

  得到：  
![\begin{array}{c|lcr} n & \text{Left} & \text{Center} & \text{Right} \\ \hline 1 & 0.24 & 1 & 125 \\ 2 & -1 & 189 & -8 \\ 3 & -20 & 2000 & 1+10i \\ \end{array}](https://math.jianshu.com/math?formula=%5Cbegin%7Barray%7D%7Bc%7Clcr%7D%20n%20%26%20%5Ctext%7BLeft%7D%20%26%20%5Ctext%7BCenter%7D%20%26%20%5Ctext%7BRight%7D%20%5C%5C%20%5Chline%201%20%26%200.24%20%26%201%20%26%20125%20%5C%5C%202%20%26%20-1%20%26%20189%20%26%20-8%20%5C%5C%203%20%26%20-20%20%26%202000%20%26%201%2B10i%20%5C%5C%20%5Cend%7Barray%7D)

## **矩阵**

### 基本内容

  使用`\begin{matrix}…\end{matrix}` 这样的形式来表示矩阵，在`\begin` 与`\end` 之间加入矩阵中的元素即可。矩阵的行之间使用`\\` 分隔，列之间使用`&` 分隔，例如:

```ruby
$$
\begin{matrix}
1 & x & x^2 \\
1 & y & y^2 \\
1 & z & z^2 \\
\end{matrix}
$$
```

  得到：  
![\begin{matrix} 1 & x & x^2 \\ 1 & y & y^2 \\ 1 & z & z^2 \\ \end{matrix}](https://math.jianshu.com/math?formula=%5Cbegin%7Bmatrix%7D%201%20%26%20x%20%26%20x%5E2%20%5C%5C%201%20%26%20y%20%26%20y%5E2%20%5C%5C%201%20%26%20z%20%26%20z%5E2%20%5C%5C%20%5Cend%7Bmatrix%7D)

### 括号

  如果要对矩阵加括号，可以像上文中提到的一样，使用`\left` 与`\right` 配合表示括号符号。也可以使用特殊的`matrix` 。即替换`\begin{matrix}…\end{matrix}` 中`matrix` 为`pmatrix` ，`bmatrix` ，`Bmatrix` ，`vmatrix` , `Vmatrix` 。

1. pmatrix`$\begin{pmatrix}1 & 2 \\ 3 & 4\\ \end{pmatrix}$` : ![\begin{pmatrix}1 & 2 \\ 3 & 4\\ \end{pmatrix}](https://math.jianshu.com/math?formula=%5Cbegin%7Bpmatrix%7D1%20%26%202%20%5C%5C%203%20%26%204%5C%5C%20%5Cend%7Bpmatrix%7D)
2. bmatrix`$\begin{bmatrix}1 & 2 \\ 3 & 4\\ \end{bmatrix}$` : ![\begin{bmatrix}1 & 2 \\ 3 & 4\\ \end{bmatrix}](https://math.jianshu.com/math?formula=%5Cbegin%7Bbmatrix%7D1%20%26%202%20%5C%5C%203%20%26%204%5C%5C%20%5Cend%7Bbmatrix%7D)
3. Bmatrix`$\begin{Bmatrix}1 & 2 \\ 3 & 4\\ \end{Bmatrix}$` : ![\begin{Bmatrix}1 & 2 \\ 3 & 4\\ \end{Bmatrix}](https://math.jianshu.com/math?formula=%5Cbegin%7BBmatrix%7D1%20%26%202%20%5C%5C%203%20%26%204%5C%5C%20%5Cend%7BBmatrix%7D)
4. vmatrix`$\begin{vmatrix}1 & 2 \\ 3 & 4\\ \end{vmatrix}$` : ![\begin{vmatrix}1 & 2 \\ 3 & 4\\ \end{vmatrix}](https://math.jianshu.com/math?formula=%5Cbegin%7Bvmatrix%7D1%20%26%202%20%5C%5C%203%20%26%204%5C%5C%20%5Cend%7Bvmatrix%7D)
5. Vmatrix`$\begin{Vmatrix}1 & 2 \\ 3 & 4\\ \end{Vmatrix}$` : ![\begin{Vmatrix}1 & 2 \\ 3 & 4\\ \end{Vmatrix}](https://math.jianshu.com/math?formula=%5Cbegin%7BVmatrix%7D1%20%26%202%20%5C%5C%203%20%26%204%5C%5C%20%5Cend%7BVmatrix%7D)

### 元素省略

  可以使用`\cdots` ：⋯，`\ddots`：⋱ ，`\vdots`：⋮ 来省略矩阵中的元素，如：

```ruby
$$
\begin{pmatrix}
1&a_1&a_1^2&\cdots&a_1^n\\
1&a_2&a_2^2&\cdots&a_2^n\\
\vdots&\vdots&\vdots&\ddots&\vdots\\
1&a_m&a_m^2&\cdots&a_m^n\\
\end{pmatrix}
$$
```

  表示：  
![\begin{pmatrix} 1&a_1&a_1^2&\cdots&a_1^n\\ 1&a_2&a_2^2&\cdots&a_2^n\\ \vdots&\vdots&\vdots&\ddots&\vdots\\ 1&a_m&a_m^2&\cdots&a_m^n\\ \end{pmatrix}](https://math.jianshu.com/math?formula=%5Cbegin%7Bpmatrix%7D%201%26a_1%26a_1%5E2%26%5Ccdots%26a_1%5En%5C%5C%201%26a_2%26a_2%5E2%26%5Ccdots%26a_2%5En%5C%5C%20%5Cvdots%26%5Cvdots%26%5Cvdots%26%5Cddots%26%5Cvdots%5C%5C%201%26a_m%26a_m%5E2%26%5Ccdots%26a_m%5En%5C%5C%20%5Cend%7Bpmatrix%7D)

### 增广矩阵

  增广矩阵需要使用前面的表格中使用到的`\begin{array} ... \end{array}` 来实现。

```swift
$$
\left[  \begin{array}  {c c | c} %这里的c表示数组中元素对其方式：c居中、r右对齐、l左对齐，竖线表示2、3列间插入竖线
1 & 2 & 3 \\
\hline %插入横线，如果去掉\hline就是增广矩阵
4 & 5 & 6
\end{array}  \right]
$$
```

显示为：  
![\left[ \begin{array} {c c | c} 1 & 2 & 3 \\ \hline 4 & 5 & 6 \end{array} \right]](https://math.jianshu.com/math?formula=%5Cleft%5B%20%5Cbegin%7Barray%7D%20%7Bc%20c%20%7C%20c%7D%201%20%26%202%20%26%203%20%5C%5C%20%5Chline%204%20%26%205%20%26%206%20%5Cend%7Barray%7D%20%5Cright%5D)

## **公式标记与引用**

  使用`\tag{yourtag}` 来标记公式，如果想在之后引用该公式，则还需要加上`\label{yourlabel}` 在`\tag` 之后，如`$$a = x^2 - y^3 \tag{1}\label{1}$$` 显示为：  
![a := x^2 - y^3 \tag{1}\label{311}](https://math.jianshu.com/math?formula=a%20%3A%3D%20x%5E2%20-%20y%5E3%20%5Ctag%7B1%7D%5Clabel%7B311%7D)  
  如果不需要被引用，只使用`\tag{yourtag}` ，`$$x+y=z\tag{1.1}$$`显示为：  
![x+y=z\tag{1.1}](https://math.jianshu.com/math?formula=x%2By%3Dz%5Ctag%7B1.1%7D)  
  `\tab{yourtab}` 中的内容用于显示公式后面的标记。公式之间通过`\label{}` 设置的内容来引用。为了引用公式，可以使用`\eqref{yourlabel}` ，如`$$a + y^3 \stackrel{\eqref{1}}= x^2$$` 显示为：  
![a + y^3 \stackrel{\eqref{1}}= x^2](https://math.jianshu.com/math?formula=a%20%2B%20y%5E3%20%5Cstackrel%7B%5Ceqref%7B1%7D%7D%3D%20x%5E2)

或者使用`\ref{yourlabel}` 不带括号引用，如`$$a + y^3 \stackrel{\ref{111}}= x^2$$` 显示为:  
![a + y^3 \stackrel{\ref{1}}= x^2](https://math.jianshu.com/math?formula=a%20%2B%20y%5E3%20%5Cstackrel%7B%5Cref%7B1%7D%7D%3D%20x%5E2)

## **字体**

### 黑板粗体字

此字体经常用来表示代表实数、整数、有理数、复数的大写字母。  
`$\mathbb ABCDEF$`：![\mathbb ABCDEF](https://math.jianshu.com/math?formula=%5Cmathbb%20ABCDEF)  
`$\Bbb ABCDEF$`：![\Bbb ABCDEF](https://math.jianshu.com/math?formula=%5CBbb%20ABCDEF)

### 黑体字

`$\mathbf ABCDEFGHIJKLMNOPQRSTUVWXYZ$` :![\mathbf ABCDEFGHIJKLMNOPQRSTUVWXYZ](https://math.jianshu.com/math?formula=%5Cmathbf%20ABCDEFGHIJKLMNOPQRSTUVWXYZ)  
`$\mathbf abcdefghijklmnopqrstuvwxyz$` :![\mathbf abcdefghijklmnopqrstuvwxyz](https://math.jianshu.com/math?formula=%5Cmathbf%20abcdefghijklmnopqrstuvwxyz)

### 打印机字体

`$\mathtt ABCDEFGHIJKLMNOPQRSTUVWXYZ$` :![\mathtt ABCDEFGHIJKLMNOPQRSTUVWXYZ](https://math.jianshu.com/math?formula=%5Cmathtt%20ABCDEFGHIJKLMNOPQRSTUVWXYZ)

## **参考文档**

# 

链接地址

文档名称

1

`blog.csdn.net/dabokele/article/details/79577072`

[Mathjax公式教程](https://blog.csdn.net/dabokele/article/details/79577072)

2

`blog.csdn.net/ethmery/article/details/50670297`

[基本数学公式语法](https://blog.csdn.net/ethmery/article/details/50670297)

3

`blog.csdn.net/lilongsy/article/details/79378620`

[常用数学符号的LaTeX表示方法](https://blog.csdn.net/lilongsy/article/details/79378620)

4

`www.mathjax.org`

[Beautiful math in all browsers](https://www.mathjax.org/)


$$\begin{equation}\begin{split}

P(y=1|x)&=P(y^*>0|x)\\  

&=P(x'\beta+\epsilon>0)\\

& =i

\end{split}\end{equation}$$