# markdown操作示例

## 1.块元素

### 1. 标题

>这些红色的边都是用>打出来的
>
>` 使用 #做前缀表示标题`
>
>例如 ：
>
>#(此处有空格)这是一级标题 
>
>##(此处有空格)这是二级标题 
>
>######(此处有空格)这是六级标题

### 2.引用文字

>`引用就是这个部分使用 > 进行引用`
>
>>
>>
>>再次输入 >可以二级引用

### 3.列表操作

>1.无序列表
>
>` 使用*(空格)进行创建`
>
>例如：
>
>* 第一个
>* 第二个
>
>2.有序列表
>
>` 使用数字1./2./3.(空格)进行创建`
>
>例如：
>
>1. 第一个
>2. 第二个

### 4.任务列表

>` 使用-(空格)[空/x] (空格)表示没完成/完成`
>
>建议先挨着打-[]然后再打中间的空格。
>
>例如：
>
>- [ ] 未完成的任务
>
>- [x] 已完成的任务
>
>注意的是对勾可以点击修改状态

### 5.栅栏式代码块

>Typora仅支持 Github Flavored Markdown 中的栅栏式代码块。不支持 markdown 中的原始代码块。
>
>使用栅栏式代码块很简单：输入````之后输入一个可选的语言标识符，然后按` return键后输入代码，我们将通过语法高亮显示它：
>
>例如：输入```
>
>```
>```
>
>写一段代码示例：
>
>```python
>def hello(slef):
>    print("Hello World.")
>```
>
>或者使用```所用语法进行输入：
>
>例如：```+java
>
>```java
>function test(){
>    System.out.println("Hello World.");
>}
>```

### 6.数学公式块

>您可以使用 **MathJax** 渲染 *LaTeX* 数学表达式。
>
>输入 `$$`, 然后按“return”键将触发一个接受*Tex / LaTex*源代码的输入区域。
>
>例如：
>$$
>\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix} 
>\mathbf{i} & \mathbf{j} & \mathbf{k} \\
>\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
>\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0 \\
>\end{vmatrix}
>$$
>
>
>在 markdown 源文件中，数学公式块是由’$$’标记包装的 *LaTeX* 表达式：
>
>```
>$$
>\mathbf{V}_1 \times \mathbf{V}_2 =  \begin{vmatrix} 
>\mathbf{i} & \mathbf{j} & \mathbf{k} \\
>\frac{\partial X}{\partial u} &  \frac{\partial Y}{\partial u} & 0 \\
>\frac{\partial X}{\partial v} &  \frac{\partial Y}{\partial v} & 0 \\
>\end{vmatrix}
>$$
>```

### 7.表格

>输入 `| First Header | Second Header |` 并按下 `return` 键将创建一个包含两列的表。
>
>创建表后，焦点在该表上将弹出一个表格工具栏，您可以在其中调整表格，对齐或删除表格。您还可以使用上下文菜单来复制和添加/删除列/行。
>
>可以跳过以下描述，因为表格的 markdown 源代码是由typora自动生成的。
>
>例如：
>
>| 第一列 | 第二列 |
>| ------ | ------ |
>| one    | three  |
>| four   | five   |
>| six    | seven  |
>|        |        |

### 8.注脚

>` 可以使用[^ 内容]:进行注脚创建`
>
>例如：
>
>**对加粗内容添加注脚**[^1]

### 9.水平线

>`输入***或者---按回车即可回执一条水平线`
>
>例如：
>
>***

### 10.目录

>` 输入[toc]然后回车即可根据目前的标题创建目录`
>
>例如：
>
>[toc]

## 2.Span元素

> 输入后Span元素会被立即解析并呈现。在这些span元素上移动光标会将这些元素扩展为markdown源代码。以下将解释这些span元素的语法。

### 1.链接

>Markdown 支持两种类型的链接：内联和引用。
>
>在这两种样式中，链接文本都写在[方括号]内。
>
>要创建内联链接，请在链接文本的结束方括号后立即使用一组常规括号。在常规括号内，输入URL地址，以及可选的用引号括起来的链接标题。
>
>例如：
>
>This is [an example](http://example.com/ "Title") inline link. 
>
>[This link](http://example.net/) has no title attribute.

### 2.内部链接

>**您可以将常规括号内的 href 设置为文档内的某一个标题**，这将创建一个书签，允许您在单击后跳转到该部分。
>
>例如：
>
>Command(在Windows上：Ctrl) + 单击 [此链接](https://support.typoraio.cn/zh/Markdown-Reference/#块元素) 将跳转到标题 `块元素`处。 要查看如何编写，请移动光标或按住 `⌘` 键单击以将元素展开为 Markdown 源代码。

### 3.参考链接

>参考样式链接使用第二组方括号，在其中放置您选择的标签以标识链接：
>
>```
>This is [an example][id] reference-style link.
>
>然后，在文档中的任何位置，您可以单独定义链接标签，如下所示：
>
>[id]: http://example.com/  "Optional Title Here"
>```

### 4.隐式链接

>隐式链接名称快捷方式允许您省略链接的名称，在这种情况下，链接文本本身将用作名称。只需使用一组空的方括号，例如，将“Google”一词链接到google.com网站，您只需写下：
>
>```
>[Google][]
>然后定义链接：
>
>[Google]: http://google.com/
>```

### 5.插入URL网址

>Typora允许您将 URL 作为链接插入，用 `<`括号括起来`>`。
>
>`<i@typora.io>` 成为 [i@typora.io](mailto:i@typora.io).
>
>Typora也将自动链接标准URL。例如： www.google.com.

### 6.插入图片

>图像与链接类似， 但在链接语法之前需要添加额外的 `!` 字符。 图像语法如下所示：
>
>```
>![替代文字](/path/to/img.jpg)
>
>![替代文字](/path/to/img.jpg "可选标题")
>```
>
>您可以使用拖放操作从图像文件或浏览器来插入图像。并通过单击图像修改 markdown 源代码。如果图像在拖放时与当前编辑文档位于同一目录或子目录中，则将使用相对路径。

### 7.强调（变斜体）

>Markdown 将星号 (`*`) 和下划线(`_`) 视为强调的指示。
>
>例如：
>
>*斜体喽*
>
>_又一个斜体_

### 8.加粗和高光

>加粗（文字两侧各放两个*或者_）
>
>例如：
>
>**加粗文字**
>
>高光（文字两侧各放一个`）
>
>例如：
>
>`高光`

### 9.删除线与下划线

>删除线（两侧分别两个~）
>
>例如：
>
>~~错误文字~~
>
>下划线（使用html语法的<u></u>实现）
>
>例如：
>
><u>下划线</u>

### 10.表情符

>输入表情符号的语法是 `:smile:`
>
>用户可以通过 `ESC` 按键触发表情符号的自动完成建议，或者在偏好设置面板里启用后自动触发表情符号。此外，还支持直接从 `Edit` -> `Emoji & Symbols` 菜单栏输入UTF8表情符号字符。
>
>例如：
>
>:smile:

### 11.上标/下标/高亮

>### 上标
>
>要使用此功能，首先，请在 `偏好设置` 面板 -> `Markdown扩展语法` 选项卡中启用它。然后用 `^` 或者<sup>来包裹上标内容，例如： `X^2^`。
>
>### 下标
>
>要使用此功能，首先，请在 `偏好设置` 面板 -> `Markdown扩展语法` 选项卡中启用它。然后用 `~` 或者<sub>来包裹下标内容，例如： `H~2~O`, `X~long\ text~`/
>
>### 高亮
>
>要使用此功能，首先，请在 `偏好设置` 面板 -> `Markdown扩展语法` 选项卡中启用它。然后用 `==` 来包裹高亮内容，例如： `==highlight==`。

## 3.HTML

>可以使用HTML来设置纯 Markdown 不支持的内容，例如， `<span style="color:red">this text is red</span>` 用于添加红色文本。

### 1.嵌入内容

>有些网站提供基于iframe的嵌入代码，您也可以将其粘贴到Typora中。
>
>例如：
>
>```html
><iframe height='265' scrolling='no' title='Fancy Animated SVG Menu' src='http://codepen.io/jeangontijo/embed/OxVywj/?height=265&theme-id=0&default-tab=css,result&embed-version=2' frameborder='no' allowtransparency='true' allowfullscreen='true' style='width: 100%;'></iframe>
>```

### 2.视频

>可以使用 `<video>` HTML标记嵌入视频。
>
>例如：
>
>```html
><video src="xxx.mp4" />
>```

