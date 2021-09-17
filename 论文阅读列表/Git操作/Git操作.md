### 特定问题下的命令选择

![image-20210916204123460](/home/cold/PaperReadFastly/PaperRead/论文阅读列表/Git操作/Git操作.assets/image-20210916204123460.png)

#### 1 如果要撤销Git reset --hard 操作

> 【转】如何撤销git reset --hard操作 https://blog.csdn.net/Qidi_Huang/article/details/53839591

````python
$ git reflog
b7057a9 HEAD@{0}: reset: moving to b7057a9
98abc5a HEAD@{1}: commit: more stuff added to foo
b7057a9 HEAD@{2}: commit (initial): initial commit

$ git reset --hard 98abc5a  # 成也reset 败也reset。

$ git log
* 98abc5a (HEAD, master) more stuff added to foo
* b7057a9 initial commit
````



#### 2 如果要去除暂存区中的过大的文件

<u>进行过 add 和 commit，从提交的文件中移除超出限制的大文件，**但保留在磁盘中。**</u>

> git push时报错文件过大：remote: error: GH001: Large files detected. https://blog.csdn.net/SjwFdb_1__1/article/details/109499214

```
git rm --cached public/dist/MergerVideo.exe（此处为错误提示的文件路径及名称）
git rm --cached fogV1/build/JudgeFog5/PKG-00.pkg
（如上，将所有大文件移除）

git commit --amend -CHEAD

git push [远程仓库] [远程仓库分支]
```



#### 3 如果要大规模地从暂存区和工作区中撤销和删除（慎用）

**认识一个reset ，学会了relog，因为reset 使得本地中的文件（称为工作区）也退回，意味着某些文件被删除掉了**。

```python
1、撤掉

a、如果还没 git add file ，使用该指令进行撤销：  git checkout -- fileName  
b、如果已经git add file  ， 但是没有 git commit -m ""  分两步操作：
 b-1、git reset HEAD readme.txt
 b-2、git  status
 b-3、git checkout -- file
c、如果已经git add file 并且已经 git commit ,那么回退版本办法是：
 c-1、通过 git log 或者 git log --pretty=oneline 、git reflog
 c-2、找到对应的commit id进行回退：git reset --hard 1094a  # 这一步 任何时候都要注意使用。


2、删除

现在你有两个选择，一是确实要从版本库中删除该文件，那就用命令git rm删掉，并且git commit：
a、git rm test.txt  # 表示删除工作区的文件，慎用
b、git commit -m "remove test.txt"


另一种情况是删错了，因为版本库里还有呢，所以可以很轻松地把误删的文件恢复到最新版本：
git checkout -- test.txt
```

#### 4 慎用的操作

凡是涉及**工作区文件**的改动命令都要注意。

- [git rm](https://www.runoob.com/git/git-rm.html) *删除工作区文件*。

- git mv 移动或重命名工作区文件

<img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/Git操作/Git操作.assets/image-20210916220100978.png" alt="image-20210916220100978" style="zoom:67%;" />

- 还有git reset --hard HEAD;



#### 5 查看远程主机名和远程主机的分支、本地分支

认识到这三者对**push**的操作，能够操作灵活。

- 本地分支：main；git branch
- 远程仓库分支名：git branch -r
- 远程仓库名：git remote

> https://blog.csdn.net/ycg33/article/details/105258682
>
> 



### PRO GIT BOOK 

#### 开始

- DVCS分布式版本控制系统中，每一次克隆操作，都是对代码仓库的完整备份，包括完整的历史记录。

    <img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/Git操作/Git操作.assets/image-20210917000041573.png" alt="image-20210917000041573" style="zoom:50%;" />

- 绝大多数的 Linux 内核维护工作都花在了**提交补丁和保存归档**的繁琐事务上（1991－2002年间）。 到 2002 年，整个项目组开始启用一个专有的分布式版本控制系统BitKeeper 来管理和维护代码。

- Git与基于差异的版本控制系统的不同：如果文件没有修改，Git 不再重新存储该文件，而是只保留一个链接指向之前存储的文件，**而不是依旧保存**

- 近乎所有的操作都是本地执行，包括浏览项目历史。要浏览项目的历史，Git 不需外连到服务器去获取历史，然后再显示出来——它只需直接从本地数据库中读取。 你能立即看到项目历史。如果你想查看当前版本与一个月前的版本之间引入的修改， Git 会查找到一个月前的文件做一次本地的差异计算，而不是由远程服务器处理或从远程服务器拉回旧版本文件再来本地处理。

- Git 保证完整性
    Git 中所有的数据在存储前都计算校验和，然后以**校验和**来引用。 这意味着不可能在 Git 不知情时更改任何文件内容或目录内容。

    Git 用以计算校验和的机制叫做 **SHA-1 散列（hash，哈希**）。 这是一个由 **40 个十六进制字符（0-9 和 a-f）组成**的字符串，基于 Git 中文件的**内容或目录结构**计算出来。

    实际上，Git 数据库中保存的信息都是以**文件内容的哈希值来索引，而不是文件名**。

- 你执行的 Git 操作，几乎只往 Git 数据库中 添加 数据。 你很难使用 Git 从数据库中删除数据，**也就是说 Git 几乎不会执行任何可能导致文件不可恢复的操作**。

- 如果你希望后面的学习更顺利，请记住下面这些关于 Git 的概念。 Git 有三种状态，你的文件可能处于其中之一： 已修改（modified） 和 已暂存（staged）\已提交（committed）。

    • 已修改表示修改了文件，但还没保存到数据库中。
    • 已暂存表示对**一个已修改文件的当前版本做了标记**，使之包含在下次提交的**快照**中。
    • 已提交表示数据已经安全地保存在本地数据库中。

    对应，工作区、暂存区以及 Git 目录；

    <img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/Git操作/Git操作.assets/image-20210917001526910.png" alt="image-20210917001526910" style="zoom:50%;" />

    工作区是对项目的某个版本独立提取出来的内容。 这些从 Git 仓库的压缩数据库中提取出来的文件，放在磁盘上供你使用或修改。
    暂存区是一个文件，保存了下次将要提交的文件列表信息，一般在 Git 仓库目录中。 按照 Git 的术语叫做“索引”，不过一般说法还是叫“暂存区”。
    Git 仓库目录是 Git 用来保存项目的元数据和对象数据库的地方。 这是 Git 中最重要的部分，从其它计算机克隆仓库时，复制的就是这里的数据。

- 安装完 Git 之后，要做的第一件事就是设置你的用户名和邮件地址。

    如果使用了 --global 选项，那么该命令只需要运行一次，因为之后无论你在该系统上做任何事情， Git 都会使用那些信息。 **当你想针对特定项目使用不同的用户名称与邮件地址时，可以在那个项目目录下运行没有 --global 选项的命令来配置**。

- 检查配置信息

    如果想要检查你的配置，可以使用 git config --list 命令来列出所有 Git 当时能找到的配置

    ```
    (paper) cold@cold-OMEN-by-HP-Laptop-15-dc1xxx:~/PaperReadFastly/PaperRead$ git config --list
    user.email=1437623218@qq.com
    user.name=OMEN_Ubuntu
    core.repositoryformatversion=0
    core.filemode=true
    core.bare=false
    core.logallrefupdates=true
    remote.origin.url=git@github.com:MarToonLi/PaperRead.git
    remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
    branch.main.remote=origin
    branch.main.merge=refs/heads/main
    (paper) cold@cold-OMEN-by-HP-Laptop-15-dc1xxx:~/PaperReadFastly/PaperRead$ git config --list --show-origin
    file:/home/cold/.gitconfig      user.email=1437623218@qq.com
    file:/home/cold/.gitconfig      user.name=OMEN_Ubuntu
    file:.git/config        core.repositoryformatversion=0
    file:.git/config        core.filemode=true
    file:.git/config        core.bare=false
    file:.git/config        core.logallrefupdates=true
    file:.git/config        remote.origin.url=git@github.com:MarToonLi/PaperRead.git
    file:.git/config        remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
    file:.git/config        branch.main.remote=origin
    file:.git/config        branch.main.merge=refs/heads/main
    
    ```

#### Git 基础

- 通常有两种获取 Git 项目仓库的方式：

    - 将尚未进行版本控制的本地目录转换为 Git 仓库；

    - 从其它服务器 克隆 一个已存在的 Git 仓库。

    两种方式都会在你的本地机器上得到一个**工作就绪的 Git 仓库**。

- 当你执行 git clone 命令的时候，默认配置下远程 Git 仓库中的每一个文件的**每一个版本**都将被拉取下来。

- **在服务器上搭建 Git** 将会介绍所有这些协议（SSH、HTTP）在服务器端如何配置使用，以及各种方式之间的利弊。

- 请记住，**你工作目录下的每一个文件都不外乎这两种状态：已跟踪 或 未跟踪**。 已跟踪的文件是指那些被纳入了版本控制的文件，在上一次快照中有它们的记录，**在工作一段时间后， 它们的状态可能是未修改，已修改或已放入暂存区**。简而言之，已跟踪的文件就是 Git 已经知道的文件。

    <img src="/home/cold/PaperReadFastly/PaperRead/论文阅读列表/Git操作/Git操作.assets/image-20210917004101922.png" alt="image-20210917004101922" style="zoom:67%;" />

    

- **使用命令 git add 开始跟踪一个文件**

    如果此时提交commit，那么该文件在你运行 git add 时的**版本将被留存在后续的历史记录**中。

- git add 命令使用**文件或目录的路径**作为参数；如果参数是目录的路径，该命令将递归地跟踪该**目录下的所有文件**。

- Changes not staged for commit 这行下面的文件，**说明已跟踪文件的内容发生了变化，但还没有放到暂存区**。

- git status会显示四种文件状态下的文件：

    - 未跟踪：Untracked files
    - 处于暂存区，等待提交：Changes to be committed
    - 已修改但未放到暂存区：Changes not staged for commit

- **现在 CONTRIBUTING.md 文件同时出现在暂存区和非暂存区。 这怎么可能呢**？ 好吧，实际上 Git 只不过暂存了你运行 git add 命令时的版本。 如果你现在提交，CONTRIBUTING.md 的版本是你最后一次运行git add 命令时的那个版本，而不是你运行 git commit 时，在工作目录中的当前版本。 所以，运行了 git add 之后又作了修订的文件，需要重新运行 git add 把最新版本重新暂存起来

##### 忽略文件：参考该书部分

**一般我们总会有些文件无需纳入 Git 的管理，也不希望它们总出现在未跟踪文件列表**。 通常都是些自动生成的文件，比如日志文件，或者编译过程中创建的临时文件等。 在这种情况下，我们可以创建一个名为 .gitignore的文件，列出要忽略的文件的模式。

```
# 忽略所有的 .a 文件
*.a
# 但跟踪所有的 lib.a，即便你在前面忽略了 .a 文件
!lib.a
# 只忽略当前目录下的 TODO 文件，而不忽略 subdir/TODO
/TODO
# 忽略任何目录下名为 build 的文件夹
build/
# 忽略 doc/notes.txt，但不忽略 doc/server/arch.txt
doc/*.txt
# 忽略 doc/ 目录及其所有子目录下的 .pdf 文件
doc/**/*.pdf
```

GitHub 有一个十分详细的针对数十种项目及语言的 .gitignore 文件列表， 你可以在 https://github.com/github/gitignore 找到它

在最简单的情况下，一个仓库可能只根目录下有一个 .gitignore 文件，它递归地应用到整个仓库中。 然而，**子目录下也可以有额外的 .gitignore 文件**。子目录中**的 .gitignore文件中的规则只作用于它所在的目录中**。

##### 查看已暂存和未暂存的修改

如果 git status 命令的输出对于你来说过于简略，而你想知道具体修改了什么地方，可以用 git diff 命令。

git diff 能通过**文件补丁**的格式更加具体地显示**哪些行**发生了改变。

- 不加参数直接输入 git diff：查看**尚未暂存的文件**更新了哪些部分，此命令比较的是**工作目录中当前文件**和**暂存区域**快照之间的差异。 也就是修改之后还没有暂存起来的变化内容。
- git diff --staged|--cached （--staged 和 --cached 是同义词）：比对**已暂存文件**与**最后一次提交的文件**的差异。

请注意，git diff 本身只显示尚未暂存的改动，而不是自上次提交以来所做的所有改动。 **所以有时候你一下子暂存了所有更新过的文件，运行 git diff 后却什么也没有，就是这个原因**

##### 跳过使用暂存区域：面向已经被跟踪过的所有文件

**尽管使用暂存区域的方式可以精心准备要提交的细节，但有时候这么做略显繁琐**。 Git 提供了一个跳过使用暂存区域的方式， 只要在提交的时候，给 git commit 加上 -a 选项，Git 就会自动把所有**已经跟踪过的文件**暂存起来一并提交，从而跳过 git add 步骤；

- 这很方便，但是要小心，有时这个选项会将不需要的文件添加到提交中
- 它的好处除了避免了繁琐以外，它对commit的一个特点做了捷径。即comiit的特点是：**提交时记录的是放在暂存区域的快照，而不包括修改但未置于暂存区域的文件。**

##### 移除文件

要从 Git 中移除某个文件，就必须要从**已跟踪文件清单中移除（确切地说，是从暂存区域移除）**，然后提交。可以用 git rm 命令完成此项工作，并**连带从工作目录中删除指定的文件**，这样以后就不会出现在未跟踪文件清单中了。

###### 第一种情况：未被跟踪的文件被删除

- rm  file ;

###### 第二种情况：已经修改但未进入暂存区的文件被删除

- rm file ----> git rm file -f 

###### 第三种情况：已经进入暂存区的文件被删除

- rm file ----> git rm file -f 

（**第二种情况和第三种情况一样，主要是该文件已经被跟踪，已经被纳入版本管理，如果要实现从版本管理系统中去除对某个文件的追踪，就必须使用-f参数。是一种安全特性。**）



**但是上面的三种情况，暂存区中不仅没有了该文件，连工作目录下也不再有该文件。—— 因此，这样的文件永久不可能被恢复**



**git rm 操作结束后，将不需要再使用commit等操作。**





###### 第四种情况：仅删除暂存区中的文件，但保留其在工作目录中的备份

换句话说，你想让文件保留在磁盘，但是并不想让 Git 继续跟踪。 当你忘记添加 .gitignore 文件，不小心把一个很大的日志文件或一堆 .a 这样的编译生成文件添加到暂存区时，这一做法尤其有用。 为达到这一目的，使用 --cached 选项：

```
git rm --cached README
```



##### 移动文件

Git 并不显式跟踪文件移动操作。 如果在 Git 中重命名了某个文件，**仓库中存储的元数据并不会体现出这是一次改名操作**。

实际上，即便此时查看状态信息，也会明白无误地看到关于重命名操作的说明：

```
$ git mv README.md README
$ git status
On branch master
Your branch is up-to-date with 'origin/master'.
Changes to be committed:
(use "git reset HEAD <file>..." to unstage)
renamed:
README.md -> README 
```

**该信息体现在 Changes to be committed:部分，意味着，该操作执行后，自动执行了add操作。**

即：

```
$ mv README.md README
$ git rm README.md
$ git add README
```

###### 查看添加或删除对某一个特定函数（字符串）的提交

另一个非常有用的过滤器是 -S（俗称“pickaxe”选项，取“用鹤嘴锄在土里捡石头”之意）， 它接受一个字符串参数，并且只会显示那些添加或删除了该字符串的提交。 假设你想找出添加或删除了对某一个特定函数的引用的提交，可以调用：

```
git log -S function_name
```















### 教程

- 菜鸟：https://www.runoob.com/git/git-basic-operations.html
- 廖学锋的官方网站。
- https://blog.csdn.net/weelyy/article/details/82823798 —— 比较全的git的命令以及解释。
- https://git-scm.com/docs/git-branch  -- 官方档案
