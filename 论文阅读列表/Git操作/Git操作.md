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



#### 教程：

- 菜鸟：https://www.runoob.com/git/git-basic-operations.html
- 廖学锋的官方网站。
