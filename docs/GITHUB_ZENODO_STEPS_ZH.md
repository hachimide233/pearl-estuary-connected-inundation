# GitHub 发布与 Zenodo 永久 DOI 操作

## 结论先说

只放 GitHub 链接可以作为代码访问地址，但不建议把它作为论文中唯一的长期
存档地址。GitHub 主分支可以被修改、覆盖或删除；Zenodo 会把某一个 GitHub
Release 固化存档并分配 DOI，更适合论文引用。

Ocean & Coastal Management 的公开作者说明目前不能据此断言 DOI 是强制项，
但 GitHub 加 Zenodo DOI 是更稳妥、可核验的投稿做法。

## 什么是永久 DOI

DOI 是数字对象唯一标识符。形式通常为：

    https://doi.org/10.xxxx/zenodo.xxxxxxx

DOI 本身不是文件，而是一个长期解析标识。点击后会进入保存有标题、作者、
版本、许可证、文件清单和发布日期的仓库记录。即使 GitHub 主分支以后更新，
论文引用的那个 Zenodo 版本仍保持不变。

Zenodo 通常提供两类 DOI：

- Version DOI：对应某一次具体发布，例如 v1.0.0。
- Concept DOI：代表整个项目，并指向最新版本。

论文建议引用 Version DOI，以锁定实际用于投稿的代码和表格。

## 第一步：创建 GitHub 仓库

1. 登录 GitHub。
2. 点击右上角 New repository。
3. Repository name 建议填写：
   pearl-estuary-connected-inundation
4. Description 可填写：
   Code and aggregate tables for connected coastal inundation screening in
   the Pearl River Estuary.
5. 选择 Public。
6. 不要让 GitHub 再自动生成 README、License 或 gitignore，因为包内已经有。
7. 创建仓库。

## 第二步：上传文件

把发布包解压后，上传文件夹内部的内容，不要再套一层无关目录。

文件较少时可以在 GitHub 网页选择 Add file 和 Upload files。文件较多时建议
使用 GitHub Desktop 或 Git 命令。

首次提交说明建议为：

    Initial public release package for manuscript

上传后重点检查：

- README 首页是否正常显示；
- CITATION.cff 是否触发 Cite this repository；
- data/derived_tables 下共有 20 个 CSV；
- 仓库中没有 H5、HDF5、TIF、TIFF、NPZ 或大压缩包；
- 没有 E 盘、C 盘、用户名或超算路径。

## 第三步：先连接 Zenodo

建议在创建 GitHub Release 之前连接 Zenodo。

1. 打开 https://zenodo.org/ 并用 GitHub 登录。
2. 进入 GitHub 集成页面。
3. 在仓库列表中找到 pearl-estuary-connected-inundation。
4. 把该仓库的开关切换为 On。
5. 如列表中看不到仓库，检查 Zenodo GitHub 应用是否获得该仓库权限。

Zenodo 的 GitHub 集成只处理公开仓库。

## 第四步：创建 GitHub Release

1. 回到 GitHub 仓库。
2. 点击 Releases，然后选择 Draft a new release。
3. 新建标签 v1.0.0。
4. Release title 填写 v1.0.0 - manuscript release。
5. Release notes 简要写明：
   - scientific analysis and QC scripts;
   - aggregate manuscript tables;
   - core processed InSAR products excluded.
6. 点击 Publish release。

Zenodo 检测到 Release 后会建立不可变存档并分配 DOI。

## 第五步：核对 Zenodo 元数据

在 Zenodo 记录中核对：

- Title：与 CITATION.cff 一致；
- Creators：Baihan Wang, Gan Luo, Yixuan Wang, Wendi Gu, Yi Zhang；
- Affiliation：Shandong University of Science and Technology；
- Version：1.0.0；
- Resource type：Software；
- Description：说明含代码与汇总表，不含核心 InSAR 产品；
- Keywords：land subsidence, connected inundation, vertical datum,
  Pearl River Estuary, SBAS-InSAR, sea-level rise；
- Software licence：MIT；
- Aggregate tables：README 中说明 CC BY 4.0；
- Related identifier：后续加入论文 DOI；投稿前没有论文 DOI 时可暂不填。

## 第六步：把 DOI 写回论文

复制 Zenodo 的 Version DOI，替换
docs/DATA_AVAILABILITY_TEMPLATE.md 中的占位符，并同步更新论文。

推荐写法：

    Code availability

    The Python code ... is available from GitHub at [GitHub URL] and archived
    as release v1.0.0 in Zenodo at https://doi.org/[DOI].

数据可用性部分必须同时说明核心 InSAR 产品没有进入公开仓库，不能写成全部
数据公开。

## DOI 是否必须写进 GitHub 仓库

不是必须。第一次 Release 归档后 DOI 才生成，因此 v1.0.0 存档本身可能还
没有 DOI 徽章。可以在 GitHub 主分支 README 中补上 DOI，或者再发布 v1.0.1。
论文只要引用能够解析的 v1.0.0 Version DOI 即可。

## 只放 GitHub 链接是否行

技术上可以提供 GitHub 链接，但存在三个问题：

1. 链接指向的内容可以变化；
2. 没有固定版本和正式仓库元数据；
3. 不利于长期引用和审稿复核。

因此建议论文同时写 GitHub URL 和 Zenodo Version DOI。GitHub 用于浏览和
协作，Zenodo DOI 用于固定版本和长期引用。

## 后续更新

每次影响论文结果或复现方式的修改都发布新版本，例如 v1.0.1 或 v1.1.0。
Zenodo 会为每个新 Release 分配新的 Version DOI，同时保留项目的 Concept DOI。
