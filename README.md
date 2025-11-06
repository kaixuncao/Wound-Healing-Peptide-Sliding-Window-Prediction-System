# Wound-Healing-Peptide-Sliding-Window-Prediction-System
本软件是一款面向生命科学研究者与药物研发工程师的伤口修复多肽功能预测系统。系统以标准氨基酸序列为输入，结合序列卷积特征、数据驱动的重要k-mer片段统计与理化性质，给出多肽序列及其片段在伤口修复方向上的概率评分，并提供交互式Web服务与批量FASTA文件处理及结果下载能力。

This software is a wound repair peptide function prediction system designed for life science researchers and drug development engineers. The system takes standard amino acid sequences as input, combining sequence convolution features, data-driven statistics of important k-mer fragments, and physicochemical properties to provide probability scores for peptide sequences and their fragments in wound repair applications. It offers interactive web services along with batch FASTA file processing and result download capabilities.

<div align="center">
  
<img width="554" height="420" alt="image" src="https://github.com/user-attachments/assets/a6e79be5-7c42-48c8-9585-3a10284d4ced" />

</div>

<div align="center">
图1. 该软件的结构示意图
  
Figure 1. Schematic Diagram of the Software Architecture
</div>

#使用方法
安装依赖库：使用pip安装所需的依赖库。

#Usage
Install dependencies: Use pip to install the required dependencies.

pip install -r requirements.txt

本项目已提供一批已训练完成的.h5模型文件可直接用于验证和后续分析，您只需运行python cpu_wound_healing_app.py或python gpu_wound_healing_app.py即可，我们推荐您使用cpu_wound_healing_app.py，该软件将自动判别当前系统中有无可用的gpu，如果没有将自动运行在CPU上，运行速度取决于您的CPU性能。
如果您需要通过该算法自行训练模型，请确保按照4.1所示安装相关环境，之后请按照提供的阳性数据集“wound_healing_peptides.fasta”文件所示的标准fasta文件作为您的训练数据，此外我们提供用于验证的阴性数据集为“uniprot_sprot_filter.fasta”，您同样可以更改它以获得更好的性能。

This project provides a set of pre-trained .h5 model files ready for verification and subsequent analysis. Simply run either `python cpu_wound_healing_app.py` or `python gpu_wound_healing_app.py`. We recommend using cpu_wound_healing_app.py. This software automatically detects whether a GPU is available on your system. If no GPU is detected, it will run on the CPU, with execution speed dependent on your CPU performance.
If you wish to train your own model using this algorithm, ensure you install the relevant environment as outlined in Section 4.1. Subsequently, use the provided positive dataset “wound_healing_peptides.fasta” as your training data, adhering to the standard fasta file format. Additionally, we provide a negative dataset for validation named “uniprot_sprot_filter.fasta”. You may also modify this dataset to achieve better performance.

#运行示例

#Running Example

streamlit run wound_healing_app.py

运行以上命令等待屏幕输出如图所示。

Run the above command and wait for the screen output as shown in the figure.

<div align="center">

<img width="554" height="337" alt="image" src="https://github.com/user-attachments/assets/e6464059-bc69-47df-acff-59d53c2af677" />

</div>

运行成功后会自动打开浏览器和相应的网页，若无则您可以通过屏幕上看到相应的本地端口进行浏览，通过浏览器加载该服务网页即可得到如下网页用于提交您的修复肽的片段预测结果，受限于阅读和内容的展示，页面中的每种长度的修复肽预测结果最多展示5条，您可以通过网页中提供的下载按钮下载其余所有的结果。

Upon successful execution, the browser will automatically open to the corresponding webpage. If not, you can view the results by accessing the local port displayed on your screen. Load the service webpage via your browser to access the following page for submitting your predicted repair peptide fragment results. Due to display limitations, the page shows a maximum of five predicted repair peptides per length category. You may download all remaining results using the download button provided on the webpage.

<div align="center">

<img width="553" height="268" alt="image" src="https://github.com/user-attachments/assets/ec44e0c3-5287-4441-82b3-c23774f4934b" />

</div>
