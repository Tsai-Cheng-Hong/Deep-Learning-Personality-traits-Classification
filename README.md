# Deep-Learning-Personality-traits-Classification
深度學習-自然語言處理-人格特質分類

這篇文章為使用 Tensorflow 的 Keras 進行文件分類

需要先用 pip install 或 conda install 安裝環境套件

我tensorflow 2.2.0版本python是3.8

然後下載數據集正確的路徑

還有pre-trained model的下載連結 https://drive.google.com/drive/folders/1QEbdEn-DO-23hYCLgaB3f4vP8e8L8iIp?usp=sharing

環境安裝好後自動python PA_zh.py執行程序

模型訓練流程:

人格特質指標：五大人格特質(Big-5)

數據集：MyPersonality[1]、Essays[2]、FriendsPersona[3]

特徵萃取：Word2Vec

特徵選擇: 多層感知器(MLP)、卷積神經網路(CNN)、門控卷積神經網路(GCNN)、變形金剛編碼器(Transformer Encoder)等架構

可以從All_model資料夾中自行排列組合

分類器：三層全連接層

# Reference
[1] F. Celli, F. Pianesi, D. Stillwell, and M. Kosinski, "Workshop on computational personality recognition: Shared task," in Proceedings of the International AAAI Conference on Web and Social Media, 2013, vol. 7, no. 2, pp. 2-5.

[2] N. Majumder, S. Poria, A. Gelbukh, and E. Cambria, "Deep learning-based document modeling for personality detection from text," IEEE Intelligent Systems, vol. 32, no. 2, pp. 74-79, 2017.

[3] H. Jiang, X. Zhang, and J. D. Choi, "Automatic text-based personality recognition on monologues and multiparty dialogues using attentive networks and contextual embeddings (student abstract)," in Proceedings of the AAAI Conference on Artificial Intelligence, 2020, vol. 34, no. 10, pp. 13821-13822.

# 延伸閱讀 - 其他自然語言處理應用
文件分類:https://github.com/Tsai-Cheng-Hong/Deep-Learning-Document-Classification

語意分析:https://github.com/Tsai-Cheng-Hong/Deep-Learning-Semantic-Analysis
