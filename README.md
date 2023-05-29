# ChatGPT_Taiyi-Stable-Diffusion

如果想使用ChatGPT功能请在config.py第二行写入自己的API密钥，确保您的设备能连接ChatGPT官网。

项目运行教程

mkdir ChatGPT_Taiyi-Stable-Diffusion

使用conda创建运行环境

1.创建python环境，这里需要python版本>=3.8

conda create --name chatgpt_taiyi python=3.8

2.安装项目运行所需第三方python库

pip install -r requirements.txt

3.模型下载 

git lfs install 

git clone https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1

放在项目平级目录


3.项目执行

conda activate chatgpt_taiyi

python app.py

项目参考资料：

https://huggingface.co/IDEA-CCNL/Taiyi-Stable-Diffusion-1B-Chinese-v0.1

https://github.com/binary-husky/chatgpt_academic

