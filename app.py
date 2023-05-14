import os; os.environ['no_proxy'] = '*' # 避免代理网络产生意外污染
import gradio as gr
from predict import predict
from funtional_picture import infer_text2img
from toolbox import format_io, find_free_port, get_conf
import numpy as np

# 建议您复制一个config_private.py放自己的秘密, 如API和代理网址, 避免不小心传github被别人看到
proxies, WEB_PORT, LLM_MODEL, CONCURRENT_COUNT, AUTHENTICATION, CHATBOT_HEIGHT = \
    get_conf('proxies', 'WEB_PORT', 'LLM_MODEL', 'CONCURRENT_COUNT', 'AUTHENTICATION', 'CHATBOT_HEIGHT')

# 如果WEB_PORT是-1, 则随机选取WEB端口
PORT = find_free_port() if WEB_PORT <= 0 else WEB_PORT
if not AUTHENTICATION: AUTHENTICATION = None

initial_prompt = "Serve me as a writing and programming assistant."
title_html = "<h1 align=\"center\">展示你的机器学习模型</h1>"
description =  """"""

# 问询记录, python 版本建议3.9+（越新越好）
import logging
os.makedirs("work_log", exist_ok=True)
try:logging.basicConfig(filename="work_log/chat_secrets.log", level=logging.INFO, encoding="utf-8")
except:logging.basicConfig(filename="gpt_log/chat_secrets.log", level=logging.INFO)
print("所有问询记录将自动保存在本地目录./gpt_log/chat_secrets.log, 请注意自我隐私保护哦！")

# 一些普通功能模块
from functional import get_functionals
functional = get_functionals()



# 处理markdown文本格式的转变
gr.Chatbot.postprocess = format_io

# 做一些外观色彩上的调整
from theme import adjust_theme, advanced_css
set_theme = adjust_theme()

cancel_handles = []
with gr.Blocks(theme=set_theme, analytics_enabled=False, css=advanced_css) as demo:
    gr.HTML(title_html)
    with gr.Tab("ChatGPT"):
        with gr.Row().style(equal_height=True):
            with gr.Column(scale=2):
                chatbot = gr.Chatbot()
                chatbot.style(height=CHATBOT_HEIGHT/2)
                history = gr.State([])
                with gr.Row():
                    txt = gr.Textbox(show_label=False, placeholder="Input question here.").style(container=False)
                with gr.Row():
                    submitBtn = gr.Button("提交", variant="primary")
                with gr.Row():
                    resetBtn = gr.Button("重置", variant="secondary");
                    resetBtn.style(size="sm")
                    stopBtn = gr.Button("停止", variant="secondary");
                    stopBtn.style(size="sm")

            with gr.Column(scale=1):
                with gr.Row():
                    from check_proxy import check_proxy
                    status = gr.Markdown(f"Tip: 按Enter提交, 按Shift+Enter换行。当前模型: {LLM_MODEL} \n {check_proxy(proxies)}")
                with gr.Accordion("基础功能区", open=True) as area_basic_fn:
                    with gr.Row():
                        for k in functional:
                            variant = functional[k]["Color"] if "Color" in functional[k] else "secondary"
                            functional[k]["Button"] = gr.Button(k, variant=variant)
                with gr.Accordion("展开SysPrompt & 交互界面布局 & Github地址", open=True):
                    system_prompt = gr.Textbox(show_label=True, placeholder=f"System Prompt", label="System prompt", value=initial_prompt)
                    top_p = gr.Slider(minimum=-0, maximum=1.0, value=1.0, step=0.01,interactive=True, label="Top-p (nucleus sampling)",)
                    temperature = gr.Slider(minimum=-0, maximum=2.0, value=1.0, step=0.01, interactive=True, label="Temperature",)
                    checkboxes = gr.CheckboxGroup(["基础功能区", "函数插件区"], value=["基础功能区", "函数插件区"], label="显示/隐藏功能区")
                    gr.Markdown(description)
    with gr.Tab("AI绘画"):
        examples = [
            ["铁马冰河入梦来, 梦幻, 插画"],
            ["东临碣石, 以观沧海, 波涛汹涌, 插画"],
            ["孤帆远影碧空尽，惟见长江天际流,油画"],
            ["动漫化，帅气，插画"],
            ["女孩背影, 日落, 唯美插画"],
        ]
        with gr.Row():
            with gr.Column(scale=1, ):
                image_out = gr.Image(label='输出(output)')
            with gr.Column(scale=1, ):
                image_in = gr.Image(source='upload', elem_id="image_upload", type="pil", label="参考图（非必须）(ref)")
                prompt = gr.Textbox(label='提示词(prompt)')
                submit_btn = gr.Button("生成图像(Generate)")
                with gr.Row(scale=0.5):
                    guide = gr.Slider(2, 15, value=7, step=0.1, label='文本引导强度(guidance scale)')
                    steps = gr.Slider(10, 30, value=20, step=1, label='迭代次数(inference steps)')
                    width = gr.Slider(384, 640, value=512, step=64, label='宽度(width)')
                    height = gr.Slider(384, 640, value=512, step=64, label='高度(height)')
                    strength = gr.Slider(0, 1.0, value=0.8, step=0.02, label='参考图改变程度(strength)')
                    ex = gr.Examples(examples, fn=infer_text2img, inputs=[prompt, guide, steps, width, height],
                                     outputs=image_out)

            submit_btn.click(fn=infer_text2img, inputs=[prompt, guide, steps, width, height, image_in, strength],
                             outputs=image_out)

    # demo.queue(concurrency_count=1, max_size=8).launch()


    # 功能区显示开关与功能区的互动
    def fn_area_visibility(a):
        ret = {}
        ret.update({area_basic_fn: gr.update(visible=("基础功能区" in a))})
        return ret

    checkboxes.select(fn_area_visibility, [checkboxes], [area_basic_fn])
    # 整理反复出现的控件句柄组合
    input_combo = [txt, top_p, temperature, chatbot, history, system_prompt]
    output_combo = [chatbot, history, status]
    predict_args = dict(fn=predict, inputs=input_combo, outputs=output_combo)
    empty_txt_args = dict(fn=lambda: "", inputs=[], outputs=[txt]) # 用于在提交后清空输入栏
    # 提交按钮、重置按钮
    cancel_handles.append(txt.submit(**predict_args)) #; txt.submit(**empty_txt_args) 在提交后清空输入栏
    cancel_handles.append(submitBtn.click(**predict_args)) #; submitBtn.click(**empty_txt_args) 在提交后清空输入栏
    resetBtn.click(lambda: ([], [], "已重置"), None, output_combo)
    # 基础功能区的回调函数注册
    for k in functional:
        click_handle = functional[k]["Button"].click(predict, [*input_combo, gr.State(True), gr.State(k)], output_combo)
        cancel_handles.append(click_handle)
    cancel_handles.append(click_handle)
    # 终止按钮的回调函数注册
    stopBtn.click(fn=None, inputs=None, outputs=None, cancels=cancel_handles)



# gradio的inbrowser触发不太稳定，回滚代码到原始的浏览器打开函数
def auto_opentab_delay():
    import threading, webbrowser, time
    print(f"如果浏览器没有自动打开，请复制并转到以下URL: http://localhost:{PORT}")
    def open():
        time.sleep(2)
        webbrowser.open_new_tab(f"http://localhost:{PORT}")
    threading.Thread(target=open, name="open-browser", daemon=True).start()
auto_opentab_delay()
demo.title = "展示你的机器学习模型"
demo.queue(concurrency_count=CONCURRENT_COUNT).launch(server_name="0.0.0.0", share=True, server_port=PORT, auth=AUTHENTICATION)
