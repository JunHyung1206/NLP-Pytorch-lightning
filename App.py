import streamlit as st
import argparse
from streamlit_chat import message

from Instances.instance import load_model, run_sts
from omegaconf import OmegaConf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--saved_model",
    "-s",
    default="SaveModels/klue/roberta-small_swept-morning-1/epoch=9-test_pearson=0.8735774755477905.ckpt",
    help="저장된 모델의 파일 경로를 입력해주세요. 예시: save_models/klue/roberta-small/epoch=?-step=?.ckpt 또는 save_models/model.pt",
)
args, _ = parser.parse_known_args()
conf = OmegaConf.load(f"./config/base_config.yaml")
model, tokenizer = load_model(args, conf)
print(model)

# st.balloons() # 풍선 나옴
st.title("Streamlit STS")
if "input" not in st.session_state:
    st.session_state["input"] = ""
if "input2" not in st.session_state:
    st.session_state["input2"] = ""
if "messages" not in st.session_state:
    st.session_state["messages"] = []

with st.form(key="my_form", clear_on_submit=True):
    text_input = st.text_input(
        "첫 번째 문장을 입력해주세요",
        key="input",
    )
    text_input2 = st.text_input(
        "두 번째 문장을 입력해주세요",
        key="input2",
    )
    submit = st.form_submit_button(label="Check")

if submit:
    # text = tokenizer.special_tokens_map["sep_token"].join([st.session_state["input"], st.session_state["input2"]])
    # st.write(text)
    result = run_sts(st.session_state["input"], st.session_state["input2"], conf, model)
    score = round(result, 1)
    if result > 4.0:
        st.write(f"두 문장이 매우 유사합니다\t\t\t\t유사도 : {score}")
    elif result > 3.0:
        st.write(f"두 문장이 유사합니다\t\t\t\t유사도 : {score}")
    else:
        st.write(f"두 문장이 유사하지 않습니다\t\t\t\t유사도 : {score}")
