import streamlit as st
import torch
from pathlib import Path
from tokenizers import Tokenizer
from model import build_transformer


SEQ_LEN = 350
D_MODEL = 512
WEIGHTS_PATH = "weights/tmodel_27.pt"
TOKENIZER_SRC_PATH = "tokenizer_en.json"
TOKENIZER_TGT_PATH = "tokenizer_it.json"

st.set_page_config(page_title="EN → IT Translator", page_icon="🇮🇹")
st.title("English → Italian Translator")
st.caption("A from-scratch transformer, trained on OPUS Books (en-it)")


@st.cache_resource
def load_model():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer_src = Tokenizer.from_file(TOKENIZER_SRC_PATH)
    tokenizer_tgt = Tokenizer.from_file(TOKENIZER_TGT_PATH)

    model = build_transformer(
        tokenizer_src.get_vocab_size(),
        tokenizer_tgt.get_vocab_size(),
        SEQ_LEN,
        SEQ_LEN,
        d_model=D_MODEL,
    ).to(device)

    state = torch.load(WEIGHTS_PATH, map_location=device)
    model.load_state_dict(state["model_state_dict"])
    model.eval()

    return model, tokenizer_src, tokenizer_tgt, device


def translate(sentence: str, model, tokenizer_src, tokenizer_tgt, device):
    with torch.no_grad():
        source = tokenizer_src.encode(sentence)
        source = torch.cat(
            [
                torch.tensor([tokenizer_src.token_to_id("[SOS]")], dtype=torch.int64),
                torch.tensor(source.ids, dtype=torch.int64),
                torch.tensor([tokenizer_src.token_to_id("[EOS]")], dtype=torch.int64),
                torch.tensor(
                    [tokenizer_src.token_to_id("[PAD]")] * (SEQ_LEN - len(source.ids) - 2),
                    dtype=torch.int64,
                ),
            ],
            dim=0,
        ).to(device)
        source_mask = (
            (source != tokenizer_src.token_to_id("[PAD]")).unsqueeze(0).unsqueeze(0).int().to(device)
        )
        encoder_output = model.encode(source, source_mask)

        decoder_input = (
            torch.empty(1, 1).fill_(tokenizer_tgt.token_to_id("[SOS]")).type_as(source).to(device)
        )

        while decoder_input.size(1) < SEQ_LEN:
            decoder_mask = (
                torch.triu(
                    torch.ones((1, decoder_input.size(1), decoder_input.size(1))), diagonal=1
                )
                .type(torch.int)
                .type_as(source_mask)
                .to(device)
            )
            out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)
            prob = model.project(out[:, -1])
            _, next_word = torch.max(prob, dim=1)
            decoder_input = torch.cat(
                [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)],
                dim=1,
            )
            if next_word == tokenizer_tgt.token_to_id("[EOS]"):
                break

    return tokenizer_tgt.decode(decoder_input[0].tolist())


model, tokenizer_src, tokenizer_tgt, device = load_model()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if prompt := st.chat_input("Type an English sentence..."):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Translating..."):
            translation = translate(prompt, model, tokenizer_src, tokenizer_tgt, device)
        st.write(translation)

    st.session_state.messages.append({"role": "assistant", "content": translation})
