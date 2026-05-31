# -*- coding: utf-8 -*-
"""
UTF-8 byte-level BPE 토크나이저 과제 템플릿.

외부 tokenizer 라이브러리 없이 BPE(Byte Pair Encoding)를 직접 구현합니다.
한국어 NSMC 리뷰를 다루므로 문자열을 글자/공백 단위로 먼저 자르지 말고,
항상 `text.encode("utf-8")`로 byte ID 시퀀스를 만든 뒤 merge를 적용하세요.
"""

import json
from pathlib import Path


PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
BOS_TOKEN = "<bos>"
EOS_TOKEN = "<eos>"

SPECIAL_TOKENS = [PAD_TOKEN, UNK_TOKEN, BOS_TOKEN, EOS_TOKEN]
SPECIAL_IDS = {token: idx for idx, token in enumerate(SPECIAL_TOKENS)}
BYTE_OFFSET = len(SPECIAL_TOKENS)
NUM_BYTES = 256


class BPETokenizer:
    """
    UTF-8 byte-level BPE 토크나이저.

    권장 ID 배치:
    - 0~3: <pad>, <unk>, <bos>, <eos>
    - 4~259: 원본 byte 0~255
    - 260 이상: BPE merge로 생성한 토큰
    """

    def __init__(self, vocab_size: int = 3000):
        self.vocab_size = vocab_size
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

    def _init_special_tokens(self):
        """
        TODO:
        1. 특수 토큰 4개를 고정 ID 0~3에 등록합니다.
        2. byte 0~255를 ID 4~259에 bytes([byte_value]) 형태로 등록합니다.
        """

        # 먼저 <pad>, <unk>, <bos>, <eos>를 정해진 번호에 넣습니다.
        for token, token_id in SPECIAL_IDS.items():
            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        # byte 0~255를 4번부터 259번까지 차례대로 등록합니다.
        # 예: byte 65는 b"A"이고, ID는 65 + 4 = 69입니다.
        for byte_value in range(NUM_BYTES):
            token_id = BYTE_OFFSET + byte_value
            token = bytes([byte_value])

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        # raise NotImplementedError("_init_special_tokens를 구현하세요.")

    def get_pad_id(self):
        """padding 토큰 ID."""
        return SPECIAL_IDS[PAD_TOKEN]

    def get_unk_id(self):
        """unknown 토큰 ID."""
        return SPECIAL_IDS[UNK_TOKEN]

    def get_bos_id(self):
        """문장 시작 토큰 ID."""
        return SPECIAL_IDS[BOS_TOKEN]

    def get_eos_id(self):
        """문장 끝 토큰 ID."""
        return SPECIAL_IDS[EOS_TOKEN]

    def train(self, corpus: str):
        """
        TODO: 코퍼스에서 BPE merge rule과 vocabulary를 학습합니다.

        구현 힌트:
        - `corpus.encode("utf-8")`로 byte ID 시퀀스를 만듭니다.
        - 가장 자주 등장하는 이웃 token pair를 찾습니다.
        - 새 token ID를 만들고, 시퀀스의 해당 pair를 새 ID로 치환합니다.
        - `self.merges`, `self.id_to_token`, `self.token_to_id`를 갱신합니다.
        """
        raise NotImplementedError("BPETokenizer.train을 구현하세요.")

    def save(self, path: str | Path):
        """
        TODO: vocabulary와 merge rule을 JSON 파일로 저장합니다.

        bytes와 tuple은 JSON에 바로 저장할 수 없으므로 type 정보를 함께 저장하세요.
        """
        """
        tokenizer의 vocabulary와 merge rule을 JSON 파일로 저장합니다.

        저장하는 것:
            - vocab_size
            - id_to_token
            - merges

        주의:
            JSON은 bytes와 tuple을 그대로 저장할 수 없기 때문에,
            token마다 type 정보를 함께 저장합니다.
        """

        if not self.id_to_token:
            self._init_special_tokens()

        def serialize_token(token):
            """
            Python token을 JSON에 저장 가능한 형태로 바꿉니다.

            str   -> {"type": "str", "value": "..."}
            bytes -> {"type": "bytes", "value": [byte 값들]}
            tuple -> {"type": "tuple", "value": [left_id, right_id]}
            """

            if isinstance(token, str):
                return {
                    "type": "str",
                    "value": token,
                }

            if isinstance(token, bytes):
                return {
                    "type": "bytes",
                    "value": list(token),
                }

            if isinstance(token, tuple):
                return {
                    "type": "tuple",
                    "value": list(token),
                }

            raise TypeError(f"저장할 수 없는 token 타입입니다: {type(token)}")

        data = {
            "vocab_size": self.vocab_size,
            "id_to_token": {},
            "merges": [],
        }

        for token_id, token in self.id_to_token.items():
            data["id_to_token"][str(token_id)] = serialize_token(token)

        for pair in self.merges:
            data["merges"].append(list(pair))

        path = Path(path)
        path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        # raise NotImplementedError("BPETokenizer.save를 구현하세요.")

    def load(self, path: str | Path):
        """
        TODO: save()로 저장한 JSON 파일을 읽어 vocabulary와 merge rule을 복원합니다.
        """
        """
        save()로 저장한 JSON 파일을 읽어 tokenizer 상태를 복원합니다.

        복원하는 것:
            - vocab_size
            - id_to_token
            - token_to_id
            - merges

        token_to_id는 따로 저장하지 않고,
        복원된 id_to_token을 뒤집어서 다시 만듭니다.
        """

        def deserialize_token(token_data):
            """
            JSON에 저장된 token 정보를 다시 Python token으로 바꿉니다.

            {"type": "str", "value": "..."}
                -> str

            {"type": "bytes", "value": [65]}
                -> b"A"

            {"type": "tuple", "value": [69, 70]}
                -> (69, 70)
            """

            token_type = token_data["type"]
            value = token_data["value"]

            if token_type == "str":
                return value

            if token_type == "bytes":
                return bytes(value)

            if token_type == "tuple":
                return tuple(value)

            raise ValueError(f"알 수 없는 token 타입입니다: {token_type}")

        path = Path(path)
        data = json.loads(path.read_text(encoding="utf-8"))

        self.vocab_size = data["vocab_size"]
        self.id_to_token = {}
        self.token_to_id = {}
        self.merges = []

        for token_id_text, token_data in data["id_to_token"].items():
            token_id = int(token_id_text)
            token = deserialize_token(token_data)

            self.id_to_token[token_id] = token
            self.token_to_id[token] = token_id

        for pair in data["merges"]:
            self.merges.append(tuple(pair))

        # raise NotImplementedError("BPETokenizer.load를 구현하세요.")

    def encode(self, text: str, add_bos_eos: bool = False) -> list[int]:
        """
        TODO: 문자열을 token ID 리스트로 변환합니다.

        구현 힌트:
        - 먼저 UTF-8 byte ID 리스트를 만듭니다.
        - train/load에서 얻은 merge rule을 학습 순서대로 적용합니다.
        - add_bos_eos=True이면 앞뒤에 bos/eos ID를 붙입니다.
        """

        if not self.id_to_token:
            self._init_special_tokens()

        # 1. 문자열을 UTF-8 byte로 바꿉니다.
        text_bytes = text.encode("utf-8")

        # 2. byte 값을 기본 token ID로 바꿉니다.
        token_ids = []

        for byte_value in text_bytes:
            token_id = BYTE_OFFSET + byte_value
            token_ids.append(token_id)

        # 3. 학습된 merge rule을 순서대로 적용합니다.
        #
        # 예를 들어 pair가 (69, 70)이고, 이 pair의 새 token id가 260이라면
        # [69, 70, 71]은 [260, 71]이 됩니다.
        for pair in self.merges:
            merged_token_id = self.token_to_id[pair]

            new_token_ids = []
            index = 0

            while index < len(token_ids):
                has_next_token = index < len(token_ids) - 1

                if has_next_token:
                    current_pair = (token_ids[index], token_ids[index + 1])

                    if current_pair == pair:
                        new_token_ids.append(merged_token_id)
                        index += 2
                        continue

                new_token_ids.append(token_ids[index])
                index += 1

            token_ids = new_token_ids

        # 4. 문장 시작/끝 토큰이 필요하면 붙입니다.
        if add_bos_eos:
            token_ids = [self.get_bos_id()] + token_ids + [self.get_eos_id()]

        return token_ids
        raise NotImplementedError("BPETokenizer.encode를 구현하세요.")

    def decode(self, ids: list[int], skip_special: bool = True) -> str:
        """
        TODO: token ID 리스트를 문자열로 복원합니다.

        주의:
        - merge token은 원본 byte token까지 재귀적으로 펼칩니다.
        - byte를 하나씩 decode하지 말고, 마지막에 `bytes(...).decode("utf-8")`를 한 번만 호출합니다.
        """

        """
        token ID 리스트를 문자열로 복원합니다.

        Input:
            ids:
                token ID 리스트입니다.
                예: [69]
                예: [238, 180, 132]
                예: [2, 69, 3]

            skip_special:
                True이면 <pad>, <unk>, <bos>, <eos> 같은 특수 토큰은 복원 결과에서 제외합니다.

        Output:
            복원된 문자열입니다.

            예:
                decode([69])
                -> "A"

            예:
                decode([238, 180, 132])
                -> "가"

            예:
                decode([2, 69, 3], skip_special=True)
                -> "A"

        중요한 점:
            - 기본 byte token은 bytes 객체입니다.
            - BPE merge token은 (left_id, right_id) 형태의 tuple입니다.
            - merge token은 바로 문자로 바꾸지 않고, 내부 token들을 재귀적으로 펼쳐 byte로 만듭니다.
            - 한글은 UTF-8에서 여러 byte로 이루어질 수 있으므로 byte 하나씩 decode하면 안 됩니다.
            - 모든 byte를 합친 뒤 마지막에 decode("utf-8")를 한 번만 호출합니다.
        """

        if not self.id_to_token:
            self._init_special_tokens()

        def token_to_bytes(token_id: int) -> bytes:
            """
            token ID 하나를 bytes로 복원합니다.

            token 종류:
                - str:
                    <pad>, <unk>, <bos>, <eos> 같은 특수 토큰
                - bytes:
                    기본 byte token
                - tuple:
                    BPE merge token, 예: (69, 70)
            """

            token = self.id_to_token[token_id]

            # 특수 토큰입니다. 보통 decode 결과에서는 제외합니다.
            if isinstance(token, str):
                if skip_special:
                    return b""
                return token.encode("utf-8")

            # 기본 byte token입니다.
            # 예: id 69 -> b"A"
            if isinstance(token, bytes):
                return token

            # BPE merge token입니다.
            # 예: id 260 -> (69, 70)
            # 이 경우 왼쪽 token과 오른쪽 token을 각각 bytes로 펼친 뒤 이어 붙입니다.
            if isinstance(token, tuple):
                left_id, right_id = token
                left_bytes = token_to_bytes(left_id)
                right_bytes = token_to_bytes(right_id)
                return left_bytes + right_bytes

            raise TypeError(f"지원하지 않는 token 타입입니다: {type(token)}")

        byte_chunks = []

        for token_id in ids:
            token_bytes = token_to_bytes(token_id)
            byte_chunks.append(token_bytes)

        text_bytes = b"".join(byte_chunks)

        return text_bytes.decode("utf-8")

        raise NotImplementedError("BPETokenizer.decode를 구현하세요.")
