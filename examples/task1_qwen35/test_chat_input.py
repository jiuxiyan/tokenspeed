# Copyright (c) 2026 LightSeek Foundation
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""Chat token IDs and IPC serialization with a local, real fast tokenizer."""

import importlib.util
import unittest

from examples.task1_qwen35.common import chat_input_ids


@unittest.skipUnless(
    all(
        importlib.util.find_spec(name)
        for name in ("transformers", "tokenizers", "msgspec")
    ),
    "requires Transformers, tokenizers and msgspec",
)
class ChatInputTests(unittest.TestCase):
    def make_tokenizer(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from tokenizers.pre_tokenizers import WhitespaceSplit
        from transformers import PreTrainedTokenizerFast

        backend = Tokenizer(
            WordLevel(
                {
                    "[UNK]": 0,
                    "user": 1,
                    "assistant": 2,
                    "hello": 3,
                    "there": 4,
                    "你好": 5,
                    "thinking": 6,
                },
                unk_token="[UNK]",
            )
        )
        backend.pre_tokenizer = WhitespaceSplit()
        return PreTrainedTokenizerFast(
            tokenizer_object=backend,
            unk_token="[UNK]",
            chat_template=(
                "{% for message in messages %}"
                "{{ message['role'] }} {{ message['content'] }} "
                "{% endfor %}"
                "{% if add_generation_prompt %}assistant {% endif %}"
                "{% if enable_thinking %}thinking {% endif %}"
            ),
        )

    def assert_ipc_round_trip(self, ids, expected):
        import msgspec

        self.assertIsInstance(ids, list)
        self.assertTrue(all(type(token) is int for token in ids))
        self.assertEqual(ids, expected)
        payload = msgspec.msgpack.encode({"input_ids": ids})
        self.assertEqual(msgspec.msgpack.decode(payload), {"input_ids": expected})

    def test_single_message_is_integer_ids_with_generation_prompt(self):
        ids = chat_input_ids(
            self.make_tokenizer(), [{"role": "user", "content": "hello"}]
        )
        self.assert_ipc_round_trip(ids, [1, 3, 2])

    def test_multiturn_unicode_template_remains_flat_and_thinking_disabled(self):
        ids = chat_input_ids(
            self.make_tokenizer(),
            [
                {"role": "user", "content": "hello"},
                {"role": "assistant", "content": "there"},
                {"role": "user", "content": "你好"},
            ],
        )
        self.assert_ipc_round_trip(ids, [1, 3, 2, 4, 1, 5, 2])


if __name__ == "__main__":
    unittest.main(verbosity=2)
