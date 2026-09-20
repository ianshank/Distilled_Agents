import pytest
from scripts.training.distill.dpo_collator import format_dpo_example


class DummyTokenizer:
    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=False):
        # Extremely simple mock of chat template
        result = ""
        for m in messages:
            result += f"<{m['role']}>{m['content']}</{m['role']}>"
        if add_generation_prompt:
            result += "<assistant>"
        return result


@pytest.mark.unit
def test_format_dpo_example():
    example = {
        "prompt": "Write a python script",
        "chosen": "print('hello')",
        "rejected": "import os; os.system('rm -rf /')",
    }
    tokenizer = DummyTokenizer()
    result = format_dpo_example(example, tokenizer)

    assert "prompt" in result
    assert "chosen" in result
    assert "rejected" in result

    assert result["prompt"] == "<user>Write a python script</user><assistant>"
    assert result["chosen"] == "<assistant>print('hello')</assistant>"
    assert result["rejected"] == "<assistant>import os; os.system('rm -rf /')</assistant>"


@pytest.mark.unit
def test_format_dpo_example_with_history():
    example = {
        "history": [{"role": "user", "content": "Hi"}],
        "prompt": "Next step",
        "chosen": "Ok",
        "rejected": "No",
    }
    tokenizer = DummyTokenizer()
    result = format_dpo_example(example, tokenizer)

    assert result["prompt"] == "<user>Hi</user><user>Next step</user><assistant>"
