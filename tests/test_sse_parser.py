import json

import pytest

from admission_lab.sse import SSEParser, content_event


def test_sse_arbitrary_byte_boundaries_and_metadata():
    events = [json.dumps({"choices": [{"text": "", "index": 0}]}, ensure_ascii=False),
              json.dumps({"choices": [{"text": "你好", "token_ids": [1, 2]}]}, ensure_ascii=False),
              json.dumps({"choices": [], "usage": {"completion_tokens": 2}}), "[DONE]"]
    raw = (": heartbeat\r\n\r\n" + "".join("data: " + e + "\r\n\r\n" for e in events)).encode()
    for size in (1, 2, 3, 7, len(raw)):
        parser = SSEParser()
        parsed = []
        for start in range(0, len(raw), size):
            parsed.extend(parser.feed(raw[start:start + size]))
        parsed.extend(parser.feed(b"", final=True))
        assert parsed == events
        assert not content_event(parsed[0])[1]
        assert content_event(parsed[1])[1]
        assert not content_event(parsed[2])[1]
        assert content_event(parsed[3]) == (None, False)


def test_stream_error_is_not_content():
    with pytest.raises(ValueError, match="upstream_stream_error"):
        content_event('{"error":{"message":"failed"}}')
