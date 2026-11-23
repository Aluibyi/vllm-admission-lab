"""Incremental UTF-8 SSE parser, independent of HTTP chunk boundaries."""
import codecs
import json


class SSEParser:
    def __init__(self):
        self.decoder = codecs.getincrementaldecoder("utf-8")()
        self.buffer = ""
        self.lines = []

    def feed(self, chunk: bytes, final=False):
        self.buffer += self.decoder.decode(chunk, final=final)
        events = []
        while "\n" in self.buffer:
            line, self.buffer = self.buffer.split("\n", 1)
            line = line.removesuffix("\r")
            if not line:
                event = self._event()
                if event is not None:
                    events.append(event)
            else:
                self.lines.append(line)
        if final:
            if self.buffer:
                self.lines.append(self.buffer.removesuffix("\r"))
                self.buffer = ""
            event = self._event()
            if event is not None:
                events.append(event)
        return events

    def _event(self):
        data = []
        for line in self.lines:
            if line == "data":
                data.append("")
            elif line.startswith("data:"):
                value = line[5:]
                data.append(value[1:] if value.startswith(" ") else value)
        self.lines.clear()
        return "\n".join(data) if data else None


def content_event(event: str):
    if event == "[DONE]":
        return None, False
    value = json.loads(event)
    if "error" in value:
        raise ValueError(f"upstream_stream_error: {value['error']}")
    has_content = any(c.get("text") or c.get("token_ids") or
                      c.get("delta", {}).get("content")
                      for c in value.get("choices", []))
    return value, bool(has_content)
