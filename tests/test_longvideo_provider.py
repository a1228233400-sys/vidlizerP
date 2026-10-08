from vidlizer.longvideo.provider import ProviderClient


def test_provider_json_parser_accepts_fenced_json():
    result = ProviderClient._parse_json(
        chr(96) * 3 + "json\n{\"answer\":\"ok\"}\n" + chr(96) * 3
    )
    assert result == {"answer": "ok"}
