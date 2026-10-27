import json

import pytest

from common.preprocessing import UserVocabulary


def test_unseen_user_maps_to_shared_unk():
    vocabulary = UserVocabulary.fit(["alice", "bob", "alice"])
    encoded = vocabulary.encode(["alice", "new-user"])
    assert encoded[0] != 0
    assert encoded[1] == 0


def test_literal_unk_user_cannot_replace_reserved_row():
    vocabulary = UserVocabulary.fit(["<UNK>", "alice", "bob", "<UNK>"])
    encoded = vocabulary.encode(["<UNK>", "alice", "bob", "new-user"])

    assert vocabulary.user_to_id[UserVocabulary.UNK] == 0
    assert len(vocabulary) == 3
    assert encoded[0] == 0
    assert encoded[-1] == 0
    assert all(0 <= int(index) < len(vocabulary) for index in encoded)


@pytest.mark.parametrize(
    "mapping",
    [
        {"alice": 0},
        {"<UNK>": 1, "alice": 0},
        {"<UNK>": 0, "alice": 2},
        {"<UNK>": 0, "alice": 0},
        {"<UNK>": 0, "alice": -1},
    ],
)
def test_loading_malformed_user_vocabulary_fails(tmp_path, mapping):
    path = tmp_path / "users.json"
    path.write_text(json.dumps(mapping), encoding="utf-8")

    with pytest.raises(ValueError, match="User vocabulary|Reserved user token"):
        UserVocabulary.load(path)
