from unittest.mock import patch
import numpy as np
import pytest

from datasets.steam import filter_5_core, SteamDataLoader


def test_filter_5_core():
    # User 1 has 5 interactions
    # User 2 has 2 interactions (should be filtered)
    # Item A has 5 interactions
    # Item B has 2 interactions (should be filtered)
    
    interactions = [
        {"user": "u1", "item": "iA"},
        {"user": "u1", "item": "iA"},
        {"user": "u1", "item": "iA"},
        {"user": "u1", "item": "iA"},
        {"user": "u1", "item": "iB"},  # iB will be filtered out because u2 only has 1 other interaction with it, totaling 2 interactions for iB
        {"user": "u2", "item": "iB"},
        {"user": "u2", "item": "iA"},  # u2 only has 2 interactions, so u2 will be filtered out
    ]
    
    # Let's test with k=3 to make it easier to construct a small passing test case
    # User 1 has 5 interactions
    # User 2 has 2 interactions -> u2 gets filtered out.
    # After filtering u2:
    # Interactions are left with: u1:iA (4 times), u1:iB (1 time)
    # Now check item counts: iA has 4 interactions (>=3), iB has 1 interaction (<3) -> iB gets filtered out.
    # Remaining: u1:iA (4 times)
    # Both u1 (4 interactions >=3) and iA (4 interactions >=3) satisfy 3-core.
    filtered = filter_5_core(interactions, k=3)
    assert len(filtered) == 4
    for x in filtered:
        assert x["user"] == "u1"
        assert x["item"] == "iA"


mock_steam_interactions = [
    {"user": "user_1", "item": "game_1", "date": "2026-06-01", "hours": 10.0},
    {"user": "user_1", "item": "game_2", "date": "2026-06-02", "hours": 5.0},
    {"user": "user_1", "item": "game_3", "date": "2026-06-03", "hours": 20.0},
    {"user": "user_1", "item": "game_4", "date": "2026-06-04", "hours": 1.0},
    {"user": "user_1", "item": "game_5", "date": "2026-06-05", "hours": 15.0},
    
    # 5-core requires game_1 to game_5 to have at least 5 reviews. So we replicate them across 5 users.
    {"user": "user_2", "item": "game_1", "date": "2026-06-01", "hours": 1.0},
    {"user": "user_2", "item": "game_2", "date": "2026-06-02", "hours": 1.0},
    {"user": "user_2", "item": "game_3", "date": "2026-06-03", "hours": 1.0},
    {"user": "user_2", "item": "game_4", "date": "2026-06-04", "hours": 1.0},
    {"user": "user_2", "item": "game_5", "date": "2026-06-05", "hours": 1.0},

    {"user": "user_3", "item": "game_1", "date": "2026-06-01", "hours": 1.0},
    {"user": "user_3", "item": "game_2", "date": "2026-06-02", "hours": 1.0},
    {"user": "user_3", "item": "game_3", "date": "2026-06-03", "hours": 1.0},
    {"user": "user_3", "item": "game_4", "date": "2026-06-04", "hours": 1.0},
    {"user": "user_3", "item": "game_5", "date": "2026-06-05", "hours": 1.0},

    {"user": "user_4", "item": "game_1", "date": "2026-06-01", "hours": 1.0},
    {"user": "user_4", "item": "game_2", "date": "2026-06-02", "hours": 1.0},
    {"user": "user_4", "item": "game_3", "date": "2026-06-03", "hours": 1.0},
    {"user": "user_4", "item": "game_4", "date": "2026-06-04", "hours": 1.0},
    {"user": "user_4", "item": "game_5", "date": "2026-06-05", "hours": 1.0},

    {"user": "user_5", "item": "game_1", "date": "2026-06-01", "hours": 1.0},
    {"user": "user_5", "item": "game_2", "date": "2026-06-02", "hours": 1.0},
    {"user": "user_5", "item": "game_3", "date": "2026-06-03", "hours": 1.0},
    {"user": "user_5", "item": "game_4", "date": "2026-06-04", "hours": 1.0},
    {"user": "user_5", "item": "game_5", "date": "2026-06-05", "hours": 1.0},
]
mock_steam_titles = {
    "game_1": "Counter-Strike",
    "game_2": "Dota 2",
    "game_3": "Portal",
    "game_4": "Half-Life",
    "game_5": "Left 4 Dead",
}


@patch("datasets.steam.download_steam_files")
@patch("datasets.steam.parse_steam_reviews")
@patch("datasets.steam.parse_steam_games")
def test_steam_loader(mock_parse_games, mock_parse_reviews, mock_download, tmp_path):
    mock_download.return_value = ("/dummy/reviews.json.gz", "/dummy/games.json.gz")
    mock_parse_reviews.return_value = mock_steam_interactions
    mock_parse_games.return_value = mock_steam_titles

    loader = SteamDataLoader(data_dir=str(tmp_path), load_metadata=True)

    # Check stats
    assert loader.num_users == 5
    assert loader.num_items == 5

    # Check mapping
    assert loader.token_to_title[0] == "<pad>"
    assert loader.token_to_title[1] == "Counter-Strike"

    # Check test split (leave-one-out target is the last item, which is game_5 -> mapped ID 5)
    test_ds = loader.get_split("test", max_len=4, format_type="index")
    inputs, targets = test_ds.to_numpy()
    
    # user_1 history is game_1, game_2, game_3, game_4, game_5 (mapped 1, 2, 3, 4, 5)
    # test input should be [1, 2, 3, 4], target 5
    np.testing.assert_array_equal(inputs[0], np.array([1, 2, 3, 4]))
    assert targets[0] == 5

    # Check text formatting
    text_inputs, text_targets = loader.get_split("test", max_len=4, format_type="text")
    assert text_inputs[0] == ["Counter-Strike", "Dota 2", "Portal", "Half-Life"]
    assert text_targets[0] == "Left 4 Dead"
