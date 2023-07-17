import pandas as pd
from data_processing import load_data, sort_train_data


def test_load_data():
    # Test that the function loads the data correctly
    path = "test_data.feather"
    test_data = pd.DataFrame({"A": [1, 2, 3], "B": [4, 5, 6]})
    test_data.to_feather(path)
    result = load_data(path)
    pd.testing.assert_frame_equal(result, test_data)


def test_sort_train_data():
    # Test that the function sorts the train data correctly
    train_data = pd.DataFrame(
        {
            "customer_ID": [1, 2, 1, 3],
            "S_2": ["2021-01-01", "2021-01-02", "2021-01-03", "2021-01-01"],
        }
    )
    expected_result = pd.DataFrame(
        {
            "customer_ID": [1, 1, 2, 3],
            "S_2": ["2021-01-01", "2021-01-03", "2021-01-02", "2021-01-01"],
        }
    )
    expected_result["S_2"] = pd.to_datetime(expected_result["S_2"])
    result = sort_train_data(train_data)
    # Reset the index of the data frames before comparing them
    result = result.reset_index(drop=True)
    expected_result = expected_result.reset_index(drop=True)

    pd.testing.assert_frame_equal(result, expected_result)
