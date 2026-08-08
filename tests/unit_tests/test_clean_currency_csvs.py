from src.hedge_forge.scripts.clean_currency_csvs import merge_currency_fields


def test_merge_currency_fields_rejoins_split_thousands_and_currency_values() -> None:
    row = [
        "ACC-001",
        "LOT-017",
        "BAC",
        "060505104",
        "Bank of America Corp.",
        "Equity",
        "Financials",
        "2020-11-09",
        "1",
        "000",
        "$24.00",
        "$24",
        "000.00",
        "",
        "$30.50",
        "$30",
        "500.00",
        "$6",
        "500.00",
        "USD",
    ]

    merged = merge_currency_fields(row)

    assert merged == [
        "ACC-001",
        "LOT-017",
        "BAC",
        "060505104",
        "Bank of America Corp.",
        "Equity",
        "Financials",
        "2020-11-09",
        "1,000",
        "$24.00",
        "$24,000.00",
        "",
        "$30.50",
        "$30,500.00",
        "$6,500.00",
        "USD",
    ]
