def train_test_split(df, split=0.8):
    size = len(df)
    train_df = df[: int(size * 0.8)]
    val_df = df[int(size * 0.8) :]

    train_df.reset_index(drop=True, inplace=True)
    val_df.reset_index(drop=True, inplace=True)

    return train_df, val_df
