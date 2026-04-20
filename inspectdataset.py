from datasets import load_dataset


def preview_recipes(dataset_name: str = "corbt/all-recipes", rows: int = 10, max_chars: int = 300) -> None:
    ds = load_dataset(dataset_name, split=f"train[:{rows}]")
    for row in ds:
        print(row["input"][:max_chars])
    print("End of preview.")


if __name__ == "__main__":
    preview_recipes()