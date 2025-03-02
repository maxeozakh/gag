from datasets import load_dataset # type: ignore

dataset = load_dataset("rajuptvs/ecommerce_products_clip")

dataset["train"].remove_columns(["image"]).to_csv("ecommerce_products.csv")