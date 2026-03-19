from tokenizer import load_math_dataset

ds = load_math_dataset()

print("ALGEBRA HENDRYCKS DATASET SHAPE INFO: ") # no longer just algebra could be a subset from Hendrycks' MATH
print(f"Num rows: {len(ds)}")

print(f"Colum names: {ds.column_names}")