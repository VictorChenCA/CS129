import torch


def main() -> None:
    print(f"torch.__version__ = {torch.__version__}")
    print(f"torch.version.cuda = {torch.version.cuda}")
    print(f"cuda_available = {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        idx = torch.cuda.current_device()
        print(f"cuda_device_index = {idx}")
        print(f"cuda_device_name = {torch.cuda.get_device_name(idx)}")
    else:
        print("cuda_device_name = <none>")


if __name__ == "__main__":
    main()

