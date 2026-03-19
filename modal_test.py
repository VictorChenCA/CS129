
import modal

app = modal.App("my-app")

# # Install runtime deps into the Modal container.
# image = modal.Image.debian_slim(python_version="3.11").pip_install("torch")


def preprocess(x: int) -> int:
    return x * 2


@app.function(gpu="L4")
def train() -> int:
    x = preprocess(5)
    # Remote logs should show up in `modal run`, but return value ensures local visibility too.
    print(x, flush=True)
    return x


@app.local_entrypoint()
def main() -> None:
    # If you don't see this line, you're not actually running the entrypoint via `modal run ...`.
    print("local: launching remote train()", flush=True)

    # Force streaming output + block until we have the return value.
    # Using spawn/get is more compatible than relying on remote() return semantics.
    with modal.enable_output():
        call = train.spawn()
        x = call.get()

    print(f"local: remote returned: {x}", flush=True)
