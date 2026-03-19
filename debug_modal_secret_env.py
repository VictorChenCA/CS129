import os
import json
import modal

app = modal.App('debug-secret-env')
img = modal.Image.debian_slim().pip_install('google-cloud-storage')

@app.function(image=img, secrets=[modal.Secret.from_name('googlecloud-secret'), modal.Secret.from_name('joseph-cs224n-project')])
def f():
    keys = sorted([k for k in os.environ.keys() if any(x in k for x in ['SERVICE','ACCOUNT','JSON','GOOGLE'])])
    print(keys)
    for k in keys:
        v = os.environ.get(k,'')
        if len(v) > 0 and v.strip().startswith('{'):
            try:
                d = json.loads(v)
                print(k, 'json client_email=', d.get('client_email'))
            except Exception:
                print(k, 'non-json-brace')
        else:
            if k == 'GOOGLE_APPLICATION_CREDENTIALS':
                print(k, v)

@app.local_entrypoint()
def main():
    f.remote()
