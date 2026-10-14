"""FastAPI service run under gunicorn with several workers and --preload."""
import trainlab_optimize as trainlab

trainlab.instrument(service="acme-web")        # at import: runs in the gunicorn master with --preload

from fastapi import FastAPI, Request  # noqa: E402

from .llm import aoa, h, oa  # noqa: E402

app = FastAPI()


@app.middleware("http")
async def tag_by_route(request: Request, call_next):
    with trainlab.tag(feature=request.headers.get("x-feature", "web"), route=request.url.path):
        return await call_next(request)


@app.post("/sync/{feature}")
def sync_endpoint(feature: str):
    r = oa.chat.completions.create(model="gpt-4o-mini", messages=[{"role": "user", "content": "hi"}],
                                   extra_headers=h(feature))
    return {"ok": r.choices[0].message.content}


@app.post("/async/{feature}")
async def async_endpoint(feature: str):
    r = await aoa.chat.completions.create(model="gpt-4o", messages=[{"role": "user", "content": "hi"}],
                                          extra_headers=h(feature))
    return {"ok": r.choices[0].message.content}
