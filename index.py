from workers import WorkerEntrypoint, Response
import math


def run_simulation() -> dict:
    """
    Replace this function with the real logic from your sim.py.
    Keep it lightweight enough for a Cloudflare Worker request.
    """
    xs = [i / 10 for i in range(11)]
    ys = [math.sin(x) for x in xs]
    return {
        "message": "Starter Cloudflare Python Worker is running.",
        "note": "Replace run_simulation() with your actual sim.py logic.",
        "x": xs,
        "y": ys,
    }


class Default(WorkerEntrypoint):
    async def fetch(self, request):
        result = run_simulation()
        return Response.json(result)
