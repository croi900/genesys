from dask.distributed import Client, LocalCluster


class DaskPool:
    def __init__(self, client):
        self.client = client

    def map(self, func, iterable):
        futures = self.client.map(func, iterable)
        results = self.client.gather(futures)
        return results

    def close(self):
        self.client.close()

    def join(self):
        pass

    def terminate(self):
        self.client.close()
