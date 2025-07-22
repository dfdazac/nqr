import jinja2
from SPARQLWrapper import JSON, SPARQLWrapper, CSV
import socket
import urllib.error
import pandas as pd


class GraphDatabase:
    def __init__(self, graphdb_endpoint: str, tasks: list[str]):
        self.graphdb_endpoint = graphdb_endpoint
        env = jinja2.Environment(loader=jinja2.PackageLoader("quack",
                                                             "queries"))
        self.task_templates = {task: env.get_template(f"{task}.jinja") for task in tasks}
        self.wrapper = SPARQLWrapper(graphdb_endpoint)
        self.wrapper.setReturnFormat(JSON)

    @staticmethod
    def _extract_ids(val):
        """
        Extracts the numeric ID from a KG entity or relation IRI of the form:
        'http://example.org/Q123' or 'http://example.org/P456'.
        """
        return int(val.split('/')[-1][1:])

    def run_query(self, task, splits, flat_ids, extract_ids=True):
        query = self.task_templates[task].render({
            "splits": splits,
            "flat_ids": flat_ids
        })

        try:
            self.wrapper.setQuery(query)
            result = self.wrapper.query().convert()

            df = pd.DataFrame(result['results']['bindings'])
            df = df.map(lambda x: x['value'])

            if extract_ids:
                df = df.map(self._extract_ids)

            return df
        except (urllib.error.URLError, ConnectionRefusedError, socket.timeout, socket.error) as e:
            raise ConnectionError(f"Connection failed: {e}") from e

