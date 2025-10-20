import requests


class GraphDBError(Exception):
    pass


class KnowledgeGraph:
    def __init__(self, graphdb_url: str):
        self.graphdb_url = graphdb_url

    def create_repository(self, repo_id, ruleset="rdfs", storage_folder="storage"):
        """
        Create a new repository in GraphDB.

        Parameters:
        - repo_id: str, unique identifier for the repository
        - ruleset: str, ruleset to use (default is "rdfs")
        - storage_folder: str, storage folder for the repository (default is "storage")
        """
        url = f"{self.graphdb_url}/rest/repositories"
        config = {
            "id": repo_id,
            "title": repo_id,
            "type": "free",
            "params": {
                "ruleset": ruleset,
                "storage-folder": storage_folder
            }
        }

        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, headers=headers, json=config)
        except requests.exceptions.ConnectionError:
            raise GraphDBError(f"Failed to connect to GraphDB."
                               f" Is the server running at {self.graphdb_url}?")

        if response.status_code == 201:
            print(f"Repository '{repo_id}' created successfully!")
        elif response.status_code == 409:
            print(f"Repository '{repo_id}' already exists.")
        else:
            print(f"Failed to create repository: {response.status_code} - {response.text}")

    def run(self):
        pass   


def test_engine():
    engine = KnowledgeGraph("http://localhost:7200")
    try:
        engine.create_repository("test_repo")
    except GraphDBError as e:
        print(e)

if __name__ == "__main__":
    test_engine()
