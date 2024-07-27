import git
import networkx as nx
from collections import defaultdict

def generate_graph_from_tag(repo_path, tag):
    # Initialize the Git repository
    repo = git.Repo(repo_path)

    # Get the commit corresponding to the given tag
    tagged_commit = repo.tags[tag].commit

    # Initialize an empty graph
    graph = nx.Graph()

    # Iterate through the commits
    for commit in repo.iter_commits(rev=tagged_commit):
        # Get the list of files modified in this commit
        files = [item.a_path for item in commit.stats.files]

        # Add nodes and update edges with weights
        for i, file in enumerate(files):
            graph.add_node(file)
            for other_file in files[i + 1:]:
                if file != other_file:
                    if graph.has_edge(file, other_file):
                        # If the edge already exists, increment the weight
                        graph[file][other_file]['weight'] += 1
                    else:
                        # If the edge doesn't exist, add it with a weight of 1
                        graph.add_edge(file, other_file, weight=1)

    return graph


# Usage example
repo_path = '/path/to/your/repo'
tag = 'v1.0'
graph = generate_graph_from_tag(repo_path, tag)


def calculate_commit_count_per_developer(repo_path, tag):
    # Initialize the Git repository
    repo = git.Repo(repo_path)

    # Get the commit corresponding to the given tag
    tagged_commit = repo.tags[tag].commit

    # Dictionary to hold the count of commits per developer for each file
    # Structure: {file: {developer_email: commit_count}}
    file_commit_counts = defaultdict(lambda: defaultdict(int))

    # Iterate through the commits up to the tagged commit
    for commit in repo.iter_commits(rev=tagged_commit):
        # Analy ze files in the commit
        for file in commit.stats.files:
            # Increment the commit count for the author of this commit for the file
            file_commit_counts[file][commit.author.email] += 1

    return file_commit_counts
