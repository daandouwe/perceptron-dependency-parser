import numpy as np

"""
Functions to get a Maximum Spanning Tree given a matrix of arc scores.
All this assumes a ROOT token to be at position 0!

Source: https://github.com/tdozat/Parser-v2/blob/master/parser/misc/mst.py
License: Apache 2.0
This code has been modified.
"""


def chu_liu_edmonds(probs):
  """The CLE algorithm"""

  vertices = np.arange(len(probs))
  edges = np.argmax(probs, axis=1)
  cycles = find_cycles(edges)

  if cycles:

    # print("found cycle, fixing...")
    cycle_vertices = cycles.pop()                             # (c)
    non_cycle_vertices = np.delete(vertices, cycle_vertices)  # (nc)
    cycle_edges = edges[cycle_vertices]                       # (c)

    # get rid of cycle nodes
    non_cycle_probs = np.array(
      probs[non_cycle_vertices, :][:, non_cycle_vertices])    # (nc x nc)

    # add a node representing the cycle
    # (nc+1 x nc+1)
    non_cycle_probs = np.pad(non_cycle_probs, [[0, 1], [0, 1]], 'constant')

    # probabilities of heads outside the cycle
    # (c x nc) / (c x 1) = (c x nc)
    backoff_cycle_probs = probs[cycle_vertices][:, non_cycle_vertices] / \
                          probs[cycle_vertices, cycle_edges][:, None]

    # probability of a node inside the cycle depending on
    # something outside the cycle
    # max_0(c x nc) = (nc)
    non_cycle_probs[-1, :-1] = np.max(backoff_cycle_probs, axis=0)

    # probability of a node outside the cycle depending on
    # something inside the cycle
    # max_1(nc x c) = (nc)
    non_cycle_probs[:-1, -1] = np.max(
      probs[non_cycle_vertices][:, cycle_vertices], axis=1)

    # (nc+1)
    non_cycle_edges = chu_liu_edmonds(non_cycle_probs)

    # This is the best source vertex into the cycle
    non_cycle_root, non_cycle_edges = non_cycle_edges[-1], non_cycle_edges[:-1]  # in (nc)
    source_vertex = non_cycle_vertices[non_cycle_root]  # in (v)

    # This is the vertex in the cycle we want to change
    cycle_root = np.argmax(backoff_cycle_probs[:, non_cycle_root])  # in (c)
    target_vertex = cycle_vertices[cycle_root]  # in (v)
    edges[target_vertex] = source_vertex

    # update edges with any other changes
    mask = np.where(non_cycle_edges < len(non_cycle_probs) - 1)
    edges[non_cycle_vertices[mask]] = non_cycle_vertices[non_cycle_edges[mask]]
    mask = np.where(non_cycle_edges == len(non_cycle_probs) - 1)

    # FIX
    stuff = np.argmax(probs[non_cycle_vertices][:, cycle_vertices], axis=1)
    stuff2 = cycle_vertices[stuff]
    stuff3 = non_cycle_vertices[mask]
    edges[stuff3] = stuff2[mask]

  return edges


def greedy(probs):
  """
  A simpler alternative to CLE algorithm.
  Might give different performance.
  """
  edges = np.argmax(probs, axis=1)
  cycles = True

  while cycles:
    cycles = find_cycles(edges)
    for cycle_vertices in cycles:

      # Get the best heads and their probabilities
      cycle_edges = edges[cycle_vertices]
      cycle_probs = probs[cycle_vertices, cycle_edges]

      # Get the second-best edges and their probabilities
      probs[cycle_vertices, cycle_edges] = 0
      backoff_edges = np.argmax(probs[cycle_vertices], axis=1)
      backoff_probs = probs[cycle_vertices, backoff_edges]
      probs[cycle_vertices, cycle_edges] = cycle_probs

      # Find the node in the cycle that the model is the
      # least confident about and its probability
      new_root_in_cycle = np.argmax(backoff_probs/cycle_probs)
      new_cycle_root = cycle_vertices[new_root_in_cycle]

      # Set the new root
      probs[new_cycle_root, cycle_edges[new_root_in_cycle]] = 0
      edges[new_cycle_root] = backoff_edges[new_root_in_cycle]

  return edges


def find_roots(edges):
  """Return a list of vertices that were considered root by a dependent."""
  return np.where(edges[1:] == 0)[0] + 1


def softmax(x):
  """Numpy softmax function (normalizes rows)."""
  x -= np.max(x, axis=1, keepdims=True)
  x = np.exp(x)
  return x / np.sum(x, axis=1, keepdims=True)


def make_root(probs, root, eta=1e-9):
  """Make specified vertex (index) root and nothing else."""
  probs = np.array(probs)
  probs[1:, 0] = 0
  probs[root, :] = 0
  probs[root, 0] = 1
  probs /= np.sum(probs + eta, axis=1, keepdims=True)
  return probs


def score_edges(probs, edges, eta=1e-9):
  """score a graph (so we can choose the best one)"""
  return np.sum(np.log(probs[np.arange(1, len(probs)), edges[1:]] + eta))


def get_best_graph(probs):
  """
  Returns the best graph, applying the CLE algorithm and making sure
  there is only a single root.
  """

  # zero out the diagonal (no word can be its own head)
  probs *= 1 - np.eye(len(probs)).astype(np.float32)
  probs[0] = 0     # zero out first row (root points to nothing else)
  probs[0, 0] = 1  # root points to itself
  probs /= np.sum(probs, axis=1, keepdims=True)  # normalize

  # apply CLE algorithm
  # edges = chu_liu_edmonds(probs)
  edges = greedy(probs)

  # deal with multiple roots
  roots = find_roots(edges)
  best_edges = edges
  best_score = -np.inf
  if len(roots) > 1:
    # print("more than 1 root!", roots)
    for root in roots:
      # apply CLE again with each of the possible roots fixed as the root
      # we return the highest scoring graph
      probs_ = make_root(probs, root)
      # edges_ = chu_liu_edmonds(probs_)
      edges_ = greedy(probs_)
      score = score_edges(probs_, edges_)
      if score > best_score:
        best_edges = edges_
        best_score = score

  return best_edges


def find_cycles(edges):
  """
  Finds cycles in a graph. Returns empty list if no cycles exist.
  Cf. https://en.wikipedia.org/wiki/Tarjan%27s_strongly_connected_components_algorithm
  """

  vertices = np.arange(len(edges))
  indices = np.zeros_like(vertices) - 1
  lowlinks = np.zeros_like(vertices) - 1
  stack = []
  onstack = np.zeros_like(vertices, dtype=np.bool)
  current_index = 0
  cycles = []

  def _strong_connect(vertex, current_index):
    indices[vertex] = current_index
    lowlinks[vertex] = current_index
    stack.append(vertex)
    current_index += 1
    onstack[vertex] = True

    for vertex_ in np.where(edges == vertex)[0]:
      if indices[vertex_] == -1:
        current_index = _strong_connect(vertex_, current_index)
        lowlinks[vertex] = min(lowlinks[vertex], lowlinks[vertex_])
      elif onstack[vertex_]:
        lowlinks[vertex] = min(lowlinks[vertex], indices[vertex_])

    if lowlinks[vertex] == indices[vertex]:
      cycle = []
      vertex_ = -1
      while vertex_ != vertex:
        vertex_ = stack.pop()
        onstack[vertex_] = False
        cycle.append(vertex_)
      if len(cycle) > 1:
        cycles.append(np.array(cycle))
    return current_index

  for vertex in vertices:
    if indices[vertex] == -1:
      current_index = _strong_connect(vertex, current_index)

  return cycles


def test():
  """test out MST"""
  np.random.seed(6)

  n = 20
  probs = np.random.randint(0, 99, [n, n])
  # probs = probs * (1-np.eye(n, dtype=np.int64))
  print(probs)
  probs = softmax(probs)
  greedy = probs.argmax(axis=1)

  edges = get_best_graph(probs)

  print("greedy edges:", greedy)
  print("CLE edges:   ", edges)
  print("nodes:       ", np.arange(n))


if __name__ == '__main__':
  test()
