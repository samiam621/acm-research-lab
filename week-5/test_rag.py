#checks retrieval only, no API call
from rag import retrieve

assert retrieve("Can a person step through every calculation by hand?")[0][1].startswith("Simulatability")
assert retrieve("showing similar training cases like a doctor")[0][1].startswith("Explanation by example")
print("ok")
