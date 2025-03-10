import re


regex_mqa_topic_entity = re.compile("\[(.*?)\]")
def extract_predicted_edges(answer_string, group=0):
    """
    Given a string that contains:
    `1. (Albert Einstein; awardReceived; Connects a person to the awards they have received.)`
    Extracts (
        Albert Einstein, 
        honorificAward, 
        Links individuals to awards given in honor of their achievements.
    )
    """
    pattern = re.compile("\d+\.\s*\(([^;]+);\s*([^;]+);\s*(.+?)\)")
    pos = 0
    rels = []
    while m := pattern.search(answer_string, pos):
        pos = m.start() + 1
        entity, rel, reason = m[group].split(";")[:3]
        rels.append(rel.strip())
    return rels
