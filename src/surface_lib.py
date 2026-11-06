# ABOUTME: Organically surfaces candidate idea-neighborhoods from the Obsidian zettel graph.
# ABOUTME: Combines embedding proximity, graph distance, and source-disjointness. No keyword matching.
import os, re, glob, json, collections, argparse
import numpy as np, chromadb

VAULT = os.environ.get("ZK_VAULT", "/path/to/your/obsidian/vault")
ZDIR = f"{VAULT}/20_ZETTLEKASTEN"
DB = "chromastores/zettledb"
COLL = "zettels__sentence_transformer__all_minilm_l6_v2"

def norm(s): return s.strip().lower().replace(" ", "-")

def tension_links(raw):
    m = re.search(r"## Tensions(.*?)(?=\n# References|\Z)", raw, re.S)
    if not m: return set()
    return {norm(x) for x in re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", m.group(1))}


def load_graph():
    """Body wikilinks (Tensions), source refs per zettel. Ignores frontmatter Tags."""
    adj = collections.defaultdict(set); src = {}; text = {}
    for f in glob.glob(f"{ZDIR}/*.md"):
        n = norm(os.path.basename(f)[:-3]); raw = open(f).read()
        lines = raw.split("\n")
        body = "\n".join(lines[4:])
        text[n] = raw
        refs = re.search(r"# References(.*)$", raw, re.S)
        src[n] = {norm(x) for x in re.findall(r"\[\[([^\]|]+)\]\]", refs.group(1))} if refs else set()
        ref_span = refs.group(1) if refs else ""
        body_only = body.replace(ref_span, "")
        for l in re.findall(r"\[\[([^\]|]+)(?:\|[^\]]*)?\]\]", body_only):
            l = norm(l)
            if l != n: adj[n].add(l); adj[l].add(n)
    return adj, src, text

def hops(adj, a, b, maxd=3):
    if b in adj.get(a, ()): return 1
    seen = {a}; frontier = [a]
    for d in range(1, maxd + 1):
        nxt = []
        for u in frontier:
            for v in adj.get(u, ()):
                if v == b: return d
                if v not in seen: seen.add(v); nxt.append(v)
        frontier = nxt
        if not frontier: break
    return 99

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-sim", type=float, default=0.60)
    ap.add_argument("--min-hops", type=int, default=3)
    ap.add_argument("--knn", type=int, default=8)
    ap.add_argument("--min-size", type=int, default=2)
    ap.add_argument("--min-sources", type=int, default=0, help="0 = allow uncited notes")
    ap.add_argument("--require-disjoint", action="store_true", help="only pairs whose source papers share nothing")
    ap.add_argument("--quality", action="store_true", help="only current-template zettels containing numerals")
    ap.add_argument("--max-size", type=int, default=8)
    ap.add_argument("--scope", type=str, default=None, help="literal substring match on file contents (e.g. a Created: date). Blunt \u2014 prefer --about")
    ap.add_argument("--about", type=str, default=None, help="describe the problem space in a sentence; restricts to the semantically nearest zettels. No keywords.")
    ap.add_argument("--about-k", type=int, default=80, help="how many zettels --about retains")
    ap.add_argument("--limit", type=int, default=12)
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--min-docs", type=int, default=0, help="require N distinct SOURCE DOCUMENTS (guards against one doc shredded into many zettels)")
    ap.add_argument("--require-tension", action="store_true", help="require >=1 intra-cluster link inside a ## Tensions section (density without variance is redundancy)")
    ap.add_argument("--seen", type=str, default=None, help="path to a seen-clusters file; skips groups overlapping >=50% with any prior group and appends new ones")
    a = ap.parse_args()

    col = chromadb.PersistentClient(path=DB).get_collection(COLL)
    got = col.get(include=["embeddings"])
    ids = [norm(x) for x in got["ids"]]
    E = np.array(got["embeddings"], dtype=np.float32)
    E /= np.linalg.norm(E, axis=1, keepdims=True)
    adj, src, text = load_graph()
    qual = None
    if a.quality:
        qual = {z for z, t in text.items() if "## Tensions" in t and re.search(r"\d", t)}
        if not a.json: print(f"quality filter -> {len(qual)} zettels")
    about = None
    if a.about:
        r = col.query(query_texts=[a.about], n_results=min(a.about_k, len(ids)), include=["distances"])
        about = {norm(x) for x in r["ids"][0]}
        if not a.json: print(f"about: '{a.about[:60]}' -> {len(about)} zettels")
    scope = None
    if a.scope:
        scope = {z for z, t in text.items() if a.scope in t}
        if not a.json: print(f"scope '{a.scope}' -> {len(scope)} zettels")

    # candidate edges: semantically near, graph-far, source-disjoint
    S = E @ E.T; np.fill_diagonal(S, -1)
    cand = collections.defaultdict(set); w = {}
    for i, zi in enumerate(ids):
        for j in np.argpartition(-S[i], a.knn)[: a.knn]:
            j = int(j)
            if j <= i: continue
            zj = ids[j]; sim = float(S[i, j])
            if scope is not None and (zi not in scope and zj not in scope): continue
            if about is not None and (zi not in about or zj not in about): continue
            if sim < a.min_sim: continue
            if qual is not None and (zi not in qual or zj not in qual): continue
            si, sj = src.get(zi, set()), src.get(zj, set())
            if a.require_disjoint:
                if not si or not sj or (si & sj): continue
            elif si and sj and (si & sj): continue
            if hops(adj, zi, zj) <= a.min_hops: continue
            cand[zi].add(zj); cand[zj].add(zi); w[(zi, zj)] = sim

    # connected components over the candidate edges = neighborhoods
    seen = set(); groups = []
    for z in cand:
        if z in seen: continue
        comp = set(); stack = [z]
        while stack:
            u = stack.pop()
            if u in comp: continue
            comp.add(u); seen.add(u)
            stack.extend(cand[u] - comp)
        if len(comp) < a.min_size: continue
        if len(comp) <= a.max_size: groups.append(comp); continue
        # too big: keep the densest core by edge weight
        inner = sorted(((w[k], k) for k in w if k[0] in comp and k[1] in comp), reverse=True)
        core = set()
        for sim_, (u, v) in inner:
            if len(core | {u, v}) > a.max_size: continue
            core |= {u, v}
        if len(core) >= a.min_size: groups.append(core)

    scored = []
    for g in groups:
        g = set(g)
        srcs = set()
        for z in g: srcs |= src.get(z, set())
        if a.min_sources and len(srcs) < a.min_sources: continue
        if a.min_docs and len(srcs) < a.min_docs: continue
        n_tension = sum(1 for z in g if tension_links(text.get(z, "")) & (g - {z}))
        if a.require_tension and n_tension == 0: continue
        sims = [w[k] for k in w if k[0] in g and k[1] in g]
        cohesion = float(np.mean(sims)) if sims else 0
        # score: many distinct sources, tight semantics, compact group
        score = max(len(srcs), 1) * cohesion / max(1, len(g) ** 0.5)
        scored.append((score, cohesion, sorted(g), sorted(srcs), n_tension))
    scored.sort(key=lambda x: -x[0])

    if a.seen:
        prior = []
        if os.path.exists(a.seen):
            for line in open(a.seen):
                line = line.strip()
                if line: prior.append(set(line.split("|")))
        kept, fresh = [], []
        for rec in scored:
            g = set(rec[2])
            dup = any(len(g & p) / max(1, min(len(g), len(p))) >= 0.5 for p in prior)
            if dup: continue
            kept.append(rec); prior.append(g); fresh.append(g)
        skipped = len(scored) - len(kept)
        scored = kept
        with open(a.seen, "a") as fh:
            for g in fresh: fh.write("|".join(sorted(g)) + "\n")
        if not a.json: print(f"dedup: skipped {skipped} previously-seen group(s)\n")

    if a.json:
        out=[]
        for s_, c, g, sr, nt in scored[: a.limit]:
            out.append({"score": round(s_,3), "cohesion": round(c,3), "zettels": g,
                        "sources": sr, "tension_links": nt,
                        "bodies": {z: text.get(z,"")[:2600] for z in g}})
        print(json.dumps(out, indent=1))
        return
    print(f"{len(ids)} zettels | {sum(len(v) for v in cand.values())//2} candidate edges | {len(groups)} groups | {len(scored)} qualifying\n")
    for s, c, g, sr, nt in scored[: a.limit]:
        flag = f" | {nt} tension-link(s)" if nt else " | NO intra-cluster tension (redundancy risk)"
        print(f"### score={s:.2f} cohesion={c:.3f} | {len(g)} zettels, {len(sr)} distinct sources{flag}")
        for z in g: print(f"    - {z}")
        print(f"    sources: {', '.join(x[:44] for x in sr[:6])}{' ...' if len(sr)>6 else ''}\n")

if __name__ == "__main__":
    main()
