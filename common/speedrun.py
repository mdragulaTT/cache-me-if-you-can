"""Act IV, the Speedrun: the race, the measuring tools and the texts.

The race reads each text token by token. The first SPEEDRUN_PROMPT_TOKENS go in at once through
`prefill` (not timed); the next SPEEDRUN_RACE_TOKENS go in one at a time through `decode` (timed).
After every token the generator returns its prediction of the next one. That is exactly the work
of generating text, but because the text is fed from outside, nobody can compute the answers ahead
of time. Score: median milliseconds per decode step. A run counts only if the predictions are about
as good as the original model's (see config.SPEEDRUN_QUALITY_TOLERANCE).
"""
import os
import statistics
import time

import ttnn

from . import config
from . import gpt2_reference as ref

TEXTS = [
    "The river town woke slowly. Before the first bus left the square, the baker had already opened the shutters, "
    "and the smell of bread drifted down to the water, where two fishermen argued about the weather in low voices. "
    "Nobody in the town remembered when the bridge had been built, only that it had been repaired after every flood, "
    "each time with stones of a slightly different colour, so that it now looked like a patchwork quilt laid across "
    "the river. Children crossed it on their way to school, counting the colours, and the old men on the benches "
    "counted the children. In the summer, tourists came for a day, took photographs of the bridge and the church, "
    "ate ice cream on the steps of the town hall and left before evening, when the swallows came out and the town "
    "returned to itself. The mayor, a retired teacher, kept a notebook of every flood since 1950. She liked to say "
    "that the river was the oldest citizen and the only one who never voted, and that it was wise to listen to it "
    "anyway. Last spring the water rose higher than anyone had seen, and for three nights the whole town carried "
    "sandbags, students next to farmers, the priest next to the owner of the only bar. When the water finally went "
    "down, the bridge was still standing, and someone painted a small blue line on its pillar to mark the new record.",

    "Printing changed the speed at which ideas could travel. Before movable type, a book was copied by hand, and a "
    "skilled scribe might finish a few pages a day. Errors crept in with every copy, and two copies of the same text "
    "were rarely identical. With a press, a workshop could produce hundreds of identical pages in the time a scribe "
    "needed for one, and the cost of a book fell so far that merchants, students and craftsmen could own them. "
    "Printers did not only copy old texts. They printed calendars, maps, price lists, pamphlets and arguments, and "
    "very quickly the arguments became the most popular product. A pamphlet could be written in the morning, set in "
    "type in the afternoon and sold on the street the next day. Governments noticed, and soon most countries required "
    "a licence to print. Historians still debate how much the press caused the great changes of the following century "
    "and how much it simply made them visible. What is clear is that once information became cheap to copy, the "
    "scarce resource was no longer the copy itself but the attention of the reader, and printers began to compete "
    "with titles, pictures and promises on the first page, much as newspapers and websites do today. The same pattern "
    "repeated with the telegraph, the radio and the internet: each new way of copying made the old bottleneck "
    "disappear and revealed a new one, and each time the people who understood the new bottleneck first were the "
    "ones who shaped what came next.",

    "To make a good soup on a cold evening you need patience more than skill. Start by cutting two onions, a carrot "
    "and a stick of celery into small pieces, and let them cook slowly in butter until the kitchen smells sweet. "
    "Do not hurry this step; it is where most of the flavour comes from. Add a few cloves of garlic, a spoon of "
    "tomato paste and a pinch of paprika, and stir for another minute. Then pour in the stock, add the potatoes and "
    "the beans, and let everything simmer under a lid. While the soup cooks, there is time to set the table, slice "
    "the bread and call everyone who promised to help. After half an hour, taste it. It will almost always need more "
    "salt than you expect, and often a little acid, such as a splash of vinegar or a squeeze of lemon, to wake it up. "
    "Some people blend half of the soup to make it creamy and leave the rest in pieces. Others insist that a soup "
    "should never be blended at all. Both camps agree on one thing: it tastes better the next day, after the flavours "
    "have had a night to get to know each other, which is why a wise cook always makes twice as much as needed. If "
    "there is any left on the third day, it can become a sauce for pasta, a filling for pies or the start of an "
    "entirely new soup, and in some families a pot of soup has been passed from one week to the next for so long "
    "that nobody remembers what the first one was.",

    "The night train left the capital a few minutes after ten. In the first carriage a group of students spread a map "
    "across two seats and argued about which mountain they would climb first, while an elderly couple across the aisle "
    "pretended not to listen and occasionally corrected the pronunciation of a village name. The conductor knew most "
    "of the regular passengers by sight. There was the nurse who travelled home every Friday, the musician who always "
    "carried his cello in a seat of its own, and the man who never slept and spent the whole journey writing letters "
    "by hand. Around midnight the lights were dimmed and the carriage grew quiet, except for the rhythm of the wheels "
    "and the occasional bell of a crossing in the dark. At two in the morning the train stopped for twenty minutes at "
    "a border station, where a sleepy officer walked through the carriages with a torch and checked the documents of "
    "anyone who was awake. By dawn the flat fields had turned into hills, the hills into forests, and the students "
    "were pressed against the windows, pointing at the first snow on the peaks. The musician played a few quiet notes "
    "to check whether the cold had put his instrument out of tune, and for a moment the whole carriage listened. "
    "When the train finally arrived, an hour late as usual, nobody seemed to mind, and the conductor wished every "
    "passenger a good holiday as they stepped down onto the frozen platform.",

    "A beehive is a small city that runs without a mayor. The queen lays the eggs, but she does not give orders; the "
    "decisions are made by thousands of workers, each following a few simple rules. Young workers clean the cells and "
    "feed the larvae, older ones build wax, guard the entrance or fly out to collect nectar and pollen. When a forager "
    "finds a rich field of flowers, she returns and performs a dance on the comb. The direction of the dance tells the "
    "others where the flowers are relative to the sun, and its length tells them how far away they are. Bees that "
    "follow the dance fly out, and if they agree that the field is good, they come back and dance too, so that good "
    "sources quickly attract more and more workers while poor ones are forgotten. The same kind of vote is used when "
    "a colony grows too large and part of it leaves to find a new home. Scouts visit hollow trees, chimneys and empty "
    "boxes, and each returns to advertise the place she found. Over a day or two the dances for the best site grow "
    "more numerous, until enough scouts agree and the whole swarm lifts into the air and flies there together. "
    "Biologists have studied these decisions for decades, and engineers now borrow the same ideas to coordinate "
    "fleets of robots and to route traffic in computer networks, where no single machine sees the whole picture.",
]

_reference = {}


class _Reference:
    """The starting point of the race: Act II's attention and KV cache, unchanged."""

    def __init__(self, model, device):
        self.model = model

    def prefill(self, ids):
        self.cache = ref.KVCache(self.model.n_layers)
        return int(ref.forward_with_cache(self.model, self.cache, ids).argmax())

    def decode(self, token, pos):
        return int(ref.forward_with_cache(self.model, self.cache, [token]).argmax())


def _run(make_generator, model, device, texts):
    P, R = config.SPEEDRUN_PROMPT_TOKENS, config.SPEEDRUN_RACE_TOKENS
    gen = make_generator(model, device)                                   # setup: not timed
    step_ms, hits, n = [], 0, 0
    try:
        for text in texts:
            ids = model.tokenizer.encode(text)
            assert len(ids) > P + R, f"a race text needs more than {P + R} tokens, this one has {len(ids)}"
            hits += int(gen.prefill(ids[:P])) == ids[P]                   # not timed
            n += 1
            for pos in range(P, P + R):
                t0 = time.perf_counter()
                prediction = int(gen.decode(ids[pos], pos))
                step_ms.append((time.perf_counter() - t0) * 1e3)
                hits += prediction == ids[pos + 1]
                n += 1
    finally:
        if hasattr(gen, "close"):
            gen.close()
        _release_leftover_traces(gen, device)
    return statistics.median(step_ms), 100.0 * hits / n


def _release_leftover_traces(gen, device):
    """Free any recording the generator still holds, so forgotten close() calls don't fill the trace region."""
    for name, value in list(vars(gen).items()):
        if isinstance(value, ttnn.MeshTraceId):
            try:
                ttnn.release_trace(device, value)
            except Exception:
                pass                                                      # already released by close()
            setattr(gen, name, None)


def reference(model, device, texts=None):
    """(ms per token, accuracy %) of the starting point on these texts. Computed once, then remembered."""
    texts = tuple(texts or TEXTS)
    if texts not in _reference:
        _reference[texts] = _run(_Reference, model, device, texts)
    return _reference[texts]


def race(make_generator, model, device, texts=None, verbose=True):
    """Race a generator. `make_generator(model, device)` builds it (not timed); it needs
    prefill(token_ids) -> next token and decode(token, position) -> next token, and may have close()."""
    texts = list(texts or TEXTS)
    base_ms, base_acc = reference(model, device, texts)
    try:
        ms, acc = _run(make_generator, model, device, texts)
    except RuntimeError as e:
        if "trace capture" in str(e):
            print("Something failed while the step was being recorded, and the card is still in recording mode.\n"
                  "Restart the kernel and run the notebook from the top. Before trying again, check the trace\n"
                  "rules: run step() once before recording it, and no copies to or from the host inside step().")
        raise
    grid = device.compute_with_storage_grid_size()
    card = f"{device.arch().name} with {grid.x * grid.y} cores"   # so results from different hardware stand out
    too_good = acc > base_acc + 10              # far better than GPT-2 itself: not computed by the model
    valid = base_acc - config.SPEEDRUN_QUALITY_TOLERANCE <= acc and not too_good
    result = dict(ms_per_token=round(ms, 3), tokens_per_s=round(1000 / ms, 1), speedup=round(base_ms / ms, 2),
                  accuracy=round(acc, 1), reference_accuracy=round(base_acc, 1), valid=valid, card=card)
    if verbose:
        name = getattr(make_generator, "__name__", "generator")
        ratio = base_ms / ms
        compared = "about the same as the starting point" if 0.9 <= ratio <= 1.1 else f"{ratio:.1f}x the starting point"
        print(f"{name}: {ms:.2f} ms per token ({1000 / ms:.0f} tokens/s), {compared}, on {card}")
        print(f"  predicts the real next token {acc:.1f}% of the time (original GPT-2: {base_acc:.1f}%)")
        if valid:
            print("  VALID")
        elif too_good:
            print("  DOES NOT COUNT: far better than the original model, so the predictions cannot be GPT-2's")
        else:
            print(f"  DOES NOT COUNT: more than {config.SPEEDRUN_QUALITY_TOLERANCE:.0f} points below the original")
    return result


def submit(result, nickname, what_you_changed):
    """Send a valid race result to the results form, which feeds the leaderboard on the projector."""
    import urllib.parse
    import urllib.request
    if not result.get("valid"):
        print("This run does not count, so it was not sent.")
        return False
    if nickname.strip().lower() in ("", "your nickname"):
        print("Put your nickname in first.")
        return False
    if os.environ.get("MLINPL_NO_SUBMIT"):                                # automated test runs
        print(f"(test run: {result['ms_per_token']:.2f} ms per token not sent)")
        return False
    fields = config.SPEEDRUN_FORM_FIELDS
    data = urllib.parse.urlencode({fields["nickname"]: nickname, fields["ms_per_token"]: f"{result['ms_per_token']:.2f}",
                                   fields["what_you_changed"]: f"{what_you_changed} [{result.get('card', '?')}]"}).encode()
    try:
        with urllib.request.urlopen(config.SPEEDRUN_FORM_URL, data=data, timeout=20) as reply:
            sent = reply.status == 200
    except OSError as e:
        print(f"Could not reach the results form ({e}). Tell a presenter your result instead.")
        return False
    print(f"Sent: {nickname}, {result['ms_per_token']:.2f} ms per token." if sent else "The form did not accept the result.")
    return sent


# --- measuring tools -------------------------------------------------------------------------------
def time_sent(device, fn, repeats=20):
    """Milliseconds for fn(), with every operation sent from Python as usual."""
    fn()                                                                  # first run compiles kernels
    ttnn.synchronize_device(device)
    t0 = time.perf_counter()
    for _ in range(repeats):
        fn()
    ttnn.synchronize_device(device)
    return (time.perf_counter() - t0) / repeats * 1e3


def time_traced(device, fn, repeats=50):
    """Milliseconds for fn() when it is recorded once and replayed: the time the card itself needs.
    fn may only use tensors that are already on the card (no copies to or from the host)."""
    fn()                                                                  # first run compiles kernels
    trace = ttnn.begin_trace_capture(device, cq_id=0)
    fn()
    ttnn.end_trace_capture(device, trace, cq_id=0)
    ttnn.execute_trace(device, trace, cq_id=0, blocking=True)
    t0 = time.perf_counter()
    for _ in range(repeats):
        ttnn.execute_trace(device, trace, cq_id=0, blocking=False)
    ttnn.synchronize_device(device)
    ms = (time.perf_counter() - t0) / repeats * 1e3
    ttnn.release_trace(device, trace)
    return ms


def decode_layouts(device):
    """Memory layouts the single-token ("decode") attention operations expect, for this card.
    Returns (layout for the split heads, layout for merging the heads back)."""
    def height_sharded(cores):
        return ttnn.create_sharded_memory_config(shape=(32, 64), core_grid=cores, strategy=ttnn.ShardStrategy.HEIGHT,
                                                 orientation=ttnn.ShardOrientation.ROW_MAJOR,
                                                 use_height_and_width_as_shard_shape=True)
    if device.arch() == ttnn.device.Arch.BLACKHOLE:
        heads = height_sharded(ttnn.CoreGrid(y=4, x=8))
    else:
        heads = ttnn.L1_HEIGHT_SHARDED_MEMORY_CONFIG
    merge = height_sharded(ttnn.CoreRangeSet({ttnn.CoreRange(ttnn.CoreCoord(0, 0), ttnn.CoreCoord(0, 0))}))
    return heads, merge


def warm_decode_ops(model, device):
    """Run the toolbox's single-token attention operations once, so their kernels are compiled."""
    import math
    import torch
    H, dh, D = model.n_heads, model.d_head, model.d_model
    heads, merge = decode_layouts(device)
    tile = dict(layout=ttnn.TILE_LAYOUT, device=device)
    kc = ttnn.from_torch(torch.zeros(1, H, 1024, dh, dtype=torch.bfloat16), **tile)
    vc = ttnn.from_torch(torch.zeros(1, H, 1024, dh, dtype=torch.bfloat16), **tile)
    T = config.SPEEDRUN_PROMPT_TOKENS
    qkv = ttnn.from_torch(torch.randn(1, 1, T, 3 * D, dtype=torch.bfloat16), **tile)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads(qkv, num_heads=H, num_kv_heads=H, transpose_k_heads=False)
    ttnn.fill_cache(kc, k, 0)
    ttnn.fill_cache(vc, v, 0)
    a = ttnn.transformer.scaled_dot_product_attention(q, k, v, is_causal=True, scale=1 / math.sqrt(dh))
    ttnn.experimental.nlp_concat_heads(a)
    pos = ttnn.from_torch(torch.tensor([T], dtype=torch.int32), dtype=ttnn.int32, device=device)
    qkv1 = ttnn.from_torch(torch.randn(1, 1, 1, 3 * D, dtype=torch.bfloat16), **tile)
    q, k, v = ttnn.experimental.nlp_create_qkv_heads_decode(qkv1, num_heads=H, num_kv_heads=H, memory_config=heads)
    ttnn.experimental.paged_update_cache(kc, k, update_idxs_tensor=pos)
    ttnn.experimental.paged_update_cache(vc, v, update_idxs_tensor=pos)
    a = ttnn.transformer.scaled_dot_product_attention_decode(q, kc, vc, cur_pos_tensor=pos, scale=1 / math.sqrt(dh))
    ttnn.experimental.nlp_concat_heads_decode(ttnn.to_memory_config(a, merge), num_heads=H)
    ttnn.synchronize_device(device)
