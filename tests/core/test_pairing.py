"""The pairing rule, case by case."""

from rvc_next.core.models.pairing import G_TO_D, IndexCandidate, Named, VoiceCandidate, experiment_of_index, pair_indexes, pair_partners


def v(ref, name, version="v2", speakers=(), group=None):
    return VoiceCandidate(ref, name, version, tuple(speakers), group)


def i(ref, name, dim=768, ntotal=100, group=None):
    return IndexCandidate(ref, name, dim, ntotal, group)


def by_index(proposals):
    return {p.index: p for p in proposals}


def test_unrelated_names_pair_when_only_one_fits():
    p = pair_indexes([i("file:b", "my_final_index.index")], [v("file:a", "Kiki singing 2024.pth")])[0]
    assert (p.voice, p.status, p.confident, p.key) == ("file:a", "unique", True, "default")
    assert p.reasons[0] == "only_match"


def test_version_separates_two_voices_with_unrelated_names():
    props = by_index(pair_indexes([i("file:x", "foo.index", dim=256), i("file:y", "bar.index", dim=768)], [v("file:a", "alpha.pth", "v1"), v("file:b", "beta.pth", "v2")]))
    assert props["file:x"].voice == "file:a" and props["file:x"].status == "unique"
    assert props["file:y"].voice == "file:b" and props["file:y"].status == "unique"


def test_two_voices_same_version_unrelated_names_ask_the_user():
    props = pair_indexes([i("file:x", "foo.index"), i("file:y", "bar.index")], [v("file:a", "alpha.pth"), v("file:b", "beta.pth")])
    assert all(p.status == "choose" and p.voice is None and not p.confident for p in props)
    assert {c.voice for c in props[0].candidates} == {"file:a", "file:b"}


def test_rvc_names_pair_confidently_among_several():
    props = by_index(
        pair_indexes(
            [i("file:x", "added_IVF256_Flat_nprobe_1_alice_v2.index"), i("file:y", "added_IVF300_Flat_nprobe_1_bob_v2.index")],
            [v("file:a", "alice_e100_s2000.pth"), v("file:b", "bob.pth")],
        )
    )
    assert props["file:x"].voice == "file:a" and props["file:x"].status == "evidence" and props["file:x"].confident
    assert props["file:y"].voice == "file:b" and "experiment_name" in props["file:y"].reasons
    assert experiment_of_index("added_IVF256_Flat_nprobe_1_alice_v2_spkid3.index") == "alice"


def test_same_folder_decides_between_otherwise_equal_voices():
    props = by_index(pair_indexes([i("file:x", "index_a.index", group="pack1")], [v("file:a", "one.pth", group="pack1"), v("file:b", "two.pth", group="pack2")]))
    assert props["file:x"].voice is None  # 0.2 alone is below the proposing score
    props = by_index(pair_indexes([i("file:x", "kiki_v2.index", group="pack1")], [v("file:a", "kiki.pth", group="pack1"), v("file:b", "kiki2.pth", group="pack2")]))
    assert props["file:x"].voice == "file:a" and "same_folder" in props["file:x"].reasons


def test_empty_and_incompatible_indexes():
    props = by_index(
        pair_indexes([i("file:t", "trained_IVF1_Flat_x_v2.index", ntotal=0), i("file:w", "odd.index", dim=512), i("file:z", "z.index", dim=256)], [v("file:a", "a.pth", "v2")])
    )
    assert props["file:t"].status == "empty" and props["file:t"].voice is None
    assert props["file:w"].status == "incompatible"
    assert props["file:z"].status == "incompatible" and props["file:z"].reasons == ["no_v1_voice"]


def test_speaker_indexes_get_speaker_keys():
    voice = v("file:a", "multi.pth", speakers=(0, 3))
    props = by_index(pair_indexes([i("file:x", "added_IVF1_Flat_nprobe_1_multi_v2_spkid3.index"), i("file:y", "added_IVF1_Flat_nprobe_1_multi_v2_spkid0.index")], [voice]))
    assert props["file:x"].key == "spk3" and props["file:y"].key == "spk0"
    assert props["file:x"].voice == props["file:y"].voice == "file:a"


def test_duplicates_leave_the_weaker_for_the_user():
    props = by_index(pair_indexes([i("file:x", "added_IVF1_Flat_nprobe_1_kiki_v2.index"), i("file:y", "kiki_old.index")], [v("file:a", "kiki.pth")]))
    assert props["file:x"].voice == "file:a"
    assert props["file:y"].voice is None and props["file:y"].status == "duplicate"


def test_a_lone_index_is_proposed_to_library_voices_but_never_confident():
    library = [v("voice:kiki", "kikiV1", "v1"), v("voice:keruan", "keruanV1", "v1")]
    p = pair_indexes([i("file:x", "added_IVF1_Flat_nprobe_1_kikiV1_v1.index", dim=256)], library)[0]
    assert p.voice == "voice:kiki" and p.status == "evidence" and not p.confident
    p = pair_indexes([i("file:x", "random.index", dim=256)], library)[0]
    assert p.status == "choose" and len(p.candidates) == 2


def test_partners():
    g = [Named("file:g", "f0G40k.pth", "v2"), Named("file:h", "gang_G_2333333.pth", "v2")]
    d = [Named("file:d", "f0D40k.pth", "v2"), Named("file:e", "gang_D_2333333.pth", "v2")]
    props = {p.left: p for p in pair_partners(g, d, swap=G_TO_D, same_version=True)}
    assert props["file:g"].right == "file:d" and props["file:h"].right == "file:e"
    one = pair_partners([Named("file:c", "whatever.ckpt")], [Named("file:y", "config.yaml")])
    assert one[0].right == "file:y" and one[0].status == "unique"
