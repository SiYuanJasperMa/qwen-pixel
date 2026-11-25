import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'scripts'))
from build_paired_method_sft import freeze, variant


def test_ti_replaces_answer_without_changing_task_inputs_or_count():
    pairs=[{'key':str(i),'rft':{'id':f'{i}_rft','image':f'{i}.png','instruction':'Edit it'},
            'ti':{'id':f'{i}_ti','image':f'{i}.png','instruction':'Edit it'}} for i in range(16)]
    a,b,c=variant(pairs,0),variant(pairs,25),variant(pairs,50)
    assert len(a)==len(b)==len(c)==16
    assert sum(r['id'].endswith('_ti')for r in b)==4
    assert sum(r['id'].endswith('_ti')for r in c)==8
    assert {r['id'].split('_')[0]for r in b if r['id'].endswith('_ti')} <= \
        {r['id'].split('_')[0]for r in c if r['id'].endswith('_ti')}
    assert [r['image']for r in a]==[r['image']for r in b]==[r['image']for r in c]


def test_frozen_snapshot_refuses_changed_content(tmp_path):
    p=tmp_path/'snapshot.jsonl'
    freeze(p,b'original');freeze(p,b'original')
    try:freeze(p,b'changed')
    except ValueError:pass
    else:raise AssertionError('Changed snapshot accepted')
    assert p.read_bytes()==b'original'
